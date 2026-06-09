import cv2
import numpy as np
import subprocess
import os
import requests
import time
import base64
import sys
import concurrent.futures
from collections import OrderedDict

# ================= 配置参数 =================
# 原始视频文件存放目录
INPUT_DIR = "input"
# 水印文件存放目录
MASK_DIR = "masks"
# 临时视频文件
TEMP_NO_AUDIO = "temp_na.mp4"
# 目标视频文件输出目录
OUTPUT_DIR = "output"

TEMP_NO_AUDIO_FILE = os.path.join(MASK_DIR, TEMP_NO_AUDIO)

# 视频屏幕水印区域列表 (x, y, w, h) - 多个矩形
WATERMARK_RECTS = [
    (1050, 623, 222, 77),  # 右下角水印
    (23, 26, 222, 77),  # 左上角水印
]
MASK_EXPAND = 8

# 视频处理速度优化配置
USE_SCALE = True  # 是否启用缩放处理，设为 False 则使用原始分辨率
SCALE_FACTOR = 0.5  # 缩放系数 (0.5 表示将长宽都缩小一半)
# 本地小模型API地址：
API_URL = "http://localhost:8080/api/v1/inpaint"
# 本地小模型CPU启动命令 iopaint start --model=lama --device=cpu --port=8080 --cpu-offload
# 本地小模型CUDA启动命令 iopaint start --model=lama --device=cuda --port=8080
API_TIMEOUT = 200  # API 请求超时时间（秒）

# 并发配置
MAX_WORKERS = 6  # 并发线程数，建议设置为 CPU 核心数
BUFFER_SIZE = 10  # 缓冲区大小，用于防止内存溢出


# ===========================================
def image_to_base64(image):
    """
    将 OpenCV 图像（numpy.ndarray）编码为 PNG 格式的 Base64 字符串。
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def send_inpaint_request(payload, original_frame, frame_idx, use_scale, original_size):
    """
    发送修复请求到后端 API
    """
    try:
        # 发送 POST 请求
        resp = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
        resp.raise_for_status()

        # 解析返回的二进制图片数据
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        result_small = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if result_small is None:
            raise ValueError(f"第 {frame_idx + 1} 帧解码修复后的图像失败")

        # 若之前做了缩放，将修复后的图像放大回原始尺寸
        if use_scale:
            result_frame = cv2.resize(result_small, original_size, interpolation=cv2.INTER_LINEAR)
        else:
            result_frame = result_small

        return frame_idx, result_frame, None

    except requests.exceptions.RequestException as e:
        return frame_idx, original_frame, f"请求失败: {e}"
    except Exception as e:
        return frame_idx, original_frame, f"处理异常: {e}"


def process_video_parallel(file_name):
    """
    并行处理视频帧
    """
    # 1. 检查后端服务
    try:
        requests.options(API_URL, timeout=5)
        print("✅ 后端服务可达")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到后端，请确保已运行命令：iopaint start --model=lama --device=cpu --port=8080 --cpu-offload")
        sys.exit(1)
    except Exception:
        print("⚠️ 无法确认后端状态，将尝试继续...")

    # 2. 读取视频并获取基本信息
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {file_name}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if USE_SCALE:
        # 计算缩放后的尺寸，并保证宽度和高度为偶数
        width = int(width_original * SCALE_FACTOR)
        height = int(height_original * SCALE_FACTOR)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        print(f"⚙️ 启用缩放加速: {width_original}x{height_original} -> {width}x{height}")
    else:
        width, height = width_original, height_original
        print("⚙️ 使用原始分辨率")

    print(f"📹 视频信息: {total_frames} 帧, {fps:.2f} fps")

    # 3. 创建掩码（只需要创建一次，所有帧共用）
    print("🎨 创建水印掩码...")
    mask_base = np.zeros((height, width), dtype=np.uint8)

    for (wx, wy, ww, wh) in WATERMARK_RECTS:
        if USE_SCALE:
            x1 = int(wx * SCALE_FACTOR)
            y1 = int(wy * SCALE_FACTOR)
            x2 = x1 + int(ww * SCALE_FACTOR)
            y2 = y1 + int(wh * SCALE_FACTOR)
        else:
            x1, y1 = wx, wy
            x2, y2 = wx + ww, wy + wh

        # 边界裁剪
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:
            mask_base[y1:y2, x1:x2] = 255

    # 膨胀掩码
    kernel = np.ones((MASK_EXPAND * 2 + 1, MASK_EXPAND * 2 + 1), np.uint8)
    mask = cv2.dilate(mask_base, kernel, iterations=1)
    mask_base64 = image_to_base64(mask)
    print("✅ 掩码创建完成")

    # 4. 设置视频写入器
    writer = cv2.VideoWriter(TEMP_NO_AUDIO_FILE, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width_original, height_original))

    # 5. 并行处理视频帧
    print(f"🎬 开始并行处理视频帧 (并发线程数: {MAX_WORKERS})...")

    # 创建线程池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # 用于存储结果的字典和队列
    frame_buffer = OrderedDict()
    next_frame_to_write = 0
    failed_frames = 0
    start_time = time.time()

    # 重新打开视频读取器，因为之前读取了一些帧用于获取信息
    cap.release()
    cap = cv2.VideoCapture(file_name)

    # 提交所有任务
    future_to_frame = {}
    print("📤 提交所有帧处理任务...")

    for frame_idx in range(total_frames):
        ret, frame_original = cap.read()
        if not ret:
            print(f"⚠️ 读取第 {frame_idx + 1} 帧失败，停止提交")
            total_frames = frame_idx
            break

        # 缩放当前帧
        if USE_SCALE:
            frame = cv2.resize(frame_original, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            frame = frame_original

        # 转为 Base64
        frame_base64 = image_to_base64(frame)

        # 构建 JSON 请求体
        payload = {
            "image": frame_base64,
            "mask": mask_base64,
            # ================= 新增以下三个参数 =================
            "hd_strategy": "Crop",  # 1. 开启精修模式 (Crop/Resize/Original)
            "hd_strategy_crop_margin": 64,  # 2. 修复区域向外扩展像素，提供上下文
            "sd_mask_blur": 5,  # 3. 模糊遮罩边缘，消除生硬的“方块感”
            # =================================================
        }

        # 提交任务到线程池
        future = executor.submit(
            send_inpaint_request,
            payload,
            frame_original,
            frame_idx,
            USE_SCALE,
            (width_original, height_original)
        )
        future_to_frame[future] = frame_idx

    cap.release()

    # 6. 收集结果并按顺序写入
    print("📥 收集处理结果并写入视频...")

    for future in concurrent.futures.as_completed(future_to_frame):
        frame_idx, result_frame, error = future.result()

        if error:
            failed_frames += 1
            print(f"❌ 第 {frame_idx + 1} 帧处理失败: {error}")
        else:
            # 成功处理
            pass

        # 存入缓冲区
        frame_buffer[frame_idx] = result_frame

        # 按顺序写入已完成的帧
        while next_frame_to_write in frame_buffer:
            writer.write(frame_buffer.pop(next_frame_to_write))
            next_frame_to_write += 1

            # 进度显示
            if next_frame_to_write % 30 == 0 or next_frame_to_write == total_frames:
                elapsed = time.time() - start_time
                frames_done = next_frame_to_write
                if frames_done > 0:
                    avg_time = elapsed / frames_done
                    remaining_frames = total_frames - frames_done
                    eta = remaining_frames * avg_time
                    print(f"⏳ 进度: {frames_done}/{total_frames} 帧 "
                          f"({frames_done / total_frames * 100:.1f}%), "
                          f"平均 {avg_time:.2f} 秒/帧, "
                          f"预计剩余 {eta:.0f} 秒")

        # 限制缓冲区大小，防止内存溢出
        if len(frame_buffer) > BUFFER_SIZE:
            print(f"⚠️ 缓冲区过大 ({len(frame_buffer)}), 等待写入...")
            while len(frame_buffer) > BUFFER_SIZE // 2:
                time.sleep(0.1)

    # 确保所有帧都已写入
    while next_frame_to_write < total_frames:
        if next_frame_to_write in frame_buffer:
            writer.write(frame_buffer.pop(next_frame_to_write))
            next_frame_to_write += 1
        else:
            # 理论上不应该出现，如果出现说明有帧丢失
            print(f"⚠️ 警告：第 {next_frame_to_write + 1} 帧丢失，跳过")
            next_frame_to_write += 1

    writer.release()
    executor.shutdown(wait=True)

    elapsed_time = time.time() - start_time
    print(f"\n📊 处理完成！")
    print(f"   - 总帧数: {total_frames}")
    print(f"   - 成功: {total_frames - failed_frames}")
    print(f"   - 失败: {failed_frames}")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒")
    print(f"   - 平均速度: {elapsed_time / total_frames:.2f} 秒/帧")

    return total_frames - failed_frames > 0


def merge_audio(file_name):
    """
    合并原视频的音频到处理后的视频
    """
    print("🎵 开始合并音频...")

    print("TEMP_NO_AUDIO_FILE=", TEMP_NO_AUDIO_FILE)
    if not os.path.exists(TEMP_NO_AUDIO_FILE) or os.path.getsize(TEMP_NO_AUDIO_FILE) == 0:
        print(f"❌ 临时视频文件无效，跳过音频合并")
        return False
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, file_name)
    print("OUTPUT_FILE=",OUTPUT_FILE)
    # 优化的 FFmpeg 参数，在保持画质的同时加快速度
    cmd = [
        'ffmpeg', '-y',
        '-i', TEMP_NO_AUDIO_FILE,
        '-i', file_name,
        '-map', '0:v',
        '-map', '1:a?',  # '?' 表示如果没音频就不报错
        '-c:v', 'libx264',
        '-crf', '16',  # 提高画质（原20改为18）
        '-preset', 'slower',  # 保持快速编码
        '-c:a', 'aac',  # 音频编码
        '-b:a', '192k',  # 音频码率
        '-shortest',
        OUTPUT_FILE
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 最终视频已保存至: {OUTPUT_FILE}")

        # 删除临时文件
        os.remove(TEMP_NO_AUDIO_FILE)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 音频合并失败: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("❌ 未找到 ffmpeg，请确保已安装 FFmpeg")
        return False


def main():
    """
    主函数
    """
    print("=" * 60)
    print("🎬 视频水印去除工具 (并行优化版)")
    print("=" * 60)
    # 检查输入文件
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入文件: {INPUT_DIR}")
        return

    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith((".mp4")):
            print("fname=",fname)
            # 处理视频
            success = process_video_parallel(fname)

            if success:
                # 合并音频
                merge_audio(fname)
            else:
                print("❌ 视频处理失败，请检查错误信息")




if __name__ == "__main__":
    main()