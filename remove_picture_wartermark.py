import os
import cv2
import numpy as np
from pathlib import Path

import argparse
import sys

# ================== 配置区 ==================
# 原始输入图片文件存放目录
INPUT_DIR = "input"
# 中间生成的水印图片文件存放目录
MASK_DIR = "masks"
# 目标输出图片文件存放目录
OUTPUT_DIR = "output"

EXTEND_BORDER_PX = 10
threshold = 20
white_threshold = 220
feather = 0
EXPORT_DIR = "export"

# 图片尺寸（必须统一）
WIDTH = 2733
HEIGHT = 1534

# 水印区域大小（像素）
WATERMARK_W = 288
WATERMARK_H = 90

# 水印位置：left_top / right_bottom
WATERMARK_POS = "right_bottom"

# ==========================================

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_mask(position: str):
    """生成黑白 mask"""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if position == "left_top":
        x, y = 20, 20
    elif position == "right_bottom":
        x = WIDTH - WATERMARK_W - 20
        y = HEIGHT - WATERMARK_H - 20
    else:
        raise ValueError("position must be left_top or right_bottom")

    mask[y:y + WATERMARK_H, x:x + WATERMARK_W] = 255
    return mask


def generate_masks():
    """为所有图片生成 mask"""
    mask = create_mask(WATERMARK_POS)
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", "webp")):
            cv2.imwrite(os.path.join(MASK_DIR, fname), mask)


def remove_white_background(input_path, output_path, color_threshold=30,
                            white_threshold=220, feather_radius=0):
    """
    自动识别并去除图片中的白色背景/边框，生成透明背景图

    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径（建议PNG格式以保留透明度）
        color_threshold: 颜色容差，值越大允许的背景色范围越广（默认30）
        white_threshold: 种子点的白色判定阈值，各通道大于此值才认为是白色背景起点（默认220）
        feather_radius: 边缘羽化半径，0表示不羽化（默认0）

    返回:
        bool: 处理成功返回True，失败返回False
    """
    # 读取图片，保留原始通道（含透明度）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"错误：无法读取图片 {input_path}")
        return False

    # 确保图片为BGRA格式（含Alpha通道）
    if img.shape[2] == 3:
        # 添加完全不透明的Alpha通道
        bgr = img.copy()
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        img = cv2.merge([bgr, alpha])
    elif img.shape[2] == 4:
        # 已有Alpha通道，分离BGRA
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        print("不支持的图片格式")
        return False

    h, w = bgr.shape[:2]

    # 检查四个角点是否足够“白”，决定是否执行背景移除
    corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    is_white_bg = False
    for x, y in corners:
        pixel = bgr[y, x]
        if pixel[0] > white_threshold and pixel[1] > white_threshold and pixel[2] > white_threshold:
            is_white_bg = True
            break

    if not is_white_bg:
        print("警告：未检测到白色背景，原图将直接保存（无透明效果）")
        cv2.imwrite(output_path, img)
        return True

    # 准备用于背景标记的mask（floodFill需要比原图大2的边界）
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # 复制图像用于floodFill操作（避免修改原图）
    img_cpy = bgr.copy()

    # 设置颜色容差（低/高差异，基于种子点颜色）
    lo_diff = (color_threshold, color_threshold, color_threshold)
    up_diff = (color_threshold, color_threshold, color_threshold)

    # 从四个角点进行泛洪填充，标记背景区域
    seeds = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    for seed_x, seed_y in seeds:
        # 检查种子点是否已经是背景标记区域
        if mask[seed_y + 1, seed_x + 1] != 0:
            continue
        # 检查种子点颜色是否符合白色预期（可选，提高鲁棒性）
        seed_color = bgr[seed_y, seed_x]
        if seed_color[0] < white_threshold or seed_color[1] < white_threshold or seed_color[2] < white_threshold:
            continue

        # 执行泛洪填充，填充的区域会在mask中标记（非零）
        cv2.floodFill(img_cpy, mask, (seed_x, seed_y), (0, 0, 255),
                      lo_diff, up_diff, cv2.FLOODFILL_MASK_ONLY)

    # 从mask中提取背景区域（mask大小 h+2 x w+2，有效区域为 [1:h+1, 1:w+1]）
    bg_mask = mask[1:h + 1, 1:w + 1] > 0

    # 根据背景掩码设置Alpha通道
    new_alpha = alpha.copy()
    new_alpha[bg_mask] = 0  # 背景区域完全透明

    # 可选：边缘羽化处理，使边缘过渡更自然
    if feather_radius > 0:
        # 对透明边界进行高斯模糊
        blurred_alpha = cv2.GaussianBlur(new_alpha.astype(np.float32),
                                         (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        new_alpha = blurred_alpha.astype(np.uint8)
        # 由于模糊可能导致非背景区域变半透明，重新确保原背景区域为0，非背景区域保持较高不透明度
        # 实际上简单羽化会改变前景边缘，此处仅做轻量处理
        new_alpha[bg_mask] = 0
        # 对非背景区域，如果模糊后<255但原本应该不透明，可以阈值化保边
        new_alpha[~bg_mask] = np.maximum(new_alpha[~bg_mask], 200)

    # 合并BGRA通道
    result = cv2.merge([bgr, new_alpha])

    # 保存结果（PNG格式自动保留透明度）
    success = cv2.imwrite(output_path, result)
    if success:
        print(f"处理成功！透明背景图片已保存至: {output_path}")
    else:
        print(f"保存失败: {output_path}")

    return success


def main():
    '''
    parser = argparse.ArgumentParser(description='自动识别并去除图片中的白色背景/边框，生成透明背景图')
    parser.add_argument('input', help='输入图片路径')
    parser.add_argument('-o', '--output', default=None, help='输出图片路径（建议PNG格式）')
    parser.add_argument('-t', '--threshold', type=int, default=30,
                        help='颜色容差，值越大背景去除范围越广（默认30）')
    parser.add_argument('-w', '--white_threshold', type=int, default=220,
                        help='白色判定阈值，角点各通道大于此值才启动去除（默认220）')
    parser.add_argument('-f', '--feather', type=int, default=0,
                        help='边缘羽化半径，0表示不羽化（默认0）')

    args = parser.parse_args()
    '''
    # 自动生成输出文件名
    # 检查输入文件
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ 找不到输入文件: {OUTPUT_DIR}")
        return

    if not os.path.exists(EXPORT_DIR):
        print(f"❌ 找不到输出文件: {EXPORT_DIR}")
        return

    for fname in os.listdir(OUTPUT_DIR):
        if fname.lower().endswith((".png")):
            print("****************待处理图片：fname=",fname)
            input_file_name = os.path.join(OUTPUT_DIR, fname)
            output_file_name = os.path.join(EXPORT_DIR, fname)
            success = remove_white_background(
                input_file_name, output_file_name,
                color_threshold=threshold,
                white_threshold=white_threshold,
                feather_radius=feather
            )


def run_iopaint():
    """调用 IOPaint（CPU）"""
    os.system(
        f"iopaint run "
        f"--model lama "
        f"--device cpu "
        f"--image {INPUT_DIR} "
        f"--mask {MASK_DIR} "
        f"--output {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    print("🚀 生成 Mask...")
    generate_masks()

    print("🧠 开始 LaMa 修复（CPU，请耐心等待）...")
    run_iopaint()

    main()
    print("✅ 全部完成！结果在 output 目录")
