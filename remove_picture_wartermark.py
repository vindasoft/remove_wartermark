import os
import cv2
import numpy as np
from pathlib import Path

# ================== 配置区 ==================
# 原始输入图片文件存放目录
INPUT_DIR = "input"
# 中间生成的水印图片文件存放目录
MASK_DIR = "masks"
# 目标输出图片文件存放目录
OUTPUT_DIR = "output"

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

    print("✅ 全部完成！结果在 output 目录")