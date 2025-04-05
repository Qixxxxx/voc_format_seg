import os
import random

import cv2
import numpy as np
from PIL import Image


def save_merge_image(base_img_path, smoke_img_path, concentration):
    base_img = cv2.imread(base_img_path)
    smoke_img = cv2.imread(smoke_img_path)
    image = overlay_smoke(base_img, smoke_img, concentration)
    save_merge_image_dir = "D:/vir_smoke/merge_images"
    os.makedirs(save_merge_image_dir, exist_ok=True)
    image.save(os.path.join(save_merge_image_dir, os.path.basename(base_img_path)))

    save_merge_image_gt_dir = "D:/vir_smoke/merge_images_gt"
    os.makedirs(save_merge_image_gt_dir, exist_ok=True)
    smoke = Image.fromarray(smoke_img)
    r, g, b = smoke.split()
    smoke_result = Image.merge("RGB", (b, g, r))
    filename, _ = os.path.splitext(os.path.basename(base_img_path))
    smoke_result.save(os.path.join(save_merge_image_gt_dir, filename+'.png'))



def overlay_smoke(base_img, smoke_img, concentration=1.0):
    # 确保烟雾图与底图尺寸相同
    if smoke_img.shape != base_img.shape:
        smoke_img = cv2.resize(smoke_img, (base_img.shape[1], base_img.shape[0]))

    # 将图像转换为浮点数以便计算
    base = base_img.astype(np.float32) / 255.0
    smoke = smoke_img.astype(np.float32) / 255.0

    # 创建alpha蒙版（烟雾非零区域）
    alpha = cv2.cvtColor(smoke_img, cv2.COLOR_BGR2GRAY)
    alpha = np.clip(concentration * alpha.astype(np.float32), 0.0, 255.0) / 255.0
    alpha = np.expand_dims(alpha, axis=-1)  # 增加通道维度

    # 进行alpha混合
    blended = smoke * alpha + base * (1.0 - alpha)

    # 将结果转换回0-255范围
    blended = (blended * 255).astype(np.uint8)
    result = Image.fromarray(blended)
    r, g, b = result.split()
    rgb_result = Image.merge("RGB", (b, g, r))
    return rgb_result


if __name__ == '__main__':
    # base = cv2.imread("C:/Users/Qx/Downloads/wallhaven-72v1go_3840x2160.png")
    # smoke = cv2.imread("D:/raw_smoke/video_video_104_smoke_000417.jpg")
    # rgb_result = overlay_smoke(base, smoke, 0.8)
    # # 展示
    # rgb_result.show()

    # # 创建输出目录
    # output_dir = "D:/virtual_smoke_data"
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, "result1.jpg")
    # rgb_result.save(output_path)

    """
    批量融合，并且生成gt图
    """
    source_dir = "D:/vir_smoke/all_images"
    smoke_dir = "D:/vir_smoke/all_smoke"

    image_list = [f for f in os.listdir(source_dir)]
    smoke_list = [f for f in os.listdir(smoke_dir)]

    for image in image_list:
        img_path = os.path.join(source_dir, image)
        random_num = random.randint(0, len(smoke_list) - 1)
        # concentration = random.uniform(0.8, 1.5)  # 随机浓度
        smoke_path = os.path.join(smoke_dir, smoke_list[random_num])
        # save_merge_image(img_path, smoke_path, round(concentration, 2))
        save_merge_image(img_path, smoke_path, 1)

