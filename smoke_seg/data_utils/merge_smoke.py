import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_merge_image(arg):
    base_img_path, smoke_img_path, concentration = arg
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
    smoke_result.save(os.path.join(save_merge_image_gt_dir, filename + '.png'))


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
    """
    批量融合，并且生成gt图
    """
    source_dir = "D:/vir_smoke/all_images"
    smoke_dir = "D:/vir_smoke/all_smoke"

    image_list = [f for f in os.listdir(source_dir)]
    smoke_list = [f for f in os.listdir(smoke_dir)]
    smoke_num = len(smoke_list)
    task_args = []
    for i, image in enumerate(image_list):
        img_path = os.path.join(source_dir, image)
        # index = random.randint(0, len(smoke_list) - 1) # 随机取值
        index = i % smoke_num
        # concentration = random.uniform(0.8, 1.5)  # 随机浓度
        smoke_path = os.path.join(smoke_dir, smoke_list[index])
        # save_merge_image(img_path, smoke_path, round(concentration, 2))
        task_args.append((img_path, smoke_path, 1))
        # save_merge_image(img_path, smoke_path, 1)

    # 多进程
    progress = tqdm(total=len(image_list), desc="Processing Images", unit="img")
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for args in task_args:
            future = executor.submit(save_merge_image, args)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)
    progress.close()
