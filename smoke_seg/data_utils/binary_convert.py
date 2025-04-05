import cv2
import os
import numpy as np

def binary_convert(src_dir, dst_dir, threshold=1):
    """
    灰度图二值化处理函数
    :param src_dir: 源目录路径
    :param dst_dir: 目标目录路径
    :param threshold: 判断"有值"的阈值（默认>=1视为有值）
    """
    # 支持的图片格式
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历源目录
    for filename in os.listdir(src_dir):
        # 检查文件格式
        filepath = os.path.join(src_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in valid_ext:
            continue

        try:
            # 读取灰度图（强制转为单通道）
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("无法读取图像文件")

            # 执行二值化转换
            # 方法1：使用NumPy向量化操作（更快）
            binary = np.where(img >= threshold, 255, 0).astype(np.uint8)

            # 方法2：使用OpenCV阈值函数（更灵活）
            # _, binary = cv2.threshold(img, threshold-1, 255, cv2.THRESH_BINARY)

            # 保留原始元数据
            params = []
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, 100]  # 最高质量
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 无压缩

            # 保存结果
            dst_path = os.path.join(dst_dir, filename)
            cv2.imwrite(dst_path, binary, params)

            print(f"已处理: {filename} ({img.shape[1]}x{img.shape[0]})")

        except Exception as e:
            print(f"处理失败: {filename} - {str(e)}")


if __name__ == "__main__":
    # 配置路径
    SOURCE_DIR = "D:/vir_smoke/merge_images_gt"  # 源目录
    TARGET_DIR = "D:/vir_smoke/merge_images_gt_gray"  # 目标目录

    # 执行转换（调整threshold参数控制灵敏度）
    binary_convert(SOURCE_DIR, TARGET_DIR, threshold=10)
