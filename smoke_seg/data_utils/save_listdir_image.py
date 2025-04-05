import os
import shutil


def collect_images(src_dir, dst_dir, extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    """
    将多层级目录中的图片收集到单个文件夹
    参数：
        src_dir: 源目录路径
        dst_dir: 目标目录路径
        extensions: 支持的图片扩展名元组
    """
    # 创建目标目录（如果不存在）
    os.makedirs(dst_dir, exist_ok=True)

    # 记录复制数量
    count = 0

    # 遍历源目录
    for root, _, files in os.walk(src_dir):
        for filename in files:
            # 检查文件扩展名
            if filename.lower().endswith(extensions):
                # 构造完整路径
                src_path = os.path.join(root, filename)
                dst_path = os.path.join(dst_dir, filename)

                # 执行复制（使用shutil.copy2保留元数据）
                shutil.copy2(src_path, dst_path)
                count += 1

    print(f"共复制 {count} 张图片到 {dst_dir}")


# 使用示例
if __name__ == "__main__":
    source_directory = "D:/vir_smoke/raw/raw_cityscapes/cityscapes/leftImg8bit"  # 修改为你的源目录
    destination_directory = "D:/vir_smoke/temp"  # 修改为目标目录

    collect_images(source_directory, destination_directory)
