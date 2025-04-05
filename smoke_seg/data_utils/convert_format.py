from PIL import Image
import os


def convert_png_to_jpg(input_folder, output_folder=None, quality=95):
    """
    将指定文件夹内的 PNG 图片转换为 JPG 格式
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径（默认与输入文件夹相同）
    :param quality: JPG 质量（1-100），默认95
    """
    # 设置默认输出路径
    output_folder = output_folder or input_folder

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有PNG文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            # 处理文件路径
            png_path = os.path.join(input_folder, filename)
            jpg_name = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(output_folder, jpg_name)

            try:
                # 打开PNG图像并转换为RGB模式（移除Alpha通道）
                with Image.open(png_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG', quality=quality)
                print(f"已转换: {filename} -> {jpg_name}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")


if __name__ == '__main__':
    convert_png_to_jpg(
        input_folder='D:/vir_smoke/temp',
        output_folder='D:/vir_smoke/t',
        quality=100  # 可选质量参数
    )
