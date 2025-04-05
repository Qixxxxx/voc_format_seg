import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from tqdm import tqdm


def process_single_file(args):
    """
    并发处理单个文件的函数
    """
    filename, source_dir, target_dir, target_size = args
    try:
        # 构造完整路径
        src_path = os.path.join(source_dir, filename)
        new_name = os.path.splitext(filename)[0] + ".jpg"
        dst_path = os.path.join(target_dir, new_name)

        # 读取图片（支持中文路径）
        img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, f"{filename}: 无法读取图片"

        # 执行缩放
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

        # # 处理透明通道
        # ext = os.path.splitext(filename)[1].lower()
        # if img.shape[2] == 4 and ext != '.png':
        #     dst_path = os.path.splitext(dst_path)[0] + '.png'

        # 保存文件（保留元数据）
        success, encoded = cv2.imencode(".jpg", resized_img, [
            int(cv2.IMWRITE_JPEG_QUALITY), 95,
            int(cv2.IMWRITE_PNG_COMPRESSION), 3
        ])
        if success:
            encoded.tofile(dst_path)
            return True, filename
        return False, f"{filename}: 保存失败"

    except Exception as e:
        return False, f"{filename}: {str(e)}"


def concurrent_resize(source_dir, target_dir, target_size=(256, 256), workers=10):
    """
    并发版图片尺寸调整
    :param workers: 并发进程数
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 获取文件列表
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = [
        f for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    # 创建进度条
    progress = tqdm(total=len(files), desc="Processing Images", unit="img")

    # 使用上下文管理器管理进程池
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 生成任务参数
        task_args = [(f, source_dir, target_dir, target_size) for f in files]

        # 提交任务并立即开始处理
        futures = []
        for args in task_args:
            future = executor.submit(process_single_file, args)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)

        # 收集结果并处理异常
        results = []
        for future in futures:
            success, msg = future.result()
            results.append((success, msg))
 
    progress.close()

    # 输出统计信息
    success_count = sum(1 for r in results if r[0])
    error_logs = [r[1] for r in results if not r[0]]

    print(f"\n处理完成: {success_count}/{len(files)} 成功")
    if error_logs:
        print("\n错误日志:")
        print("\n".join(error_logs[:5]))  # 最多显示5条错误


if __name__ == "__main__":
    # 配置参数
    source_folder = "D:/raw_smoke"  # 替换为源文件夹路径
    target_folder = "D:/vir_smoke/all_smoke"  # 替换为目标文件夹路径
    WORKERS = 20  # 根据CPU核心数调整

    # 执行并发处理
    concurrent_resize(source_folder, target_folder, workers=WORKERS)
