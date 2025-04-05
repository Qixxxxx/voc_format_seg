import os


def batch_rename(directory):
    """
    批量重命名目录下文件
    :param directory: 目录路径
    """
    # 获取目录下所有文件（不区分大小写）
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(".mov")      # 打开指定的文件后缀
             and os.path.isfile(os.path.join(directory, f))]

    if not files:
        print("未找到文件")
        return

    # 按文件名排序（如需按修改时间排序请修改此处）
    files.sort()

    # 执行重命名
    success_count = 0
    for index, filename in enumerate(files, start=1):
        # 生成新文件名
        new_name = f"smoke_plume_{index:04d}.mov"     # 新名字
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        # 跳过无需修改的文件
        if old_path == new_path:
            continue

        try:
            os.rename(old_path, new_path)
            success_count += 1
            print(f"[成功] {filename} -> {new_name}")
        except Exception as e:
            print(f"[失败] {filename} -> {new_name} | 错误：{str(e)}")

    print(f"\n操作完成！成功处理 {success_count}/{len(files)} 个文件")


if __name__ == "__main__":
    # 设置需要处理的目录路径（当前目录）
    target_dir = "D:/BaiduNetdiskDownload/smoke_video/smoke_video3"

    # 验证目录有效性
    if not os.path.isdir(target_dir):
        print(f"错误：目录不存在 {target_dir}")
    else:
        batch_rename(target_dir)