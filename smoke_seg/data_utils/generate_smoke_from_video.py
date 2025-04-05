import os

import cv2


def video_to_frames(video_path, output_dir="D:/raw_smoke1"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 从视频路径提取文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")

    frame_count = 0
    i = 0
    num_frames = 30   # 每多少帧保存一次
    while True:
        ret, frame = cap.read()
        i += 1
        if not ret:
            break  # 视频结束

        # 生成带补零的序号（如000001）
        frame_id = f"{frame_count:06d}"

        # 构建输出路径
        output_filename = f"video_{video_name}_smoke_{frame_id}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 保存帧为JPG图片
        if i % num_frames == 0:
            cv2.imwrite(output_path, frame)
        frame_count += 1

    cap.release()
    print(f"转换完成！共生成 {frame_count} 帧图片，保存至: {output_dir}")


if __name__ == "__main__":
    base_path = "D:/BaiduNetdiskDownload/smoke_video/smoke_video3";
    video_paths = os.listdir(base_path)
    for path in video_paths:
        video_path = base_path + "/" + path
        video_to_frames(video_path)
