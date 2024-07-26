import cv2
import os

'''
func: 从视频中提取帧
'''

def extract_frames_from_video(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        output_file = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(output_file, frame)
        frame_count += 1

    video.release()
    print(f"Extracted {frame_count} frames and saved them in {output_folder}")

if __name__ == "__main__":
    video_path = "/home/ywh/Videos/zhuitong.mp4"
    output_folder = "/media/ywh/1/yanweihao/dataset/ivfc/zhuitong/output_frames"

    extract_frames_from_video(video_path, output_folder)

