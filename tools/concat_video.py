import cv2
import numpy as np
import argparse

class VideoConcatenator:
    def __init__(self, inputs, interval, output):
        self.inputs = inputs
        self.interval = interval
        self.output = output
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
        self.interval_frames = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    def get_video_info(self, cap):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, total_frames
    
    def concatenate(self):
        # 打开输入视频文件
        caps = []
        for input_file in self.inputs:
            caps.append(cv2.VideoCapture(input_file))

        # 获取视频的帧率和尺寸
        self.fps, self.width, self.height, self.total_frames = self.get_video_info(caps[0])
        self.interval_frames = int(self.interval * self.fps)
        total_inputs = len(self.inputs)

        # 创建输出视频文件
        out = cv2.VideoWriter(self.output, self.fourcc, self.fps, (total_inputs*self.width+(total_inputs-1)*self.interval, self.height))

        # 逐帧读取多个视频，并水平拼接
        for i in range(self.total_frames):
            frames_concat = []
            for j in range(total_inputs):
                ret, frame = caps[j].read()
                if ret:
                    frames_concat.append(frame)
                else:
                    break
            
            if len(frames_concat) == total_inputs:
                frame_concat = frames_concat[0]
                for k in range(1, total_inputs):
                    frame_concat = np.concatenate((frame_concat, np.zeros((self.height, self.interval, 3), np.uint8), frames_concat[k]), axis=1)
                out.write(frame_concat)
            else:
                break

        # 释放资源并关闭输出视频文件
        for cap in caps:
            cap.release()
        out.release()

        print('Done!')

def main(args):
    vc = VideoConcatenator(args.inputs, args.interval, args.output)
    vc.concatenate()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Concatenator')
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='input video paths')
    parser.add_argument('--interval', type=int, default=50, help='interval distance between two videos')
    parser.add_argument('--output', type=str, required=True, help='output video path')
    args = parser.parse_args()

    main(args) 
