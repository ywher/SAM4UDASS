import os
import argparse
import cv2
import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='将文件夹中的所有图片按照文件名排序后生成视频')
parser.add_argument('--input_folder', type=str, help='输入文件夹路径')
parser.add_argument('--frame_rate', type=int, help='视频帧率')
parser.add_argument('--output_file', type=str, help='输出文件路径')
parser.add_argument('--data_ratio', type=float, default=1.0, help='使用多少比例的图片')
args = parser.parse_args()

# 获取输入参数
input_folder = args.input_folder
frame_rate = args.frame_rate
output_file = args.output_file
print('output_file:', output_file)

# 读取文件夹下的所有图片并按文件名排序
images = []
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        images.append(os.path.join(input_folder, filename))
        
images = images[:int(len(images) * args.data_ratio)]

# 读取第一张图片以获取图片尺寸
img = cv2.imread(images[0])
height, width, channels = img.shape

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

# 逐一读取图片并写入视频
bar = tqdm.tqdm(total=len(images))
for image in images:
    img = cv2.imread(image)
    out.write(img)
    bar.update(1)
bar.close()

# 释放资源
out.release()
