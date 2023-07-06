import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def calculate_brightness(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的平均亮度值
    brightness = cv2.mean(gray_image)[0]
    return brightness

def analyze_brightness(folder_path, save_filename):
    # 读取目标文件夹中的图像文件
    image_files = os.listdir(folder_path)

    # 存储图像的亮度值
    brightness_values = []

    bar = tqdm.tqdm(total=len(image_files), desc='Calculating brightness')
    for file in image_files:
        # 构建图像路径
        image_path = os.path.join(folder_path, file)

        # 读取图像
        image = cv2.imread(image_path)

        # 计算图像的亮度值
        brightness = calculate_brightness(image)

        # 将亮度值添加到列表中
        brightness_values.append(brightness)
        bar.update(1)
    bar.close()

    # 将亮度值转换为NumPy数组
    brightness_values = np.array(brightness_values)

    # 保存亮度分布为.npy数组
    np.save('outputs/{}_brightness_distribution.npy'.format(save_filename), brightness_values)

    # 绘制亮度分布直方图
    plt.hist(brightness_values, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.title('{} Brightness Distribution'.format(save_filename.capitalize()))
    plt.savefig('outputs/{}_brightness_distribution.pdf'.format(save_filename))
    plt.show()

if __name__ == '__main__':
    # 输入目标图像文件夹路径
    # save_filename = 'fog' ['fog', 'night', 'rain', 'snow']
    for save_filename in ['fog', 'night', 'rain', 'snow']:
        # 分析亮度
        folder_path = "/media/ywh/1/yanweihao/projects/uda/DAFormer/data/acdc/rgb_anon/{}/train_all".format(save_filename)
        print('Analyzing {} brightness'.format(save_filename))
        analyze_brightness(folder_path, save_filename)
        print('')
