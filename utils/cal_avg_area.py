import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

# 函数用于统计分割标签图文件夹中各类别的像素数量总数、出现频次和平均像素面积
def analyze_segmentation_results(input_folder, sub_folder, label_suffix, num_class, output_csv):
    class_stats = {}  # 用于存储各类别的统计信息
    # image_area = input_width * input_height  # 计算图像面积
    for class_id in range(num_class):
        class_stats[class_id] = {'num_pixels': 0, 'number_occur': 0}
    
    if not sub_folder:
        num_files = len(os.listdir(input_folder))
        file_pathes = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(label_suffix)]
    else:
        sub_folders = os.listdir(input_folder)
        num_files = 0
        file_pathes = []
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(input_folder, sub_folder)
            num_files += len(os.listdir(sub_folder_path))
            file_pathes += [os.path.join(sub_folder_path, filename) for filename in os.listdir(sub_folder_path) if filename.endswith(label_suffix)]
    
    # 遍历文件夹中的每个文件
    for file_path in tqdm(file_pathes):
        file_path = os.path.join(input_folder, file_path)

        # 使用OpenCV读入图像
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 统计像素数量
        unique, counts = np.unique(image, return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        # 更新每个类别的统计信息
        for class_id in range(num_class):
            if class_id in pixel_counts:
                class_stats[class_id]['num_pixels'] += pixel_counts[class_id]
                class_stats[class_id]['number_occur'] += 1

    # 计算每个类别的平均像素面积
    for class_id in class_stats:
        if class_stats[class_id]['number_occur'] == 0:
            class_stats[class_id]['avg_area'] = 0
            class_stats[class_id]['occur_freq'] = 0
        else:
            class_stats[class_id]['avg_area'] = class_stats[class_id]['num_pixels'] / class_stats[class_id]['number_occur']
            class_stats[class_id]['occur_freq'] = class_stats[class_id]['number_occur'] / num_files

    # 将统计结果保存到CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['id', 'num_pixels', 'number_occur', 'avg_area', 'occur_freq']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in class_stats:
            writer.writerow({'id': class_id,
                             'num_pixels': class_stats[class_id]['num_pixels'],
                             'number_occur': class_stats[class_id]['number_occur'],
                             'avg_area': class_stats[class_id]['avg_area'],
                             'occur_freq': class_stats[class_id]['occur_freq']})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析语义分割标签图文件夹的统计信息")
    parser.add_argument("--input_folder", type=str, help="语义分割标签图文件夹路径")
    parser.add_argument("--sub_folder", action="store_true", default=False, help="是否遍历子文件夹")
    parser.add_argument("--label_suffix", type=str, default=".png", help="语义分割标签图文件后缀")
    parser.add_argument("--num_class", type=int, help="类别数量")
    # parser.add_argument("--input_width", type=int, default=2048, help="输入图像宽度")
    # parser.add_argument("--input_height", type=int, default=1024, help="输入图像高度")
    parser.add_argument("--output_csv", type=str, help="保存统计结果的CSV文件路径")
    args = parser.parse_args()

    print('sub folder', args.sub_folder)
    analyze_segmentation_results(args.input_folder, args.sub_folder, args.label_suffix, args.num_class, args.output_csv)
