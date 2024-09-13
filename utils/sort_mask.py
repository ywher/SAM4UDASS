'''
func:对制定文件夹中的二值掩膜按照前景像素数目进行排序,面积大小从大到小
autor: yanweihao
data: 2023/7/1
'''

import os
import cv2
import json
import shutil
from tqdm import tqdm
import argparse

def sort_masks(folder_path):
    folder_list = os.listdir(folder_path)
    folder_list.sort()
    folder_dict = {}

    for folder_name in tqdm(folder_list, desc='Processing folders'):
        folder_dict[folder_name] = {}
        folder_dir = os.path.join(folder_path, folder_name)
        tmp_folder_name = folder_name + '_tmp'
        tmp_folder_dir = os.path.join(folder_path, tmp_folder_name)

        if not os.path.isdir(folder_dir):
            continue

        # Create the temporary folder if it doesn't exist
        os.makedirs(tmp_folder_dir, exist_ok=True)

        mask_list = []
        for file_name in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, file_name)
            if not os.path.isfile(file_path) or not file_name.endswith('.png'):
                continue  # Skip files that are not PNG images

            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            foreground_pixels = cv2.countNonZero(mask)
            mask_list.append((file_path, foreground_pixels))

        sorted_masks = sorted(mask_list, key=lambda x: x[1], reverse=True)

        for idx, (file_path, _) in enumerate(sorted_masks):
            original_name = os.path.basename(file_path)
            ext = os.path.splitext(original_name)[1]
            new_file_name = f"{idx}{ext}"

            new_file_path = os.path.join(tmp_folder_dir, new_file_name)
            shutil.move(file_path, new_file_path)

            folder_dict[folder_name][original_name] = new_file_name

        csv_path = os.path.join(folder_dir, 'metadata.csv')
        shutil.move(csv_path, os.path.join(tmp_folder_dir, 'metadata.csv'))

        # Remove the original folder and rename the temporary folder to the original folder name
        shutil.rmtree(folder_dir)
        shutil.move(tmp_folder_dir, folder_dir)

    return folder_dict

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--folder_path', type=str, default='/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/train2')
    args.add_argument('--output_file', type=str, default='/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/mapping_train2.json')
    return args.parse_args()


if __name__ == '__main__':
    # Example usage
    args = get_args()
    # folder_path = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/train2'
    mapping_dict = sort_masks(args.folder_path)

    # Save the mapping dictionary to a JSON file
    # output_file = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/mapping_train2.json'
    with open(args.output_file, 'w') as f:
        json.dump(mapping_dict, f, indent=2)
