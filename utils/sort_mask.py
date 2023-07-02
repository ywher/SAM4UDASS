import os
import cv2
import json
import shutil
from tqdm import tqdm


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
        if not os.path.exists(tmp_folder_dir):
            os.makedirs(tmp_folder_dir)

        mask_list = []
        for file_name in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, file_name)
            if not os.path.isfile(file_path) or not file_name.endswith('.png'):
                continue  # meta.csv

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
        shutil.move(csv_path, tmp_folder_dir + '/metadata.csv')
        # remove original folder and rename tmp folder to original folder
        shutil.rmtree(folder_dir)
        shutil.move(tmp_folder_dir, folder_dir)

    return folder_dict


if __name__ == '__main__':
    # 示例用法
    folder_path = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2'
    mapping_dict = sort_masks(folder_path)

    # 将映射关系保存到 JSON 文件
    output_file = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/mapping_train2.json'
    with open(output_file, 'w') as f:
        json.dump(mapping_dict, f, indent=2)
