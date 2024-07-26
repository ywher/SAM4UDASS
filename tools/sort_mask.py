import os
import cv2
from tqdm import tqdm

def sort_masks(folder_path):
    folder_list = os.listdir(folder_path)
    folder_dict = {}

    for folder_name in tqdm(folder_list, desc='Processing folders'):
        folder_dict[folder_name] = {}
        folder_dir = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_dir):
            continue
        
        mask_list = []
        for file_name in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, file_name)
            if not os.path.isfile(file_path) or not file_name.endswith('.png'):
                continue

            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            foreground_pixels = cv2.countNonZero(mask)
            mask_list.append((file_path, foreground_pixels))
        
        sorted_masks = sorted(mask_list, key=lambda x: x[1], reverse=True)

        for idx, (file_path, _) in enumerate(sorted_masks):
            new_file_name = f"{idx}.png"
            new_file_path = os.path.join(folder_dir, new_file_name)
            os.rename(file_path, new_file_path)

            original_name = os.path.basename(file_path)
            folder_dict[folder_name][original_name] = new_file_name

    return folder_dict

# 示例用法
folder_path = '/path/to/folder'
mapping_dict = sort_masks(folder_path)

# 将映射关系保存到文件
output_file = 'mapping.txt'
with open(output_file, 'w') as f:
    for folder_name, mapping in mapping_dict.items():
        f.write(f"{folder_name}:\n")
        for original_name, new_name in mapping.items():
            f.write(f"  {original_name} -> {new_name}\n")
        f.write('\n')
