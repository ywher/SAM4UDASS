import os
import shutil

# 文件夹路径
folder_path = "../outputs/cityscapes/unsup_fusion"

# 判断是否有trainID_bg文件夹，如果没有则创建
trainid_bg_path = os.path.join(folder_path, "trainID_bg")
if not os.path.exists(trainid_bg_path):
    os.makedirs(trainid_bg_path)

# 遍历所有子文件夹
for sub_folder in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, sub_folder)
    if os.path.isdir(sub_folder_path) and sub_folder != "trainID_bg":
        trainid_sub_path = os.path.join(sub_folder_path, "trainID_bg")
        if os.path.exists(trainid_sub_path):
            # 如果子文件夹中有trainID_bg文件夹，则将其中的图片移动到新建的trainID_bg文件夹中
            for file_name in os.listdir(trainid_sub_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    src_path = os.path.join(trainid_sub_path, file_name)
                    dest_path = os.path.join(trainid_bg_path, file_name)
                    shutil.move(src_path, dest_path)
        else:
            # 如果子文件夹中没有trainID_bg文件夹，则将trainID_bg文件夹中的图片移动到子文件夹中
            for file_name in os.listdir(trainid_bg_path):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    city_name = file_name.split('_')[0]
                    dest_folder_path = os.path.join(folder_path, sub_folder, city_name, "trainID_bg")
                    if not os.path.exists(dest_folder_path):
                        os.makedirs(dest_folder_path)
                    src_path = os.path.join(trainid_bg_path, file_name)
                    dest_path = os.path.join(dest_folder_path, file_name)
                    shutil.move(src_path, dest_path)