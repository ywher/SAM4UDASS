import os
import shutil
import argparse
from tqdm import tqdm

def reverse_rename_and_copy(input_path, output_path):
    # 获取输入路径中的所有图片文件
    image_files = [f.path for f in os.scandir(input_path) if f.is_file()]

    for image_file in tqdm(image_files, desc="Processing", unit="file"):
        # 获取文件名和文件后缀
        file_name = os.path.basename(image_file)
        base_name, extension = os.path.splitext(file_name)

        # 提取子文件夹名和图片名
        subfolder_name, image_name = base_name.split("_", 1)

        # 构建输出子文件夹路径
        output_subfolder = os.path.join(output_path, subfolder_name)

        # 如果输出子文件夹不存在，则创建它
        os.makedirs(output_subfolder, exist_ok=True)

        # 构建新的文件路径
        new_path = os.path.join(output_subfolder, f"{image_name}{extension}")

        # 复制文件到输出路径
        shutil.copy2(image_file, new_path)

def main():
    parser = argparse.ArgumentParser(description="Reverse rename and copy image files from input to output folder.")
    parser.add_argument("--input_path", help="Input folder path containing images with specific naming format.")
    parser.add_argument("--output_path", help="Output folder path where reversed renamed files will be copied.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # 创建输出文件夹
    os.makedirs(output_path, exist_ok=True)

    reverse_rename_and_copy(input_path, output_path)

if __name__ == "__main__":
    main()
