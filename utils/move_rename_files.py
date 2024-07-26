import os
import shutil
import argparse
from tqdm import tqdm

def rename_and_copy(input_path, output_path):
    # 获取输入路径中的所有子文件夹
    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]

    for subfolder in tqdm(subfolders, desc="Processing", unit="folder"):
        # 获取子文件夹中的所有文件和第二层子文件夹
        contents = [f.path for f in os.scandir(subfolder)]
        
        # 获取子文件夹的名字
        subfolder_name = os.path.basename(subfolder)
        
        for content in tqdm(contents, desc="Copying", unit="file"):
            # 生成新的文件名
            new_name = f"{subfolder_name}_{os.path.basename(content)}"
            # 构建新的文件路径
            new_path = os.path.join(output_path, new_name)
            
            # 复制文件或者文件夹到输出路径
            if os.path.isfile(content):
                shutil.copy2(content, new_path)
            elif os.path.isdir(content):
                shutil.copytree(content, new_path)

def main():
    parser = argparse.ArgumentParser(description="Rename and copy files from input to output folder.")
    parser.add_argument("--input_path", help="Input folder path containing subfolders and files.")
    parser.add_argument("--output_path", help="Output folder path where renamed files will be copied.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # 创建输出文件夹
    os.makedirs(output_path, exist_ok=True)

    rename_and_copy(input_path, output_path)

if __name__ == "__main__":
    main()
