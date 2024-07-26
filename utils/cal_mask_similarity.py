import os
import cv2
import natsort
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import argparse
from cityscapesscripts.helpers.labels import trainId2label as trainid2label

"""
func: 统计二值掩膜和语义标签的信息, 并保存到CSV文件
"""
class Cal_Mask_Similarity():
    def __init__(self, args):
        self.input_folder_masks = args.input_folder_masks
        self.mask_suffix = args.mask_suffix
        self.input_folder_labels = args.input_folder_labels
        self.label_suffix = args.label_suffix
        self.input_image_path = args.input_image_path
        self.image_suffix = args.image_suffix
        self.num_classes = args.num_classes
        self.ratio_thres = args.ratio_thres
        self.output_csv1 = args.output_csv1
        self.output_csv2 = args.output_csv2
        self.show_mask = args.show_mask
    
    def trainid2color(self, trainid):
        '''
        function: convert trainID to color using cityscapes info
        input: 
                trainid,    1,      np.uint8,   trainID
        output: 
                color,      (,,),   uint8,      color in city
        '''
        # if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask
    
    def color_segmentation(self, segmentation):
        '''
        func:   get the color segmentation result from the trainid segmentation result
        input:  segmentation, [h, w], uint8, trainid segmentation result
        '''
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        for train_id in train_ids:
            if self.num_classes == 16 and train_id in [9, 14, 16]:
                continue
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
    
    def calculate(self):
        # 初始化统计结果1
        result1_columns = ['img_name', 'mask_name', 'id1', 'id1_num', 'id2', 'id2_num', 'id3', 'id3_num', '2_div_1']
        result1_data = []

        # 初始化统计结果2
        result2 = np.zeros((self.num_classes, self.num_classes), dtype=np.uint64)

        # 遍历mask文件夹
        mask_folders = natsort.natsorted(os.listdir(self.input_folder_masks))
        for mask_folder in tqdm(mask_folders, desc='Processing folders'):
            mask_folder_path = os.path.join(self.input_folder_masks, mask_folder)
            # aachen_000000_000019_leftImg8bit.png
            # 遍历mask文件
            mask_files = natsort.natsorted([f for f in os.listdir(mask_folder_path) if f.endswith(self.mask_suffix)])
            
            # 获取图像名称
            img_name = mask_folder.replace('_leftImg8bit', '')
            
            # 获取对应标签图的路径, 单次获取就好
            label_file_path = os.path.join(self.input_folder_labels, img_name + self.label_suffix)

            if not os.path.exists(label_file_path):
                print(f"Warning: Label file not found for {img_name}")
                continue

            # 读取标签图，转化为NumPy数组
            label_img = Image.open(label_file_path).convert('L')
            label_np = np.array(label_img)
            color_label = self.color_segmentation(label_np)
            
            # 获取原始图像
            rgb_img_path = os.path.join(self.input_image_path, img_name + self.image_suffix)
            rgb_img = cv2.imread(rgb_img_path)
            
            for mask_file in mask_files:
                mask_file_path = os.path.join(mask_folder_path, mask_file)
                mask_img = Image.open(mask_file_path).convert('L')

                # 转换为NumPy数组
                mask_np = np.array(mask_img)
                
                # 统计mask区域内的类别ID及数量
                unique_ids, id_counts = np.unique(label_np[mask_np == 255], return_counts=True)
                # descending order
                sorted_ids = unique_ids[np.argsort(id_counts)][::-1]
                
                # only consider the mask with at least 2 classes
                if len(sorted_ids) < 2:
                    continue
                
                if self.show_mask:
                    print('num of ids:', len(sorted_ids))
                    # initialize the white mask with zeros
                    white_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
                    # change the mask_np in white mask to white
                    white_mask[mask_np == 255] = (255, 255, 255)
                    # mix the mask_np area of rgb image with white mask
                    rgb_img[mask_np == 255] = 0.5 * rgb_img[mask_np == 255] + 0.5 * white_mask[mask_np == 255]
                    rgb_img = np.array(rgb_img, dtype=np.uint8)
                    # mix the mask_np area of color label with white mask
                    color_label[mask_np == 255] = 0.5 * color_label[mask_np == 255] + 0.5 * white_mask[mask_np == 255]
                    color_label = np.array(color_label, dtype=np.uint8)
                    
                    # concate the rgb image and color label horizonally
                    concat_image = np.concatenate((rgb_img, color_label), axis=1)
                    cv2.imshow('mask', concat_image)
                    cv2.waitKey(0)
                    
                
                # 记录统计结果1
                row_data = [img_name, mask_file.replace(self.mask_suffix, ''), -1, 0, -1, 0, -1, 0, 0]

                for i, class_id in enumerate(sorted_ids[:3]):  # 使用3索引并不会报越界的错误
                    row_data[i * 2 + 2] = class_id
                    row_data[i * 2 + 3] = id_counts[np.where(unique_ids == class_id)][0]

                # only calculate the ratio greater than ratio_thres
                row_data[8] = row_data[5] / row_data[3]
                if self.ratio_thres > 0 and row_data[8] < self.ratio_thres:
                    continue

                result1_data.append(row_data)

                # 记录统计结果2
                result2[sorted_ids[0]][sorted_ids[1]] += 1

        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(os.path.dirname(self.output_csv1)):
            os.makedirs(os.path.dirname(self.output_csv1))
        if not os.path.exists(os.path.dirname(self.output_csv2)):
            os.makedirs(os.path.dirname(self.output_csv2))
        
        # 保存统计结果1到CSV文件
        result1_df = pd.DataFrame(result1_data, columns=result1_columns)
        result1_df.to_csv(self.output_csv1, index=False)

        # 保存统计结果2到CSV文件
        result2_df = pd.DataFrame(result2)
        result2_df.to_csv(self.output_csv2, index=False, header=False)
        
        # 保存统计结果2到npy文件
        np.save(self.output_csv2.replace('.csv', '.npy'), result2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计二值掩膜和语义标签的信息")
    parser.add_argument("--input_folder_masks", type=str, help="二值掩膜文件夹路径")
    parser.add_argument("--mask_suffix", type=str, help="二值掩膜文件后缀，如'.png'")
    parser.add_argument("--input_folder_labels", type=str, help="语义标签文件夹路径")
    parser.add_argument("--label_suffix", type=str, help="语义标签文件后缀，如'.png'")
    parser.add_argument("--input_image_path", type=str, help="原始图像文件夹路径")
    parser.add_argument("--image_suffix", type=str, help="原始图像文件后缀，如'.png'")
    parser.add_argument("--num_classes", type=int, default=19, help="类别数量, 默认为19")
    parser.add_argument("--ratio_thres", type=float, default=-1, help="类别数量占比阈值, 默认为0.01,认为是有效两个类别")
    parser.add_argument("--output_csv1", type=str, required=True, help="保存统计结果1的CSV文件路径")
    parser.add_argument("--output_csv2", type=str, required=True, help="保存统计结果2的CSV文件路径")
    parser.add_argument("--show_mask", action="store_true", help="是否显示掩膜图像")
    
    args = parser.parse_args()
    cal = Cal_Mask_Similarity(args)
    cal.calculate()