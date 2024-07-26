import os
import csv
import numpy as np
from PIL import Image
import tqdm
import argparse

class SegmentationMetrics:
    def __init__(self, pred_folder, gt_folder, num_classes, pred_suffix, gt_suffix, output_file):
        self.pred_folder = pred_folder  # Path to the folder which stores prediction results
        self.gt_folder = gt_folder      # Path to the folder which stores ground truth masks
        self.num_classes = num_classes  # Number of classes
        self.pred_suffix = pred_suffix  # Suffix of prediction results
        self.gt_suffix = gt_suffix      # Suffix of ground truth masks
        self.output_file = output_file  # Output file name

    def calculate_iou(self, pred_mask, gt_mask):
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def calculate_miou(self, pred_mask, gt_mask):
        class_iou = []
        for class_id in range(self.num_classes):
            pred_class = pred_mask == class_id
            gt_class = gt_mask == class_id
            if np.sum(gt_class) == 0:
                iou = 0
            else:
                iou = self.calculate_iou(pred_class, gt_class)
            class_iou.append(iou)
        miou = np.mean(class_iou)
        return miou, class_iou

    def process_images(self):
        file_names_pred = sorted(os.listdir(self.pred_folder))
        file_names_gt = sorted(os.listdir(self.gt_folder))
        results = []
        bar = tqdm.tqdm(total=len(file_names_pred))
        for i, file_name in enumerate(file_names_pred):
            image_path = os.path.join(self.pred_folder, file_name)
            gt_path = os.path.join(self.gt_folder, file_names_gt[i])
            pred_mask = np.array(Image.open(image_path))
            gt_mask = np.array(Image.open(gt_path))
            miou, class_iou = self.calculate_miou(pred_mask, gt_mask)
            results.append([file_name.replace(self.pred_suffix, ''), f'{miou:.4f}'] + [f'{iou:.4f}' for iou in class_iou])
            # results.append([file_name.replace(self.pred_suffix, ''), miou] + class_iou)
            bar.update(1)
        bar.close()
        
        # Write results to CSV file
        output_file = self.output_file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Name', 'mIoU'] + [f'Class {i} IoU' for i in range(self.num_classes)])
            writer.writerows(results)
        
        print(f"IoU results saved to {output_file}")

def get_parser():
    parser = argparse.ArgumentParser(description="Calculate IoU for segmentation results")
    parser.add_argument('--pred_folder', type=str, help='directory which stores segmentation results')
    parser.add_argument('--gt_folder', type=str, help='directory which stores ground truth masks')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes')
    parser.add_argument('--pred_suffix', type=str, default='_leftImg8bittrainID.png', help='suffix of segmentation results')
    parser.add_argument('--gt_suffix', type=str, default='_gtFine_labelTrainIds.png', help='suffix of ground truth masks')
    parser.add_argument('--output_file', type=str, default='iou_per_image/iou_results.csv', help='output file name')
    return parser.parse_args()
    

if __name__ == '__main__':
    # Example usage:
    parser = get_parser()
    pred_folder_path = parser.pred_folder
    gt_folder_path = parser.gt_folder
    pred_suffix = parser.pred_suffix
    gt_suffix = parser.gt_suffix
    num_classes = parser.num_classes
    output_file = parser.output_file

    metrics = SegmentationMetrics(pred_folder_path, gt_folder_path, num_classes, pred_suffix, gt_suffix, output_file)
    metrics.process_images()