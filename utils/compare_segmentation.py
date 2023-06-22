import cv2
import numpy as np

def compare_segmentation(predicted_label, ground_truth_label, predicted_image):
    # 加载预测标签图、真值标签图和预测彩色图
    predicted = cv2.imread(predicted_label, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_label, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(predicted_image)
    
    # 创建预测彩色图的副本
    overlay = image.copy()

    # 标注预测错误的区域
    error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
    error_mask_color = cv2.cvtColor(error_mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.9, error_mask_color, 0.1, 0)

    # 显示结果
    cv2.imshow("Predicted Image", image)
    cv2.imshow("Error Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
predicted_label_path = "pred.png"  # 替换为预测标签图的路径
ground_truth_label_path = "gt.png"  # 替换为真值标签图的路径
predicted_image_path = "img.png"  # 替换为预测彩色图的路径
compare_segmentation(predicted_label_path, ground_truth_label_path, predicted_image_path)
