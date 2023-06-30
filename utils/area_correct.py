import numpy as np
from scipy.ndimage import label
import cv2
from ..result_fusion import Fusion2

def detect_and_correct_suspicious_regions(segmentation_map, sidewalk_id=1, road_id=0):
    # 创建一个新的分割结果图
    corrected_map = np.copy(segmentation_map)
    
    # 对连通的sidewalk区域进行标记
    labeled_array, num_features = label(segmentation_map == sidewalk_id)
    
    # 遍历每个连通的sidewalk区域
    for feature in range(1, num_features+1):
        # 获取当前sidewalk区域的位置
        region_indices = np.where(labeled_array == feature)
        region_rows, region_cols = region_indices[0], region_indices[1]
        
        # 检查区域边界像素
        for i in range(len(region_rows)):
            row, col = region_rows[i], region_cols[i]
            
            # 检查当前像素的上下左右四个邻居像素
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            # 检查邻居像素是否为road，如果是，则将当前像素的id设置为sidewalk
            for neighbor_row, neighbor_col in neighbors:
                if 0 <= neighbor_row < segmentation_map.shape[0] and 0 <= neighbor_col < segmentation_map.shape[1]:
                    if segmentation_map[neighbor_row, neighbor_col] == road_id:
                        corrected_map[row, col] = sidewalk_id
                        break
    
    return corrected_map



if __name__ == '__main__':
    image_path = 'images/aachen_000015_000019_leftImg8bit.png'
    segmentation = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # (1024, 2048)
    segmentation = cv2.resize(segmentation, (1024, 512))
    print('seg shape', segmentation.shape)
    new_seg = detect_and_correct_suspicious_regions(segmentation)
    #show the segmentation and new_seg
    fusion = Fusion2()
    seg_color = fusion.color_segmentation(segmentation)
    new_seg_color = fusion.color_segmentation(new_seg)
    cv2.imshow('seg', seg_color)
    cv2.imshow('new_seg', new_seg_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    