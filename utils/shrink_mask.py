import numpy as np
import cv2

import numpy as np
import cv2

def shrink_region(mask, num_pixels): 
    '''
    func: 对前景标签进行区域收缩
    input:
        mask: 前景标签, numpy array, 0表示背景, 1表示前景
        num_pixels: 收缩的像素数
    output:
        result: 收缩后的前景标签
        area: 收缩后前景占据的面积
    '''   
    # 对前景标签的边界进行收缩
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=num_pixels)
    
    result = eroded.astype(np.uint8)
    
    # 计算收缩面积
    area = np.sum(result == 1)
    
    return result, area


if __name__ == '__main__':
    # 创建一个 10x10 的二值掩膜，前景标签在边界上
    mask = np.zeros((10, 10))
    mask[2:8,2:8] = 1

    # 对前景标签进行区域收缩
    result, area = shrink_region(mask, 2)

    # 输出结果
    print("Original mask:")
    print(mask)
    print("Shrunk mask:")
    print(result)
    print("Shrunk area:", area)


