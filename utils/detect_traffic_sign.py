import cv2
import numpy as np

def detect_traffic_sign(mask):
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 判断轮廓数量
    if len(contours) < 1:
        return False, None, None
    
    # 获取整个标志的轮廓
    traffic_sign_contour = contours[0]
    
    # 计算标志的矩形边界
    x, y, w, h = cv2.boundingRect(traffic_sign_contour)
    
    # 判断标志的形状
    if h < w:
        return False, None, None
    
    # 分割上下两部分
    center_y = y + h // 2
    mask_upper = np.zeros_like(mask)
    mask_lower = np.zeros_like(mask)
    mask_upper[y:center_y, x:x+w] = mask[y:center_y, x:x+w]
    mask_lower[center_y:y+h, x:x+w] = mask[center_y:y+h, x:x+w]
    
    return True, mask_upper, mask_lower


if __name__ == '__main__':
    # 读取二值掩模图像
    mask = cv2.imread('utils/images/81.png', 0)  # 替换成你的二值掩模图像路径

    # 进行形状判断和分割操作
    is_traffic_sign, mask_upper, mask_lower = detect_traffic_sign(mask)

    if is_traffic_sign:
        # 显示上部分和下部分掩模
        cv2.imshow('Upper Part', mask_upper)
        cv2.imshow('Lower Part', mask_lower)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Not a traffic sign shape.')
