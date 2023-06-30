import cv2
import numpy as np

def divide_sign_pole(mask, kernel_size=None, iterations=1):
    '''
    function: divide the sign and pole from one mask
    input:
        mask: the mask of sign and pole, [H, W, 3]
        kernel_size: the size of kernel, int
        iterations: the number of iterations, int
    output:
        two masks: upper_part, lower_part, both in [H, W, 3]
    '''
    # 转换为灰度图
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if not kernel_size:  # 如果没有指定内核大小
        # 获取前景的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 计算前景的大小
        foreground_area = cv2.contourArea(contours[0]) if len(contours) > 0 else 0
        print('foreground area', foreground_area)
        # 根据前景大小自适应调整内核的大小
        kernel_size = int(int(np.sqrt(foreground_area)) // 1.1)  # 根据实际情况调整系数
        print('kernel size', kernel_size)
        kernel_size = max(kernel_size, 1)  # 最小内核大小为1
    print('kernel size', kernel_size)
    
    # 定义形态学操作的内核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 通过腐蚀操作去除连接部分
    eroded = cv2.erode(mask, kernel, iterations=iterations)

    # 通过膨胀操作恢复上部分
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    # 分离上部分和下部分
    upper_part = dilated.copy()  # [H, W]
    lower_part = cv2.subtract(mask, upper_part)  # [H, W]
    
    kernel2 = np.ones((kernel_size//2, kernel_size//2), np.uint8)
    eroded2 = cv2.erode(lower_part, kernel2, iterations=iterations)
    lower_part2 = cv2.dilate(eroded2, kernel2, iterations=iterations)
    
    # 转化回二值三通道mask
    upper_part = cv2.cvtColor(upper_part, cv2.COLOR_GRAY2BGR)  # [H, W, 3] [0, 255]
    lower_part = cv2.cvtColor(lower_part, cv2.COLOR_GRAY2BGR)  # [H, W, 3] [0, 255]
    lower_part2 = cv2.cvtColor(lower_part2, cv2.COLOR_GRAY2BGR)  # [H, W, 3] [0, 255]
    
    print('shape of upper part', upper_part.shape, 'shape of lower part', lower_part.shape)
    print('the max and min of upper part', np.max(upper_part), \
        np.min(upper_part), np.max(lower_part), np.min(lower_part))
    cv2.imshow('upper_part', upper_part)
    cv2.waitKey(0)
    cv2.imshow('lower_part', lower_part)
    cv2.waitKey(0)
    cv2.imshow('lower_part2', lower_part2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return upper_part, lower_part

if __name__ == '__main__':
    mask = cv2.imread('utils/images/81.png')
    # kernel_size = 20
    divide_sign_pole(mask)
