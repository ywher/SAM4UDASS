import cv2
import numpy as np
import os

def mark_foreground(image_path, mask_paths, mask_index):
    # 读取原始图像
    original_image = cv2.imread(image_path)

    # 创建一个全黑的掩膜图像，与原始图像大小相同
    mask = np.zeros_like(original_image)
    mask[:,:,2] = 255

    # 遍历所有的二值掩膜图路径
    for mask_path in mask_paths:
        # 读取二值掩膜图像
        mask_image = cv2.imread(mask_path, 0)

        #将mask_image中的1值的区域在mask中设置为绿色
        mask[mask_image > 0] = [0, 255, 0]
    #将mask中剩余区域赋值为红色
    # mask

    # 将掩膜图应用到原始图像上
    marked_image = cv2.addWeighted(original_image, 0.5, mask, 0.5, 0)

    # 显示标记后的图像
    cv2.imwrite("outputs_tmp/marked_image{}.png".format(mask_index), marked_image)
    cv2.imshow("Marked Image", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
original_image_path = "images/huanshi.jpg"
index = [0, 1, 4, 23, 41]
# index = [2, 5, 9, 15, 29, 46, 74]
# index = [2, 3, 6, 11, 16, 31, 90]
output_root = "outputs_tmp"
mask_index = 1
mask_paths = [os.path.join(output_root, 'mask{}'.format(mask_index), 'mask_{}_{}.png'.format(mask_index, i)) for i in index]
# mask_paths = ["outputs_tmp/mask2/mask_2_2.png", "outputs_tmp/mask2/mask_2_5.png", "outputs_tmp/mask2/mask_2_9.png", "outputs_tmp/mask2/mask_2_15.png", "outputs_tmp/mask2/mask_2_29.png", "outputs_tmp/mask2/mask_2_46.png", "outputs_tmp/mask2/mask_2_74.png"]
mark_foreground(original_image_path, mask_paths, mask_index)