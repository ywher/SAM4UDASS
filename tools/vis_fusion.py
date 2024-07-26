import os
import cv2
import numpy as np

def show_images(folder_count, rows, cols, folder_paths):
    image_index = 0
    while True:
        # 创建一个空白画布
        canvas = np.zeros((rows * 200, cols * 400, 3), dtype=np.uint8)
        image_names = os.listdir(folder_paths[0])
        for i, folder_path in enumerate(folder_paths):
            # 读取图片
            if 'gt_color' in folder_path:
                image_path = os.path.join(folder_path, f"{image_names[image_index].split('_leftImg8bit.png')[0]}_gtFine_color.png")
            else:
                image_path = os.path.join(folder_path, f"{image_names[image_index]}")
            image = cv2.imread(image_path)

            if image is None:
                continue

            # 缩放图片以适应显示窗口
            image = cv2.resize(image, (400, 200)) # (width, height)

            # 计算图片在画布上的位置
            row, col = divmod(i, cols)
            x, y = col * 400, row * 200

            # 将图片放在画布上
            canvas[y:y + 200, x:x + 400] = image

        # 在画布每一列的第一张图片显示备注
        cv2.putText(canvas, "SAM filled with UDA", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "UDA", (10+400, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "fusion", (10+400*2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "gt", (10+400*3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # 显示画布
        cv2.imshow("Images", canvas)

        # 等待按键
        key = cv2.waitKey(0) & 0xFF

        # 按左右键切换图片
        if key == ord("d"):  # 向右键
            image_index += 1
        elif key == ord("a"):  # 向左键
            image_index -= 1
            if image_index < 0:
                image_index = 0
        elif key == 27:  # 按下 ESC 键退出
            break

    cv2.destroyAllWindows()

# 示例参数
folder_count = 8
rows = 2
cols = 4
target_folder = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/fusion2'

folder_paths = [os.path.join(target_folder, pth) for pth in os.listdir(target_folder)]
folder_paths.sort()
print('folder_paths:', folder_paths)

show_images(folder_count, rows, cols, folder_paths)
