
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm

def calculate_centers_and_rectangles(label_folder, label_id):
    label_files = os.listdir(label_folder)
    centers = []
    # 创建一个全零的numpy数组用于统计
    label_image = cv2.imread(os.path.join(label_folder, label_files[0]), cv2.IMREAD_GRAYSCALE)
    statistics_array = np.zeros(label_image.shape, dtype=np.uint64)
    print(statistics_array.shape)

    bar = tqdm.tqdm(total=len(label_files))
    for file in label_files:
        label_path = os.path.join(label_folder, file)
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 提取指定标签ID对应的掩模
        mask = (label_image == label_id).astype(np.uint8)

        # 计算几何中心
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        center = (center_x, center_y)
        centers.append(center)

        # 统计几何中心位置
        statistics_array[center_y, center_x] += 1
        
        bar.update(1)
    bar.close()

    return centers, statistics_array

def visualize_statistics(centers, statistics_array):
    # 可视化统计结果
    plt.imshow(statistics_array, cmap='gray')
    plt.scatter([center[0] for center in centers], [center[1] for center in centers], c='red', marker='.')
    plt.axis('off')
    plt.savefig('centers.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

if __name__ == '__main__':
    # 输入真值路径, 和想要统计的类别ID, 注意修改命名
    # cityscapes
    # label_folder = "/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/gtFine/train_all"  # 替换为您的数据集标签图像文件夹路径
    # label_id = 0  # 替换为您需要统计的类别标签I
    # acdc
    # label_folder = "/media/ywh/1/yanweihao/projects/uda/DAFormer/data/acdc/gt/train"
    # label_id = 10
    # centers, statistics_array = calculate_centers_and_rectangles(label_folder, label_id)
    # np.save('sky_centers.npy', statistics_array)
    # visualize_statistics(centers, statistics_array)
    
    # file_name = 'acdc_sky_centers.npy'
    # center_npt = np.load(file_name)
    # # 归一化到0-255范围
    # normalized_array = (center_npt - np.min(center_npt)) * 255 / (np.max(center_npt) - np.min(center_npt))
    # normalized_array = normalized_array.astype(np.uint8)

    # # 可视化统计结果
    # plt.imshow(normalized_array, cmap='gray')
    # plt.axis('off')
    # plt.savefig('statistics_visualization.png', dpi=500, bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    
    # import numpy as np
    # import matplotlib.pyplot as plt

    # 加载.npy文件
    array = np.load('acdc_sky_centers.npy')
    indices = np.nonzero(array)

    # 提取最小和最大的非零位置坐标
    min_x, min_y = np.min(indices, axis=1)
    max_x, max_y = np.max(indices, axis=1)
    
    bounding_box = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    print(bounding_box)
