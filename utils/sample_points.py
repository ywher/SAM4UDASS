import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cityscapesscripts.helpers.labels import trainId2label as trainid2label


def trainid2color(trainid):
    '''
    function: convert trainID to color in cityscapes
    input: trainid
    output: color
    '''
    #if the input is a number in np.uint8, it means it is a trainid
    if type(trainid) == np.uint8:
        label_object = trainid2label[trainid]
        return label_object.color[::-1]
    else:
        color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
        for i in range(trainid.shape[0]):
            label_object = trainid2label[trainid[i]]
            color_mask[i] = label_object.color[::-1]
    return color_mask
    
def color_segmentation(segmentation):
    #get the color segmentation result, initial the color segmentation result with black (0,0,0)
    #input: segmentation [h, w]
    color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    train_ids = np.unique(segmentation)
    for train_id in train_ids:
        color_segmentation[segmentation == train_id] = trainid2color(train_id)
    return color_segmentation

def uniform_sampling(binary_image, sampling_ratio=0.1, sampling_points=3):
    """对二值图进行均匀采样

    :param binary_image: 二值图, 值为0或1
    :param sampling_ratio: 采样比例系数
    :return: 采样后的点位置坐标
    """
    # 计算需要采样的点数
    num_points = int(np.sum(binary_image) * sampling_ratio)
    # num_points = min(num_points, sampling_points)
    num_points = max(num_points, 1)
    print('area', np.sum(binary_image), 'sample num points', num_points)

    # 找到所有值为1的像素的坐标
    coords = np.transpose(np.nonzero(binary_image))
    coords[:,[0,1]] = coords[:,[1,0]] #(y,x) to (x,y)
    # print('coords', coords)

    # 随机选择需要采样的点
    selected_indices = np.random.choice(len(coords), num_points, replace=True)

    # 返回采样后的点位置坐标
    return coords[selected_indices]


if __name__ == '__main__':
    # # 创建一个随机的二值图
    # binary_image = np.random.randint(0, 2, (5, 5))

    # # 设定采样比例系数为0.5
    # sampling_ratio = 0.5

    # # 进行均匀采样
    # sampled_points = uniform_sampling(binary_image, sampling_ratio)

    # # 输出采样后的点位置坐标
    # print(sampled_points)

    # #在二值图中显示采样点的位置
    # binary_image = binary_image.astype(np.uint8) * 255
    # binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('binary_image before', binary_image)
    # cv2.circle(binary_image, tuple(sampled_points[0]), radius=1, color=(0,0,255), thickness=-1) #(B,G,R)
    # cv2.imshow('binary_image after', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    label = cv2.imread('images/aachen_000029_000019_gtFine_labelTrainIds.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # print('label', label.shape)
    color_label = color_segmentation(label)
    cv2.imshow('color_label', color_label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    unique_ids = np.unique(label)
    unique_ids = unique_ids[unique_ids != 255]
    for unique_id in unique_ids:
        id_mask = label == unique_id
        sample_points = uniform_sampling(id_mask, 0.1)
        #draw all the points on the color_label
        color_label_copy = color_label.copy()
        color_label_points = cv2.circle(color_label_copy, tuple(sample_points[0]), radius=1, color=(255,255,255), thickness=-1)      
        for point in sample_points:
            color_label_points = cv2.circle(color_label_points, tuple(point), radius=1, color=(255,255,255), thickness=-1)        
        cv2.imshow('color_label_points', color_label_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()