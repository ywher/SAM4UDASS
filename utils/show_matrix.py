import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os

def get_parse():
    args = argparse.ArgumentParser()
    args.add_argument('--csv_file', type=str, default='output_similarity/similarity_matrix.csv')
    args.add_argument('--log_value', action='store_true', default=False)
    args.add_argument('--num_classes', type=int, default=19, help='num of classes, 16 for synthia, 19 for gta5')
    args.add_argument('--selected_class_ids', type=str, default="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18")
    args.add_argument('--output_filename', type=str, default='output_similarity/original_frequency_map.pdf')
    return args.parse_args()

def plot_save_matrix(matrix, figsize, class_names, output_filepath):
    plt.figure(figsize=figsize)
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    # plt.title('Original Frequency Map')
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.savefig(output_filepath, bbox_inches='tight')

if __name__ == '__main__':

    args = get_parse()

    # 读取统计结果2的CSV文件
    result2 = pd.read_csv(args.csv_file, header=None).to_numpy()
    
    # 对数值结果进行对数变换，增强对比度，注意：对数值为0的元素进行对数变换会报错，过滤这些元素
    if args.log_value:
        result2[result2 == 0] = 1
        result2 = np.log2(result2)

    # 创建新的频次列表array，只包含下半三角形的数值
    lower_triangle = np.tril(result2)

    # 要可视化的特定类别ID列表
    selected_class_ids = [int(id_str) for id_str in args.selected_class_ids.split()] # 示例选择的类别ID列表

    # define class names
    # class_names=[
    #     '0.road', '1.sidewalk', '2.building', '3.wall', '4.fence', '5.pole', '6.traffic light',
    #     '7.traffic sign', '8.vegetation', '9.terrain', '10.sky', '11.person', '12.rider',
    #     '13.car', '14.truck', '15.bus', '16.train', '17.motorcycle', '18.bicycle'
    # ]
    
    class_names=[
        'road', 'sw', 'build', 'wall', 'fence', 'pole', 'light',
        'sign', 'vege', 'terrain', 'sky', 'person', 'rider',
        'car', 'truck', 'bus', 'train', 'motor', 'bike'
    ]
    # 检查output_filename文件夹是否存在
    if not os.path.exists(os.path.dirname(args.output_filename)):
        os.makedirs(os.path.dirname(args.output_filename))

    # 创建原始频次图
    plot_save_matrix(result2, figsize=(8, 6), class_names=class_names, output_filepath=args.output_filename)

    # 创建对角线对称元素相加后的频次图
    output_path = args.output_filename.replace('.pdf', '_symmetric_diagonal_sum.pdf')
    plot_save_matrix(lower_triangle, figsize=(8, 6), class_names=class_names, output_filepath=output_path)

    # 保存下半三角形的数值到CSV文件
    np.savetxt('lower_triangle.csv', lower_triangle, delimiter=',')

    if len(selected_class_ids) < args.num_classes:
        # 选择制定ID的类别的横纵坐标对应数值，创建频次图
        result2_selected = np.zeros((len(class_names), len(class_names)))
        for i in range(len(selected_class_ids)):
            for j in range(len(selected_class_ids)):
                result2_selected[selected_class_ids[i]][selected_class_ids[j]] = result2[selected_class_ids[i]][selected_class_ids[j]]
        
        output_path = args.output_filename.replace('.pdf', '_selected.pdf')
        plot_save_matrix(result2_selected, figsize=(8, 6), class_names=class_names, output_filepath=output_path)
        
        # 选择制定ID的类别的横纵坐标对应数值，创建对角线对称元素相加后的频次图
        lower_triangle_selected = np.tril(result2_selected)
        output_path = args.output_filename.replace('.pdf', '_selected_symmetric_diagonal_sum.pdf')
        plot_save_matrix(lower_triangle_selected, figsize=(8, 6), class_names=class_names, output_filepath=output_path)
        np.savetxt('lower_triangle_selected.csv', lower_triangle_selected, delimiter=',')