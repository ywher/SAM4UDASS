import os
import argparse
import pandas as pd
from tqdm import tqdm

def sort_csv_files(folder_path, column_name, index, save_path, sort_order=0):
    '''
    author: weihao
    data: 7-**
    func: 对文件夹下的所有.csv文件按照指定列进行排序
    input:
        folder_path: 文件夹路径
        column_name: 需要排序的列名称
        index: 需要排序的索引序号
        save_path: 保存结果的路径
        sort_order: 排序方式, 0:从小到大排序, 1:从大到小排序
    '''
    # 获取文件夹下所有.csv文件的路径
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    file_paths.sort()  #从小到大进行排序
    # 创建一个空的DataFrame用于保存排序结果
    result_df = pd.DataFrame(columns=['Filename', 'Value', 'Index'])

    # 遍历每个.csv文件，并显示进度条
    for file_path in tqdm(file_paths, desc='Processing', unit='file'):
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取指定列和索引的数值
        value = df.iloc[index][column_name]  # 行列的索引, 下标都是从0开始

        # 将文件名和数值添加到结果DataFrame中
        # result_df = result_df.append({'Filename': os.path.basename(file_path), 'Value': value}, ignore_index=True)
        data = {'Filename': [os.path.basename(file_path)],
                'Value': [value],
                'Index': [file_paths.index(file_path)]}
        df_data = pd.DataFrame(data)

        # 将DataFrame与结果DataFrame进行拼接
        result_df = pd.concat([result_df, df_data], ignore_index=True)
    # 根据数值列进行排序
    if sort_order == 0:  # 从小到大排序
        result_df = result_df.sort_values(by='Value')  #　ascending=True,升序
    elif sort_order == 1:  # 从大到小排序
        result_df = result_df.sort_values(by='Value', ascending=False)  # ascending=False,降序

    # 保存排序结果为新的CSV文件
    result_df.to_csv(save_path, index=False)
    print(f"排序结果已保存至 {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV文件排序工具')
    parser.add_argument('--folder_path', type=str, help='文件夹路径', \
                        default='/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/mic_gta_new/ious')
    parser.add_argument('--column_name', type=str, help='需要统计的列名称', default='Differ_3_0')
    parser.add_argument('--index', type=int, help='需要统计的索引序号', default=0)
    parser.add_argument('--save_path', type=str, help='保存结果的路径,输入为文件名', default='result.csv')
    parser.add_argument('--order', type=int, default=0, help='0:从小到大排序, 1:从大到小排序')
    args = parser.parse_args()
    
    args.save_path = os.path.join(args.folder_path, '../', args.column_name + '_' + args.save_path)

    sort_csv_files(args.folder_path, args.column_name, args.index, args.save_path, args.order)
