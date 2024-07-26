import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_statistics(data, column_name, percentage_interval, alpha_thres, save_pdf_path, save_txt_path, args):
    # 保存参数
    args = args
    # 读取 CSV 文件
    df = pd.read_csv(data)
    
    # 获取指定列的数据
    column_data = df[column_name]
    # 使用alpha_threshold筛选数据
    if alpha_thres > 0:
        column_data = column_data[column_data >= alpha_thres]
    
    # 计算平均值和方差
    mean = np.mean(column_data)
    variance = np.var(column_data)
    
    # 计算指定百分比间隔的值
    percentages = list(range(0, 101, percentage_interval))
    percentiles = np.percentile(column_data, percentages)
    
    # 绘制分布直方图, desnity=True表示相对频率
    # plt.hist(column_data, bins=20, density=True, alpha=0.6, color='b')
    plt.hist(column_data, bins=20, density=False, alpha=1, color='b')
    plt.xlabel(r'$\alpha$')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Number of masks', rotation=90)
    # plt.title(f'{column_name} Distribution')
    plt.savefig(save_pdf_path, bbox_inches='tight')
    plt.close()
    
    # 计算频率分布并绘制频率分布直方图
    frequency_data = np.histogram(column_data, bins=20)
    frequency = frequency_data[0] / len(column_data)  # 计算频率
    bin_edges = frequency_data[1]  # 直方图的箱边界

    plt.figure()  # 创建新的图表
    plt.hist(bin_edges[:-1], bin_edges, weights=frequency, color='g', alpha=0.6)
    plt.xlabel(r'$\alpha$')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Frequency', rotation=90)
    plt.savefig(save_pdf_path.replace('.pdf', '_frequency.pdf'), bbox_inches='tight')
    plt.close()

   
    # 保存统计结果到文本文件
    with open(save_txt_path, 'w') as txt_file:
        txt_file.write(f'Mean: {mean}\n')
        txt_file.write(f'Variance: {variance}\n')
        for p, percentile_value in zip(percentages, percentiles):
            txt_file.write(f'{p}%: {percentile_value}\n')

def main():
    parser = argparse.ArgumentParser(description='CSV Data Statistics')
    parser.add_argument('--csv_file', type=str, help='Path to CSV file', required=True)
    parser.add_argument('--column_name', type=str, help='CSV column name to analyze', required=True)
    parser.add_argument('--percentage_interval', type=int, default=5, help='Percentage interval for statistics')
    parser.add_argument('--alpha_thres', type=float, default=-1, help='cal alpha >= thres', required=True)
    parser.add_argument('--pdf_output', type=str, help='Path to save PDF file', required=True)
    parser.add_argument('--txt_output', type=str, help='Path to save TXT file', required=True)
    parser.add_argument("--x_label", type=str, default="x", help='x axis label')
    parser.add_argument("--y_label", type=str, default="y", help='y axis label')

    args = parser.parse_args()

    calculate_statistics(args.csv_file, args.column_name, args.percentage_interval, args.alpha_thres, args.pdf_output, args.txt_output, args)

if __name__ == '__main__':
    main()
