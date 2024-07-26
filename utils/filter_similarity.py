import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# 创建参数解析器
parser = argparse.ArgumentParser(description="CSV Data Analysis")

# 添加命令行参数
parser.add_argument("--input", type=str, default='', required=True, help="输入CSV文件路径")
parser.add_argument("--output", type=str, default='', required=True, help="保存结果的路径")
parser.add_argument("--column", type=str, default='2_div_1', required=True, help="选定的列名称")
parser.add_argument("--threshold", type=float, default=0.2, required=True, help="比例阈值")
parser.add_argument("--num_categories", type=int, default=19, required=True, help="类别数量")
parser.add_argument("--stat_column_1", type=str, default='id1', required=True, help="统计的列名称1")
parser.add_argument("--stat_column_2", type=str, default='id2', required=True, help="统计的列名称2")

# 解析命令行参数
args = parser.parse_args()

# 读取CSV文件
df = pd.read_csv(args.input)

# 根据阈值筛选行
filtered_rows = df[df[args.column] >= args.threshold]

# 初始化二维矩阵
count_matrix = np.zeros((args.num_categories, args.num_categories), dtype=int)

# 遍历筛选后的行并进行统计
for _, row in tqdm(filtered_rows.iterrows(), total=len(filtered_rows), desc="Processing"):
    id1 = row[args.stat_column_1]
    id2 = row[args.stat_column_2]
    count_matrix[id1][id2] += 1

# 保存统计结果为CSV文件
count_df = pd.DataFrame(count_matrix)
count_df.to_csv(args.output, index=False)

# 保存统计结果为NumPy数组文件
np.save(args.output.replace('.csv', '.npy'), count_matrix)

print("save the results to {}".format(args.output))
