import pandas as pd

# 读取原始CSV文件
input_file = 'Differ_1_0_Differ_1_0.csv'
data = pd.read_csv(input_file)  # 假设使用制表符分隔

# 根据条件筛选出满足要求的行
condition = (data['Value'] > 0) & (data['Fusion 3-Fusion 1'] > 0) & (data['Class_Fusion 3-Fusion 1'] > 0)
filtered_data = data[condition]

# 将筛选后的数据保存为新的CSV文件
output_file = 'filtered_output.csv'
filtered_data.to_csv(output_file, index=False)  # 写入制表符分隔的新CSV文件
