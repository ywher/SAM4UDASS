import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file = '/media/cyber-fx/ywh_disk/projects/SAM4UDASS/outputs/cityscapes/train3/time.csv'  # 替换为实际的文件路径
data = pd.read_csv(csv_file)

# 根据mask数量分组并计算平均时间
grouped = data.groupby('num_mask')['time'].mean()

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(grouped.index, grouped.values, marker='o')
plt.xlabel('Number of Masks')
plt.ylabel('Average Time')
plt.title('Time vs. Number of Masks')
plt.grid(True)
plt.show()
