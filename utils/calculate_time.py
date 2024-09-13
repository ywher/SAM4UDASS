import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train3/time.csv'  # 替换为实际的文件路径
data = pd.read_csv(csv_file)

# 根据mask数量分组并计算平均时间和方差
grouped = data.groupby('num_mask')['time'].agg(['mean', 'std'])
# 计算time列的平均值和方差
time_avg, time_std = data['time'].mean(), data['time'].std()
print('Average time: {:.4f}s'.format(time_avg))
print('Std of time: {:.4f}s'.format(time_std))

# 绘制折线图
plt.figure(figsize=(12, 3))

plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], fmt='o', color='blue', label='Mean ± Std Deviation')
# plt.scatter(grouped.index, grouped['mean'], s=100, c='r', marker='o', label='Mean')  # 使用绿色表示均值
# plt.bar(grouped.index, grouped['mean'], yerr=grouped['std'], color='blue', label='Mean ± Std Deviation')
plt.xlabel('Number of Masks', fontsize=16)
plt.ylabel('Time of SAM (s)', fontsize=16)
# plt.title('Time vs. Number of Masks')
plt.legend(fontsize=16)
plt.grid(True)
save_path = csv_file.replace(csv_file.split('/')[-1], 'sam_time.pdf')
plt.savefig(save_path, bbox_inches='tight')
print('Save time figure to {}'.format(save_path))
plt.show()