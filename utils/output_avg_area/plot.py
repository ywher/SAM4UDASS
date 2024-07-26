import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

args=argparse.ArgumentParser(description="分析语义分割标签图文件夹的统计信息")
args.add_argument("--csv_file", type=str, help="保存统计结果的CSV文件路径")
args.add_argument("--output_pdf", type=str, help="保存统计结果的PDF文件路径")
args.add_argument("--width", type=int, default=2048, help="the width of the image")
args.add_argument("--height", type=int, default=1024, help="the height of the image")
args = args.parse_args()
print(args.csv_file)
print(args.output_pdf)

# check whether the output folder exists
if not os.path.exists(os.path.dirname(args.output_pdf)):
    os.makedirs(os.path.dirname(args.output_pdf))

image_area = args.width * args.height
# class_names = [
#     '0.road', '1.sidewalk', '2.building', '3.wall', '4.fence', '5.pole', '6.traffic light',
#     '7.traffic sign', '8.vegetation', '9.terrain', '10.sky', '11.person', '12.rider',
#     '13.car', '14.truck', '15.bus', '16.train', '17.motorcycle', '18.bicycle'
#     ]
class_names = [
    'road', 'sw', 'build', 'wall', 'fence', 'pole', 'light',
    'sign', 'vege', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motor', 'bike'
]

# 读取CSV文件
df = pd.read_csv(args.csv_file)

# 绘制四合一条形图
plt.figure(figsize=(20, 18))

# 绘制第一个图，横坐标为class，纵坐标为num_pixels
plt.subplot(321)
plt.bar(df['id'], df['num_pixels'])
plt.xlabel('class')
plt.ylabel('number of pixels')
# plt.title('Number of Pixels per Class')
plt.xticks(range(19), class_names, rotation=90)

# 绘制第二个图，横坐标为class，纵坐标为number_occur
plt.subplot(322)
plt.bar(df['id'], df['number_occur'])
plt.xlabel('class')
plt.ylabel('number of occurrences')
# plt.title('Number of Occurrences per Class')
plt.xticks(range(19), class_names, rotation=90)

# 绘制第三个图，横坐标为class，纵坐标为avg_area
plt.subplot(323)
plt.bar(df['id'], df['avg_area'])
plt.xlabel('class')
plt.ylabel('average area')
# plt.title('Average Area per Class')
plt.xticks(range(19), class_names, rotation=90)

# 绘制第4个图，横坐标为class，纵坐标为avg_area
plt.subplot(324)
plt.bar(df['id'], df['occur_freq'])
plt.xlabel('class')
plt.ylabel('occurrence frequency')
# plt.title('Occur Freq per Class')
plt.xticks(range(19), class_names, rotation=90)

# 绘制第5个图，横坐标为class，纵坐标为avg_area/image_area
plt.subplot(325)
plt.bar(df['id'], df['avg_area']/image_area)
plt.xlabel('class')
plt.ylabel('average area ratio')
# plt.title('Average Area Ratio per Class')
plt.xticks(range(19), class_names, rotation=90)

# 调整图的布局
plt.tight_layout()

# 保存图形
plt.savefig(args.output_pdf)

# 分别绘制四张图并保存

# num of pixels
plt.figure(figsize=(10, 6))
plt.bar(df['id'], df['num_pixels'])
plt.xlabel('class', fontsize=14)
plt.ylabel('number of pixels', fontsize=14)
# plt.title('Number of Pixels per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_num_pixel.pdf'), bbox_inches='tight')

# num of occurance
plt.figure(figsize=(10, 6))
plt.bar(df['id'], df['number_occur'])
plt.xlabel('class', fontsize=14)
plt.ylabel('number of occurrences', fontsize=14)
# plt.title('Number of Occurrences per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_num_occur.pdf'), bbox_inches='tight')

# num of average area
plt.figure(figsize=(10, 6))
plt.bar(df['id'], df['avg_area'])
plt.xlabel('class', fontsize=14)
plt.ylabel('average area', fontsize=14)
# plt.title('Average Area per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_avg_area.pdf'), bbox_inches='tight')

# num of occur freq
plt.figure(figsize=(10, 6))
plt.bar(df['id'], df['occur_freq'])
plt.xlabel('class', fontsize=14)
plt.ylabel('occurrence frequency', fontsize=14)
# plt.title('Occur Freq per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
y_ticks = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
y_ticks_label = ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00']
plt.yticks(y_ticks, y_ticks_label, fontsize=14)
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_occur_freq.pdf'), bbox_inches='tight')

# num of average area ratio
avg_area_ratio = df['avg_area']/image_area
plt.figure(figsize=(10, 6))
plt.bar(df['id'], avg_area_ratio)
plt.xlabel('class', fontsize=14)
plt.ylabel('average area ratio', fontsize=14)
# plt.title('Average Area Ratio per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_avg_area_ratio.pdf'), bbox_inches='tight')
# save the avg_area_ratio to csv file
df['avg_area_ratio'] = avg_area_ratio
df.to_csv(args.csv_file, index=False)

# 绘制双条形图
# 数据和设置
# 数据和设置
plt.figure(figsize=(10, 6))

bar_width = 0.4
# 绘制第一个柱形图（Average Area Ratio）
bar1=plt.bar(df['id']-bar_width/2, avg_area_ratio, width=bar_width, color='blue', alpha=0.7, label='Average Area Ratio')
plt.xlabel('Class', fontsize=14)
plt.ylabel('Average Area Ratio', fontsize=14)  # color='blue',
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)

# 创建一个共享横坐标的另一个y轴（右侧）
ax2 = plt.twinx()

# 绘制第二个柱形图（Occurrence Frequency）
bar2=ax2.bar(df['id']+bar_width/2, df['occur_freq'], width=bar_width, color='orange', alpha=0.7, label='Occurrence Frequency')
ax2.set_ylabel('Occurrence Frequency', fontsize=14) # color='blue', 
ax2.yaxis.set_tick_params(labelsize=14)

# 合并图例
bars = [bar1, bar2]
labels = [bar.get_label() for bar in bars]
plt.legend(bars, labels, loc='upper right', fontsize=12)

# 保存图像
plt.tight_layout()
plt.savefig(args.output_pdf.replace('.pdf', '_dual_bar_plot.pdf'), bbox_inches='tight')
