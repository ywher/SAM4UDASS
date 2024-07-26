import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np

# 解析命令行参数
args = argparse.ArgumentParser(description="分析语义分割标签图文件夹的统计信息")
args.add_argument("--csv_file1", type=str, help="第一个CSV文件路径")
args.add_argument("--label1", type=str, default="file1", help="第一个CSV文件的label")
args.add_argument("--csv_file2", type=str, help="第二个CSV文件路径")
args.add_argument("--label2", type=str, default="file2", help="第二个CSV文件的label")
args.add_argument("--output_pdf", type=str, help="保存统计结果的PDF文件路径")
args.add_argument("--bar_width", type=float, default=0.2, help="保存统计结果的PNG文件路径")
args.add_argument("--width", type=int, default=2048, help="the width of the image")
args.add_argument("--height", type=int, default=1024, help="the height of the image")
args.add_argument("--selected_classes", nargs='+', type=int, default=[], help="要单独展示的类别的数字列表")
args = args.parse_args()
print(args.csv_file1)
print(args.csv_file2)
print(args.output_pdf)

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(args.output_pdf)):
    os.makedirs(os.path.dirname(args.output_pdf))

image_area = args.width * args.height
class_names = [
    '0.road', '1.sidewalk', '2.building', '3.wall', '4.fence', '5.pole', '6.traffic light',
    '7.traffic sign', '8.vegetation', '9.terrain', '10.sky', '11.person', '12.rider',
    '13.car', '14.truck', '15.bus', '16.train', '17.motorcycle', '18.bicycle'
]

class_names = [
    '0.road', '1.side', '2.build', '3.wall', '4.fence', '5.pole', '6.light',
    '7.sign', '8.veget', '9.terra', '10.sky', '11.perso', '12.rider',
    '13.car', '14.truck', '15.bus', '16.train', '17.motor', '18.bicyc'
]

# 读取两个CSV文件
df1 = pd.read_csv(args.csv_file1)
df2 = pd.read_csv(args.csv_file2)

# 选择要单独展示的类别的数据
selected_classes_data1 = df1[df1['id'].isin(args.selected_classes)]
selected_classes_data2 = df2[df2['id'].isin(args.selected_classes)]
selected_class_names = [class_names[i] for i in args.selected_classes]

# 设置x轴的位置
selected_x =  np.array(range(len(selected_class_names)))
x1 = np.array(df1['id'].values) - args.bar_width / 2
x2 = np.array(df2['id'].values) + args.bar_width / 2
selected_x1 = selected_x - args.bar_width / 2
selected_x2 = selected_x + args.bar_width / 2

# 绘制四合一条形图
plt.figure(figsize=(20, 35))

# 绘制第一个图，横坐标为class，纵坐标为num_pixels，使用蓝色条形
plt.subplot(521)
plt.bar(x1, df1['num_pixels'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(x2, df2['num_pixels'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('num_pixels')
plt.title('Number of Pixels per Class')
plt.xticks(range(19), class_names, rotation=90)
plt.legend()

plt.subplot(522)
plt.bar(selected_x1, selected_classes_data1['num_pixels'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2['num_pixels'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('num_pixels')
plt.title('Number of Pixels per Class')
plt.xticks(selected_x, selected_class_names, rotation=90)
plt.legend()

# 绘制第二个图，横坐标为class，纵坐标为number_occur，使用蓝色条形
plt.subplot(523)
plt.bar(x1, df1['number_occur'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(x2, df2['number_occur'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('number_occur')
plt.title('Number of Occurrences per Class')
plt.xticks(range(19), class_names, rotation=90)
plt.legend()

plt.subplot(524)
plt.bar(selected_x1, selected_classes_data1['number_occur'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2['number_occur'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('number_occur')
plt.title('Number of Occurrences per Class')
plt.xticks(selected_x, selected_class_names, rotation=90)
plt.legend()

# 绘制第三个图，横坐标为class，纵坐标为avg_area，使用蓝色条形
plt.subplot(525)
plt.bar(x1, df1['avg_area'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(x2, df2['avg_area'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('avg_area')
plt.title('Average Area per Class')
plt.xticks(range(19), class_names, rotation=90)
plt.legend()

plt.subplot(526)
plt.bar(selected_x1, selected_classes_data1['avg_area'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2['avg_area'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('avg_area')
plt.title('Average Area per Class')
plt.xticks(selected_x, selected_class_names, rotation=90)
plt.legend()

# 绘制第4个图，横坐标为class，纵坐标为occur_freq，使用蓝色条形
plt.subplot(527)
plt.bar(x1, df1['occur_freq'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(x2, df2['occur_freq'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('occur_freq')
plt.title('Occur Freq per Class')
plt.xticks(range(19), class_names, rotation=90)
plt.legend()

plt.subplot(528)
plt.bar(selected_x1, selected_classes_data1['occur_freq'], width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2['occur_freq'], width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('occur_freq')
plt.title('Occur Freq per Class')
plt.xticks(selected_x, selected_class_names, rotation=90)
plt.legend()

# 绘制第5个图，横坐标为class，纵坐标为avg_area_ratio，使用蓝色条形
plt.subplot(529)
plt.bar(x1, df1['avg_area'] / image_area, width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(x2, df2['avg_area'] / image_area, width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('avg_area_ratio')
plt.title('Average Area Ratio per Class')
plt.xticks(range(19), class_names, rotation=90)
plt.legend()

plt.subplot(5,2,10)
plt.bar(selected_x1, selected_classes_data1['avg_area'] / image_area, width=args.bar_width/2, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2['avg_area'] / image_area, width=args.bar_width/2, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class')
plt.ylabel('avg_area_ratio')
plt.title('Average Area Ratio per Class')
plt.xticks(selected_x, selected_class_names, rotation=90)
plt.legend()

# 保存图形为PDF文件
plt.savefig(args.output_pdf)

# # 显示图形
# plt.show()
