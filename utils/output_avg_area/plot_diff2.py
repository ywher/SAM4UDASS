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
args.add_argument("--iou_csv", type=str, default=None, help="保存iou统计结果的CSV文件路径")
args.add_argument("--iou_label", type=str, default="iou", help="iou统计结果的label")
args.add_argument("--output_pdf", type=str, help="保存统计结果的PDF文件路径")
args.add_argument("--col_name", type=str, default="avg_area", help="要统计的列的名称")
args.add_argument("--divide_img_area", action="store_true", help="是否除以图像面积")
args.add_argument("--bar_width", type=float, default=0.2, help="保存统计结果的PNG文件路径")
args.add_argument("--width", type=int, default=2048, help="the width of the image")
args.add_argument("--height", type=int, default=1024, help="the height of the image")
args.add_argument("--selected_classes", nargs='+', type=int, default=[], help="要单独展示的类别的数字列表")
args.add_argument("--fig_width", type=int, default=8, help="the width of the figure")
args.add_argument("--fig_height", type=int, default=6, help="the height of the figure")
args = args.parse_args()
print(args.csv_file1)
print(args.csv_file2)
print(args.output_pdf)

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(os.path.dirname(args.output_pdf)):
    os.makedirs(os.path.dirname(args.output_pdf))

# 是否将数值除以图像面积
if args.divide_img_area:
    image_area = args.width * args.height
else:
    image_area = 1

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

class_names = [
    'road', 'sw', 'build', 'wall', 'fence', 'pole', 'light',
    'sign', 'vege', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motor', 'bike'
]

y_label = "average area ratio"

# 读取两个CSV文件
df1 = pd.read_csv(args.csv_file1)
df2 = pd.read_csv(args.csv_file2)
df3 = pd.read_csv(args.iou_csv)
iou_data = np.array(df3.iloc[0]) * 0.01
print(iou_data)

# 选择要单独展示的类别的数据
selected_classes_data1 = df1[df1['id'].isin(args.selected_classes)]
selected_classes_data2 = df2[df2['id'].isin(args.selected_classes)]
selected_iou_data = iou_data[args.selected_classes]
selected_class_names = [class_names[i] for i in args.selected_classes]

# 设置x轴的位置
selected_x =  np.array(range(len(selected_class_names)))
x1 = np.array(df1['id'].values) - args.bar_width / 2
x2 = np.array(df2['id'].values) + args.bar_width / 2
selected_x1 = selected_x - args.bar_width / 2
selected_x2 = selected_x + args.bar_width / 2

# x1 = np.array(df1['id'].values) - args.bar_width
# x2 = np.array(df2['id'].values)
# x3 = np.array(df2['id'].values) + args.bar_width
# selected_x1 = selected_x - args.bar_width
# selected_x2 = selected_x
# selected_x3 = selected_x + args.bar_width

# 绘制四合一条形图
plt.figure(figsize=(args.fig_width, args.fig_height))

# polt different
# three bars
# plt.bar(x1, iou_data, width=args.bar_width, color='red', label=args.iou_label)
# plt.bar(x2, df1[args.col_name] / image_area, width=args.bar_width, color='blue', label=args.label1)
# plt.bar(x3, df2[args.col_name] / image_area, width=args.bar_width, color='orange', label=args.label2, alpha=1.0)

# two bars
plt.bar(x1, df1[args.col_name] / image_area, width=args.bar_width, color='blue', label=args.label1)
plt.bar(x2, df2[args.col_name] / image_area, width=args.bar_width, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class', fontsize=14)
plt.ylabel(y_label, fontsize=14)
# plt.title('Average Area Ratio per Class')
plt.xticks(range(19), class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig(args.output_pdf, bbox_inches='tight')

# plot selected
new_fig_width = args.fig_width * len(args.selected_classes) / len(x1)
plt.figure(figsize=(new_fig_width, args.fig_height))

# three bars
# plt.bar(selected_x1, selected_iou_data, width=args.bar_width, color='red', label=args.iou_label)
# plt.bar(selected_x2, selected_classes_data1[args.col_name] / image_area, width=args.bar_width, color='blue', label=args.label1)
# plt.bar(selected_x3, selected_classes_data2[args.col_name] / image_area, width=args.bar_width, color='orange', label=args.label2, alpha=1.0)

# two bars
plt.bar(selected_x1, selected_classes_data1[args.col_name] / image_area, width=args.bar_width, color='blue', label=args.label1)
plt.bar(selected_x2, selected_classes_data2[args.col_name] / image_area, width=args.bar_width, color='orange', label=args.label2, alpha=1.0)
plt.xlabel('class', fontsize=14)
plt.ylabel(y_label, fontsize=14)
# plt.title('Average Area Ratio per Class')
plt.xticks(selected_x, selected_class_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# 保存图形为PDF文件
plt.savefig(args.output_pdf.replace('.pdf', '_selected.pdf'), bbox_inches='tight')

# # 显示图形
# plt.show()
