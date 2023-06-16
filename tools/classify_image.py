import os
import shutil

'''
function: 将一个文件夹下的图片按照城市名进行分类,并将图片移动到对应的子文件夹下
'''

# 图片所在文件夹路径
img_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/230524_2319_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6283e/trainid'

# 遍历图片文件夹下的所有文件，统计出现过的城市名
city_set = set()
for file_name in os.listdir(img_folder):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        city_name = file_name.split('_')[0]
        city_set.add(city_name)

# 创建对应的子文件夹，并将图片移动到子文件夹下
for city_name in city_set:
    city_folder = os.path.join(img_folder, city_name)
    if not os.path.exists(city_folder):
        os.makedirs(city_folder)
    for file_name in os.listdir(img_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            if file_name.startswith(city_name):
                file_path = os.path.join(img_folder, file_name)
                dest_path = os.path.join(city_folder, file_name)
                shutil.move(file_path, dest_path)
