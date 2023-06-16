import os
import cv2
import numpy as np
import argparse
import tqdm
import colorsys

def get_color_from_index(index):
    color = cv2.applyColorMap(np.uint8([index % 256]), cv2.COLORMAP_JET)[0][0]
    return tuple(map(int, color))

def get_color_from_index2(index):
    hue = (index * 30) % 360  # 每次增加30度的色相值，更好地区分颜色
    saturation = 1.0  # 饱和度为1，以获得鲜艳的颜色
    lightness = 0.5  # 亮度为0.5，防止颜色过亮或过暗

    # 将HSL颜色空间的值转换为0-1范围内的浮点数
    hue /= 360.0

    # 使用colorsys库将HSL颜色转换为RGB颜色
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # 将RGB值转换为0-255范围内的整数
    color = tuple(map(lambda x: int(x * 255), (r, g, b)))

    return color


def process_images(mask_folder, image_folder, output_folder, mix_ratio=0.5, mask_suffix='.png', img_suffix='.png'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    #output rgb folder
    output_rgb_folder = os.path.join(output_folder, 'rgb')
    if not os.path.exists(output_rgb_folder):
        os.makedirs(output_rgb_folder)
    #output gray folder
    output_gray_folder = os.path.join(output_folder, 'gray')
    if not os.path.exists(output_gray_folder):
        os.makedirs(output_gray_folder)
        
    
    subfolders = [f.path for f in os.scandir(mask_folder) if f.is_dir()]
    subfolders.sort()
    # img_suffix = '.'+os.listdir(image_folder)[0].split('.')[-1]
    bar = tqdm.tqdm(total=len(subfolders))
    for subfolder in subfolders:
        image_name = os.path.basename(subfolder) #000228
        image_name += img_suffix
        # image_name.remove('metadata')
        image_path2 = os.path.join(image_folder, image_name)

        original_image = cv2.imread(image_path2)

        height, width, _ = original_image.shape
        result_image = np.zeros((height, width, 3), dtype=np.float32)
        result_gray_image = np.zeros((height, width, 3), dtype=np.float32)

        color_index = 0
        for image_file in os.listdir(subfolder):
            if mask_suffix not in image_file:
                continue
            image_path1 = os.path.join(subfolder, image_file)
            black_white_image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

            mask = black_white_image == 255
            color = get_color_from_index2(color_index)
            # color = np.array(cv2.applyColorMap(np.uint8([color_index%256]), cv2.COLORMAP_JET)[0][0], dtype=np.float32)
            result_image[mask] = color
            result_gray_image[mask] = np.array((255,255,255))

            color_index += 2

        blended_image = cv2.addWeighted(result_image, mix_ratio, original_image.astype(np.float32), 1-mix_ratio, 0)
        output_image_path = os.path.join(output_rgb_folder, image_name)
        cv2.imwrite(output_image_path, blended_image)
        cv2.imwrite(output_image_path.replace(output_rgb_folder, output_gray_folder), result_gray_image)
        bar.update(1)
    bar.close()

def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mask_folder', type=str, help='the path to the segment anything result',
                       default='/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/dianyuan')
    parse.add_argument('--image_folder', type=str, help='the path to the original image path',
                       default='/media/ywh/1/yanweihao/dataset/dianyuan_driving/image_2')
    parse.add_argument('--output_folder', type=str, help='output dir', default='outputs/dianyuan_mix')
    parse.add_argument('--mix_ratio', type=float, default=0.5, help='image mixing ratio')
    parse.add_argument('--mask_suffix', type=str, default='.png')
    parse.add_argument('--img_suffix', type=str, default='.png')
    return parse.parse_args()

if __name__ == "__main__":
    args = get_parse()

    process_images(args.mask_folder, args.image_folder, args.output_folder,
                   args.mix_ratio, args.mask_suffix, args.img_suffix)
