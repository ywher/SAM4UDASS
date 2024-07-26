# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from PIL import Image, ImageDraw

def create_mask(x1, y1, x2, y2, x3, y3, x4, y4, w, h):
    # 创建全零掩膜数组
    mask = np.zeros((h, w), dtype=np.uint8)

    # 创建 PIL 图像对象
    img = Image.fromarray(mask)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 绘制四边形并填充为白色
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=255)

    # 将 PIL 图像转换回 numpy 数组
    mask = np.array(img)

    # 二值化为二进制掩膜
    mask = (mask > 0).astype(np.uint8)

    return mask

def calculate_overlap_ratio(mask1, mask2):
    # 计算两个掩膜中前景像素的逻辑与
    overlap = np.logical_and(mask1, mask2)

    # 计算重叠面积
    overlap_area = np.sum(overlap)

    # 计算两个掩膜前景面积
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)

    # 计算重叠面积占两个掩膜前景面积的比例
    overlap_ratio = overlap_area / area1

    return overlap_ratio

def show_anns(anns, mask_index):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True) #descend
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    index = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        #save the mask in order
        mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
        mask[m] = 255
        cv2.imwrite('outputs_tmp/mask_{}_{}.png'.format(mask_index, index), mask)
        index += 1
    ax.imshow(img)
    
    
def show_anns2(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True) #descend
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    w,h = sorted_anns[0]['segmentation'].shape[1], sorted_anns[0]['segmentation'].shape[0]
    mask1 = create_mask(136, 0, 247, 0, 136, 540, 247, 715, w,h)
    mask2 = create_mask(120, 1780, 120, 2063, 247, 2063, 247, 1386, w,h)
    mask3 = create_mask(210, 1233, 210, 1288, 247, 1288, 247, 1083, w,h)
    cv2.imwrite('mask1.png', mask1)
    cv2.imwrite('mask2.png', mask2)
    cv2.imwrite('mask3.png', mask3)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    index = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        if calculate_overlap_ratio(m, mask1) > 0.5 or calculate_overlap_ratio(m, mask2) > 0.5 or calculate_overlap_ratio(m, mask3) > 0.5:
            color_mask = np.concatenate([[0,1,0], [0.35]])
            img[m] = color_mask
            #save the mask in order
            mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            mask[m] = 255
            # cv2.imwrite('mask_{}.png'.format(index), mask)
            index += 1
    ax.imshow(img)


#define sam model
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h" #
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

#load image
# iamge_name = 'city.jpg'
iamge_name = 'images/huanshi.jpg'
image = cv2.imread(iamge_name)
# image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
weight, height = image.shape[1], image.shape[0]
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

#get the mask
masks = mask_generator.generate(image)
'''
Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

segmentation : the mask
area : the area of the mask in pixels
bbox : the boundary box of the mask in XYWH format
predicted_iou : the model's own prediction for the quality of the mask
point_coords : the sampled input point that generated this mask
stability_score : an additional measure of mask quality
crop_box : the crop of the image used to generate this mask in XYWH format
'''
print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks,1)
plt.axis('off')
plt.savefig('outputs_tmp/mask1.png', bbox_inches='tight', pad_inches=0)
plt.show() 


'''
Automatic mask generation options
There are several tunable parameters in automatic mask generation that 
control how densely points are sampled and what the thresholds are for 
removing low quality or duplicate masks. 
Additionally, generation can be automatically run on crops of the image to 
get improved performance on smaller objects, 
and post-processing can remove stray pixels and holes. 
Here is an example configuration that samples more masks:
'''

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks2 = mask_generator_2.generate(image)

print(len(masks2))

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2,2)
plt.axis('off')
plt.savefig('outputs_tmp/mask2.png', bbox_inches='tight', pad_inches=0)
plt.show() 


mask_generator_3 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks3 = mask_generator_3.generate(image)
print(len(masks3))

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks3,3)
plt.axis('off')
plt.savefig('outputs_tmp/mask3.png', bbox_inches='tight', pad_inches=0)
plt.show() 


