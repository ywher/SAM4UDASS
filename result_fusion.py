'''
function: fusion the result of model prediction and seg-anything
input: the path to the model prediction,
    the path to the seg-anything result,
    the path to the original image,
    the path to save the fusion result
output: the fusion result in trainID, colored fusion result mixed with original image
'''

import argparse
import os
import cv2
from cityscapesscripts.helpers.labels import trainId2label as trainid2label
import numpy as np
import tqdm
from utils.shrink_mask import shrink_region
import pandas as pd
import copy
import matplotlib.pyplot as plt
from natsort import natsorted
from utils.mask_shape import Mask_Shape
from utils.cal_mask_center import cal_center, inside_rect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from segment_anything import sam_model_registry, SamPredictor
from tools.iou_perimg import SegmentationMetrics
from utils.segmentix import Segmentix


def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mask_folder', type=str, help='the path to the segment anything result',
                       default='/media/yons/pool1/ywh/projects/Segmentation/segment-anything/outputs/cityscapes/train')
    parse.add_argument('--segmentation_folder', type=str, help='the path to the model prediction',
                       default='/media/yons/pool1/ywh/projects/UDA/MIC/seg/work_dirs/local-exp80/230422_0820_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_3e-05_s0_21197/pred_trainid')
    parse.add_argument('--image_folder', type=str, help='the path to the original image',
                       default='/media/yons/pool1/ywh/dataset/cityscapes/leftImg8bit/train_all')
    parse.add_argument('--gt_folder', type=str, help='the path to the ground truth',
                       default='/media/yons/pool1/ywh/dataset/cityscapes/gtFine/train_all')
    parse.add_argument('--mix_ratio', type=float, help='the ratio of the model prediction', default=0.5)
    parse.add_argument('--resize_ratio', type=float, help='the resize ratio of mix image', default=0.5)
    parse.add_argument('--output_folder', type=str, help='the path to save the fusion result',
                       default='outputs/cityscapes/train_fusion_1')
    parse.add_argument('--mask_suffix', type=str, help='the suffix of the mask', default='.png')
    parse.add_argument('--segmentation_suffix', type=str, help='the suffix of the segmentation result', default='_trainID.png')
    parse.add_argument('--segmentation_suffix_noimg', action='store_true', help='the suffix of the segmentation result', default=False)
    parse.add_argument('--fusion_mode', type=int, default=0, help='which type to fuse sam and uda, \
                        0: sam mask first, background using uda,\
                        1: uda pred first, choose some classes in sam, \
                        2: uda pred with confidence higher than threshold first, then choose some classes in sam, background using uda \
                        3: after mode 0, use eroded class mask in uda pred to cover sam mask')
    parse.add_argument('--sam_classes', type=list, default=[5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18], help='the classes sam performs better')
    parse.add_argument('--shrink_num', type=int, default=2, help='the shrink num of segmentation mask')
    return parse.parse_args()


class Fusion():
    def __init__(self, args):
        # the path to the sam mask
        self.mask_folder = args.mask_folder
        # the path to the uda prediction
        self.segmentation_folder = args.segmentation_folder
        # the path to the original image
        self.image_folder = args.image_folder
        # the mix ratio of the fusion result and origianl image
        self.mix_ratio = args.mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = args.resize_ratio
        # the path to the output folder
        self.output_folder = args.output_folder
        # the image suffix of the mask and segmentation result
        self.mask_suffix = args.mask_suffix
        self.segmentation_suffix = args.segmentation_suffix
        self.segmentation_suffix_noimg = args.segmentation_suffix_noimg
        # the fusion mode
        self.fusion_mode = args.fusion_mode
        # the classes sam performs better
        self.sam_classes = args.sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = args.shrink_num

        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)
        self.image_names.sort()

        # make the folder to save the fusion result
        # the fusion result in trainID
        self.check_and_make(os.path.join(self.output_folder, 'trainID'))
        # the fusion result in color
        # self.check_and_make(os.path.join(self.output_folder, 'color'))
        # the fusion result in color mixed with original image
        self.check_and_make(os.path.join(self.output_folder, 'mixed'))
        # make the folder to save the fusion result with segmentation result as the background
        # the fusion result in trainID with segmentation result as the background
        self.check_and_make(os.path.join(self.output_folder, 'trainID_bg'))
        # the fusion result in color with segmentation result as the background
        # self.check_and_make(os.path.join(self.output_folder, 'color_bg'))
        # the fusion result in color mixed with original image with segmentation as background
        self.check_and_make(os.path.join(self.output_folder, 'mixed_bg'))

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

    def get_sam_pred(self, image_name, segmentation):
        '''
        use the mask from sam and the prediction from uda
        output the train id and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]

        # sort the mask names according to the mask area from large to small
        # mask_areas = []
        # for mask_name in mask_names:
        #     mask_path = os.path.join(self.mask_folder, image_name, mask_name)
        #     mask = cv2.imread(mask_path)  # [h,w,3]
        #     mask_area = np.sum(mask[:, :, 0] == 255)
        #     mask_areas.append(mask_area)
        # mask_names = [mask_name for _, mask_name in sorted(zip(mask_areas, mask_names), reverse=True)]
        
        sam_mask = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        for mask_name in mask_names:
            mask_path = os.path.join(self.mask_folder, image_name, mask_name)
            mask = cv2.imread(mask_path)  # [h,w,3]
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]
            num_ids, counts = np.unique(trainids, return_counts=True)
            # get the most frequent trainid
            most_freq_id = num_ids[np.argmax(counts)]
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id
        
        return sam_mask
        
    def color_segmentation(self, segmentation):
        #get the color segmentation result, initial the color segmentation result with black (0,0,0)
        #input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        for train_id in train_ids:
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
        
    def fusion(self):
        bar = tqdm.tqdm(total=len(self.image_names))
        for image_name in self.image_names:
            #get the segmentation result
            prediction_path = os.path.join(self.segmentation_folder, image_name + self.segmentation_suffix)
            if self.segmentation_suffix_noimg:
                prediction_path = prediction_path.replace('_leftImg8bit', '')
            # print('load from: ', prediction_path)
            segmentation = cv2.imread(prediction_path) #[h, w, 3], 3 channels not 1 channel
            # print('prediction_path', prediction_path)
            
            #get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            
            #get the sam segmentation result using the mask
            sam_pred = self.get_sam_pred(image_name, segmentation)
            sam_color = self.color_segmentation(sam_pred)
            
            #get the mixed color image using the self.mix_ratio
            mixed_color = cv2.addWeighted(original_image, self.mix_ratio, sam_color, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color = cv2.resize(mixed_color, (int(mixed_color.shape[1] * self.resize_ratio), int(mixed_color.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            
            #save the sam mask in trainid and color to the output folder
            cv2.imwrite(os.path.join(self.output_folder, 'trainID', image_name + self.mask_suffix), sam_pred)
            # cv2.imwrite(os.path.join(self.output_folder, 'color', image_name + self.mask_suffix), fusion_color)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed', image_name + self.mask_suffix), mixed_color)

            #get the fusion result with the background
            if self.fusion_mode == 0:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_0(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 1:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 3:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
            else:
                # raise NotImplementedError
                raise NotImplementedError("This fusion mode has not been implemented yet.")
                             
            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), fusion_trainid_bg)
            # cv2.imwrite(os.path.join(self.output_folder, 'color_bg', image_name + self.mask_suffix), fusion_color_bg)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)
            # fusion_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            
            bar.update(1)
            # mask = cv2.imread(os.path.join(self.mask_folder, image_name + self.mask_suffix))
            
            # image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
            # mask = mask / 255
            # segmentation = segmentation / 255
            # mask = mask * self.mix_ratio
            # segmentation = segmentation * (1 - self.mix_ratio)
            # fusion = mask + segmentation
            # fusion = fusion * 255
            # fusion = fusion.astype('uint8')
            # fusion = cv2.cvtColor(fusion, cv2.COLOR_GRAY2BGR)
            # fusion = cv2.addWeighted(image, 0.5, fusion, 0.5, 0)
            # cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), fusion)
    
    def fusion_mode_0(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        train_ids = np.unique(sam_pred)
        train_ids = train_ids[train_ids != 255]
        for train_id in train_ids:
            fusion_trainid[sam_pred == train_id] = train_id
        # fusion_color = self.color_segmentation(fusion_trainid)
        
        #use the segmentation result to fill the pixels in fusion_trainid whose trainid is 255
        fusion_trainid_bg = fusion_trainid.copy()
        indexs = np.where(fusion_trainid == 255)
        fusion_trainid_bg[indexs] = segmentation[:, :, 0][indexs]      
        #use the corresponding color of segmentation result to fill the pixels in fusion_color whose trainid is 255
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg)
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)
        fusion_color_bg = fusion_color_bg.astype(np.uint8)
        
        return fusion_trainid_bg, fusion_color_bg
    
    def fusion_mode_1(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        fusion_trainid = segmentation[:, :, 0].copy() #fill the fusion result all with segmentation result
        
        #use sam_pred with self.sam_classes to cover the fusion_trainid
        sam_pred_ids = np.unique(sam_pred)
        sam_pred_ids = sam_pred_ids[sam_pred_ids != 255]
        for sam_class in self.sam_classes:
            if sam_class in sam_pred_ids:
                fusion_trainid[sam_pred == sam_class] = sam_class
        
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
     
    def fusion_mode_3(self, segmentation, sam_pred):
        fusion_trainid, fusion_color = self.fusion_mode_0(segmentation=segmentation, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]
        
        for class_id in unique_classes:
            #get the class mask in segmentation
            class_mask = (segmentation == class_id)
            #eroded the class mask in segmentation
            eroded_class_mask, area = shrink_region(class_mask, num_pixels=self.shrink_num)
            #assign the corresponding area in fusion_trainid with the class_id
            fusion_trainid[eroded_class_mask] = class_id
        fusion_trainid = fusion_trainid.astype(np.uint8)            
        
        fusion_color = self.color_segmentation(fusion_trainid)
        return fusion_trainid, fusion_color
    
    def trainid2color(self, trainid):
        '''
        function: convert trainID to color in cityscapes
        input: trainid
        output: color
        '''
        # if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask
            
        
        
        # if type(trainid) == tuple: #a mask
        #     ###assign the color to the mask according to the trainid
        #     color_mask = np.zeros((trainid.shape[0], trainid.shape[1], 3), dtype=np.uint8)
        #     for i in range(trainid.shape[0]):
        #         for j in range(trainid.shape[1]):
        #             label_object = trainid2label[trainid[i, j]]
        #             color_mask[i, j] = label_object.color
        #     return color_mask
        # else: #one number
        #     label_object = trainid2label[trainid]
        #     return label_object.color

class Fusion_SYN():
    def __init__(self, mask_folder=None, segmentation_folder=None, confidence_folder=None, entropy_folder=None,
                 image_folder=None, gt_folder=None, num_classes=None, road_center_rect=None,
                 mix_ratio=None, resize_ratio=None, output_folder=None, mask_suffix=None,
                 segmentation_suffix=None, segmentation_suffix_noimg=None,
                 confidence_suffix=None, entropy_suffix=None, gt_suffix=None,
                 fusion_mode=None, sam_classes=None, shrink_num=None, display_size=(200, 400),
                 sam_model_type='vit_h', sam_model_path='./models/sam_vit_h_4b8939.pth', device='cuda:0'):
        # the path to the sam mask
        self.mask_folder = mask_folder
        # the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        # the path to the confidence map
        self.confidence_folder = confidence_folder
        self.confidence_suffix = confidence_suffix
        # the path to the entropy map
        self.entropy_folder = entropy_folder
        self.entropy_suffix = entropy_suffix
        # the number of classes
        self.num_classes = num_classes
        # the rect of the road center
        self.road_center_rect = road_center_rect
        # the path to the ground truth folder
        self.gt_folder = gt_folder
        # self.gt_color_folder = self.gt_folder.replace('train_all', 'train_gt_color')
        # the path to the original image
        self.image_folder = image_folder
        # the mix ratio of the fusion result and original image
        self.mix_ratio = mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        # the path to the output folder
        self.output_folder = output_folder
        # the image suffix of the mask and segmentation result
        self.mask_suffix = mask_suffix
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        # the gt suffix
        self.gt_suffix = gt_suffix
        # the fusion mode
        self.fusion_mode = fusion_mode
        # the classes sam performs better
        self.sam_classes = sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = shrink_num
        # the size of the image
        self.display_size = display_size
        # initialize sam model
        sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path)
        sam.to(device)
        self.sam_predictor = SamPredictor(sam)
        self.segmtrix = Segmentix()
        self.iou_cal = SegmentationMetrics(num_classes=num_classes)
        # the label names
        self.label_names = [trainid2label[train_id].name for train_id in range(19)]
        if self.num_classes == 16:
            self.label_names.remove('train')
            self.label_names.remove('truck')
            self.label_names.remove('terrain')
        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)
        self.image_names.sort()

        # make the folder to save the fusion result
        # the fusion result in trainID
        if self.output_folder is not None:
            self.check_and_make(os.path.join(self.output_folder, 'trainID'))
            # the fusion result in color
            # self.check_and_make(os.path.join(self.output_folder, 'color'))
            # the fusion result in color mixed with original image
            self.check_and_make(os.path.join(self.output_folder, 'mixed'))
            # make the folder to save the fusion result with segmentation result as the background
            # the fusion result in trainID with segmentation result as the background
            self.check_and_make(os.path.join(self.output_folder, 'trainID_bg'))
            # self.check_and_make(os.path.join(self.output_folder, 'color_bg'))
            # the fusion result in color with segmentation result as the background
            # the fusion result in color mixed with original image with segmentation 
            # result as the background
            self.check_and_make(os.path.join(self.output_folder, 'horizontal'))
            self.check_and_make(os.path.join(self.output_folder, 'mixed_bg'))
            self.check_and_make(os.path.join(self.output_folder, 'ious'))

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

    def mask_merge_by_stability(self, prompt_masks, scores):
        """
        func: merge the masks by the stability, one pixel may belong to many masks
        prompt_masks : dict{mask_id : np.array}
        score : dict{mask_id : float}
        return:
            merged_mask: np.array, 0 - num_instances, 1000 is background
            score_map: np.array, the stability score of each pixel
        """
        # less than 1000 instances
        merged_mask = (np.zeros_like(list(prompt_masks.values())[0]) + 1000)
        score_map = np.zeros_like(merged_mask, dtype=float)

        for mask_id in prompt_masks:
            higher_score_region = (prompt_masks[mask_id] > 0) & (score_map < scores[mask_id])
            merged_mask[higher_score_region] = mask_id
            score_map[higher_score_region] = scores[mask_id]
            # print(f"mask id is {class_names[mask_id]}, mask score is {scores[mask_id]}")

        return merged_mask, score_map

    def mask_merge(self, prompt_masks, scores, possibilities):
        """
            prompt_masks : dict{mask_id : np.array}
            score : dict{mask_id : float}
        """
        merged_mask = (np.zeros_like(list(prompt_masks.values())[0]) + 255)
        score_map = np.zeros_like(merged_mask, dtype=float)
        possibility_map = np.zeros_like(merged_mask, dtype=float)

        for mask_id in prompt_masks:
            higher_score_region = (prompt_masks[mask_id] > 0) & (score_map < scores[mask_id])
            merged_mask[higher_score_region] = mask_id
            score_map[higher_score_region] = scores[mask_id]
            # print(f"mask id is {class_names[mask_id]}, mask score is {scores[mask_id]}")

        # import pdb;pdb.set_trace()

        # fill the blank region, still some problem,
        # the gap between wall and car cannot be sidewalk
        blank_region = (np.zeros_like(list(prompt_masks.values())[0]) + 255)

        for mask_id in possibilities:
            higher_poss_region = (possibility_map < possibilities[mask_id]) & (merged_mask == 255)
            blank_region[higher_poss_region] = mask_id
            possibility_map[higher_poss_region] = possibilities[mask_id][higher_poss_region]
            # print(mask_id)
            # plt.imshow(possibilities[mask_id][higher_poss_region], cmap = 'gray')
            # plt.show()
        merged_mask_bg = copy.deepcopy(merged_mask)
        merged_mask_bg[merged_mask == 255] = blank_region[merged_mask == 255]
        return merged_mask, merged_mask_bg

    def get_prompt_point(self, bin_mask_image):
        # 给uda中每个类别的mask, 对每个联通区域计算它们的单个质心作为点的prompt
        # 进行连通区域提取
        connectivity = 8  # 连通性，4代表4连通，8代表8连通
        output = cv2.connectedComponentsWithStats(bin_mask_image, connectivity, cv2.CV_32S)

        # 获取连通区域的数量
        num_labels = output[0]

        # 获取连通区域的属性
        labels = output[1]
        # stats = output[2]

        cps = []

        # 循环遍历每个连通区域
        for i in range(1, num_labels):
            # 获取连通区域的左上角坐标和宽高
            # x = stats[i, cv2.CC_STAT_LEFT]
            # y = stats[i, cv2.CC_STAT_TOP]
            # width = stats[i, cv2.CC_STAT_WIDTH]
            # height = stats[i, cv2.CC_STAT_HEIGHT]

            # if width * height < 00:
            #     continue

            contours, _ = cv2.findContours(np.uint8(labels == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 计算区域的质心
            M = cv2.moments(contours[0])
            if M["m00"] == 0:
                continue
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # 绘制连通区域的外接矩形
            center_point = (center_x, center_y)

            if bin_mask_image[center_point[1], center_point[0]]:
                cps.append(center_point)
            else:
                points = np.where(labels == i)
                idx = np.random.choice(list(range(len(points[0]))))
                cps.append([points[1][idx], points[0][idx]])
        return cps

    def generate_segementation_by_sam(self, image, autogenerator_mask, uda_mask, confidence_mask, entropy_mask):
        """
        func: generate pseudo label for sam by referring to uda_mask
            image : np.array
            autogenrator_mask : np.array, uint8
            uda_mask : np.array, uint8
        output:
            merged_mask: np.array, uint8, without holes
            sam_mask_result: np.array, uint8, with holes
        """

        # Load image to extract image embedding
        self.sam_predictor.set_image(image)
        sam_mask_result = np.zeros_like(autogenerator_mask).astype(np.uint8) + 255

        # Give auto generated mask corresponding semantic label
        # can update to the get_sam_pred function in result_fusion.py
        unique_ids = np.unique(autogenerator_mask)
        unique_ids.sort()
        for submask_id in unique_ids:
            if submask_id == 1000:
                break
            submask = autogenerator_mask == submask_id
            trainids = uda_mask[submask]  # [N,]
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n, ], [n1, n2, n3, ...]
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            # get the most frequent trainid
            most_freq_id = num_ids[0]

            if len(counts) >= 2:
                if num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.15:
                    # [building, pole]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.15:
                    # [building, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.1:
                    # [building, traffic sign]
                    mask = np.zeros_like(submask).astype(np.uint8)
                    mask[submask] = 255
                    # from [H,W] to [H,W,3]
                    mask = np.expand_dims(mask, axis=2)
                    mask = np.repeat(mask, 3, axis=2)
                    mask_shape = Mask_Shape(mask)
                    if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 8 and num_ids[1] == 9 and counts[1] / counts[0] >= 0.05:
                    # [traffic sign, vegetation]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    # [wall, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 9 and num_ids[1] == 1:
                    # [terrain, sidewalk]
                    num_id_0 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[0], submask), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[1], submask), entropy_mask))
                    if num_id_1 > num_id_0:
                        most_freq_id = num_ids[1]
                # for synthia
                elif num_ids[0] == 8 and num_ids[1] == 1:
                    # [vegetation, sidewalk]
                    num_id_0 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[0], submask), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[1], submask), entropy_mask))
                    if num_id_0 / counts[0] > 0.1 and self.num_classes == 19:
                        most_freq_id = 9  # terrrain
                    elif counts[1] / counts[0] >= 0.25:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 8 and num_ids[1] == 2:
                    # [vegetation, building], 窗户被判断为vegetation
                    num_id_0 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[0], submask), confidence_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(uda_mask == num_ids[1], submask), confidence_mask))
                    if num_id_0 == 0 or num_id_1 / num_id_0 > 0.25:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 0 and num_ids[1] == 1:
                    # [road, sidewalk]
                    if counts[1] / counts[0] >= 0.5:
                        most_freq_id = num_ids[1]
                elif (num_ids[0] == 1 and num_ids[1] == 0) or \
                        (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                    # [sidewalk, road]
                    mask = np.zeros_like(submask).astype(np.uint8)
                    mask[submask] = 255
                    mask_center = cal_center(mask)
                    if inside_rect(mask_center, self.road_center_rect):
                        most_freq_id = 0
                    if submask_id == 0:  # 第一张图mask通常就是road
                        most_freq_id = 0
            
            # cxy's implementation
            # max_iou = 0
            # final_mask_id = 0
            # for semantic_id in np.unique(uda_mask):
            #     iou = np.sum(submask & (uda_mask == semantic_id))
            #     if max_iou < iou:
            #         max_iou = iou
            #         final_mask_id = semantic_id
            sam_mask_result[submask] = most_freq_id

        # Generate point prompt and mask prompt
        mask_point_prompts = {}
        for semantic_id in np.unique(uda_mask):
            prompt_point = self.get_prompt_point(np.uint8(sam_mask_result == semantic_id))
            prompt_mask = np.zeros_like(uda_mask, dtype = float)

            # Give uda mask lower possibility and give sam mask higher possibility 
            if np.sum(sam_mask_result == semantic_id):  # 这两个系数可以调节, 最好是有可视化的效果
                prompt_mask[uda_mask == semantic_id] = 0.6
                prompt_mask[sam_mask_result == semantic_id] = 1
            else:
                prompt_mask[uda_mask == semantic_id] = 1
            prompt_mask = self.segmtrix.reference_to_sam_mask(prompt_mask)

            # Get the prompt point from merged_mask
            mask_point_prompts[semantic_id] = {
                    "mask" : prompt_mask,
                    "points" : prompt_point
            }
                    
        # Give prompt to sam and generate mask
        score_result = {}
        mask_result = {}
        possibility_result = {}

        for semantic_id in mask_point_prompts:
            # print(bin_mask_id, np.max(bin_masks[bin_mask_id]))
            pos = mask_point_prompts[semantic_id]["points"]
            # other id's points as negative prompts
            neg = []
            for other_semantic_id in mask_point_prompts:
                if semantic_id != other_semantic_id:
                    neg += mask_point_prompts[other_semantic_id]["points"]

            mask, score, logit = self.sam_predictor.predict(
                point_coords=np.array(pos + neg),
                point_labels=np.array([1]*len(pos) + [0]*len(neg)),
                mask_input = mask_point_prompts[semantic_id]["mask"],
                multimask_output=False,
                )

            score_result[semantic_id] = score[0]  # stability score, only one value, like confidence score
            mask_result[semantic_id] = mask[0]  # 
            # possibility_result shape is [H, W]
            possibility_result[semantic_id] = self.segmtrix.turn_logits_to_possibility(logit, (image.shape[1], image.shape[0]))

        merged_mask, merged_mask_bg = self.mask_merge(mask_result, score_result, possibility_result)
        return merged_mask, merged_mask_bg, sam_mask_result

    def get_sam_pred(self, image_name, segmentation, confidence_mask=None, entropy_mask=None):
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
        mask_names = natsorted(mask_names)  #现在已经按照mask的面积进行了从大到小的排序
        
        # sort the mask names accrording to the mask area from large to small, can offline
        # mask_areas = []
        # for mask_name in mask_names:
        #     mask_path = os.path.join(self.mask_folder, image_name, mask_name)
        #     mask = cv2.imread(mask_path)  # [h,w,3]
        #     mask_area = np.sum(mask[:, :, 0] == 255)
        #     mask_areas.append(mask_area)
        # mask_names = [mask_name for _, mask_name in sorted(zip(mask_areas, mask_names), reverse=True)]
        
        sam_mask = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        for index, mask_name in  enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, image_name, mask_name)
            mask = cv2.imread(mask_path)  # [h,w,3]
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]  # [N,]
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n, ], [n1, n2, n3, ...]
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            # get the most frequent trainid
            most_freq_id = num_ids[0]
            
            if len(counts) >= 2:
                if num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.15:
                    # [building, pole]
                    # if the building is the first class and the pole is the second class, 
                    # and the ratio of pole to building is larger than 0.25
                    # then assign the mask with pole
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.15:
                    # [building, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.1:
                    # [building, traffic sign]
                    # if the building is the first class and the traffic sign is the second class,
                    mask_shape = Mask_Shape(mask)
                    if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                        # if the mask is rectangular or triangular, 
                        # then assign the mask with traffic sign
                        most_freq_id = num_ids[1]
                    # most_freq_id = num_ids[1]
                elif num_ids[0] == 8 and num_ids[1] == 9 and counts[1] / counts[0] >= 0.05:
                    # [traffic sign, vegetation]
                    # if the vegetation is the first class and the terrain is the second class,
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    # [wall, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 9 and num_ids[1] == 1:
                    # [terrain, sidewalk]
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], mask[:, :, 0] == 255), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], mask[:, :, 0] == 255), entropy_mask))
                    if num_id_1 > num_id_0:
                        most_freq_id = num_ids[1]
                # for synthia
                elif num_ids[0] == 8 and num_ids[1] == 1:
                    # [vegetation, sidewalk]
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], mask[:, :, 0] == 255), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], mask[:, :, 0] == 255), entropy_mask))
                    if num_id_0 / counts[0] > 0.1 and self.num_classes == 19:
                        most_freq_id = 9 #terrrain
                    elif counts[1] / counts[0] >= 0.25:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 8 and num_ids[1] == 2:
                    # [vegetation, building], 窗户被判断为vegetation
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    if num_id_0 ==0 or num_id_1 / num_id_0 > 0.25:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 0 and num_ids[1] == 1:
                    # [road, sidewalk]
                    if counts[1] / counts[0] >= 0.5:
                        most_freq_id = num_ids[1]
                elif (num_ids[0] == 1 and num_ids[1] == 0) or \
                    (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                    # [sidewalk, road]
                    mask_center = cal_center(mask[:, :, 0])
                    if inside_rect(mask_center, self.road_center_rect):
                        most_freq_id = 0
                    if index == 0:  #第一张图mask通常就是road
                        most_freq_id = 0
                    
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id  # 重叠的问题
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
        return sam_mask

    def color_segmentation(self, segmentation):
        # get the color segmentation result, initial the color segmentation result with black (0,0,0)
        # input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        train_ids = train_ids.astype(np.uint8)
        for train_id in train_ids:
            if train_id == 255:
                continue
            if self.num_classes == 16 and train_id in [9, 14, 16]:
                continue
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation

    def fusion(self, config):
        index_range = list(range(config.begin_index, config.begin_index + config.debug_num))
        if config.save_all_fusion:
            f1_output_folder = os.path.join(self.output_folder, 'fusion1_trainid')
            f2_output_folder = os.path.join(self.output_folder, 'fusion2_trainid')
            f3_output_folder = os.path.join(self.output_folder, 'fusion3_trainid')
            f4_output_folder = os.path.join(self.output_folder, 'fusion4_trainid')
            f5_output_folder = os.path.join(self.output_folder, 'fusion5_trainid')
            if not os.path.exists(f1_output_folder):
                os.makedirs(f1_output_folder, exist_ok=True)
            if not os.path.exists(f2_output_folder):
                os.makedirs(f2_output_folder, exist_ok=True)
            if not os.path.exists(f3_output_folder):
                os.makedirs(f3_output_folder, exist_ok=True)
            if not os.path.exists(f4_output_folder):
                os.makedirs(f4_output_folder, exist_ok=True)
            if not os.path.exists(f5_output_folder):
                os.makedirs(f5_output_folder, exist_ok=True)

        bar = tqdm.tqdm(total=config.debug_num)
        for i in index_range:
            image_name = self.image_names[i]
            # get the segmentation result
            prediction_path = os.path.join(self.segmentation_folder,
                image_name.replace('_leftImg8bit', '') + self.segmentation_suffix)
            if self.segmentation_suffix_noimg:
                prediction_path = prediction_path.replace('_leftImg8bit', '')
            # import pdb; pdb.set_trace()
            # print('load from: ', prediction_path)
            uda_mask = cv2.imread(prediction_path, 0)  # [h, w], 3 channels not 1 channel
            # import pdb; pdb.set_trace()
            uda_color = self.color_segmentation(uda_mask)

            # get the confidence map
            confidence_path = os.path.join(self.confidence_folder, image_name + self.confidence_suffix)
            pred_confidence = np.load(confidence_path)  # [h, w]

            # get the entropy map
            entropy_path = os.path.join(self.entropy_folder, image_name + self.entropy_suffix)
            pred_entropy = np.load(entropy_path)  # [h, w]

            # get the ground truth
            gt_path = os.path.join(self.gt_folder, image_name.replace('_leftImg8bit', '') + self.gt_suffix)
            gt = cv2.imread(gt_path, 0)  # [h, w, 3]
            gt_color = self.color_segmentation(gt)
            # print(np.unique(gt))

            # get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))

            # get the confidence map and entropy map
            pred_confidence = pred_confidence.astype(np.float32)
            pred_entropy = pred_entropy.astype(np.float32)
            confidence_map = self.visualize_numpy(pred_confidence)
            entropy_map = self.visualize_numpy(pred_entropy)
            confidence_mask, confidence_img = \
                self.vis_np_higher_thres(pred_confidence, original_image, config.confidence_threshold)
            entropy_threshold = np.percentile(pred_entropy, config.entropy_ratio)
            entropy_mask, entropy_img = \
                self.vis_np_lower_thres(pred_entropy, original_image, entropy_threshold)
            
            # get the mask names
            mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
            mask_names = natsorted(mask_names)  # order the mask names according to the id, from big to small

            autogenerator_masks = {}
            masks_stability_score = {}
            meta_data = pd.read_csv(os.path.join(self.mask_folder, image_name, "metadata.csv"))  

            blank_region = np.zeros((original_image.shape[0], original_image.shape[1]))

            for i, mask_name in enumerate(mask_names):
                mask_path = os.path.join(self.mask_folder, image_name, mask_name)
                mask = cv2.imread(mask_path, 0)  # [h,w,3]
                autogenerator_masks[i] = mask == 255
                masks_stability_score[i] = meta_data[meta_data["id"] == int(mask_name.split('.')[0])]["stability_score"].values[0]
                blank_region[mask==255] = True
            print(f"{np.sum(blank_region) / (blank_region.shape[0] * blank_region.shape[1])} ")
            # import pdb; pdb.set_trace()
            # TODO: add mask 
            merged_auto_mask, score_map  = self.mask_merge_by_stability(autogenerator_masks, masks_stability_score)

            # Fusion both of them
            prompt_result, prompt_result_bg, sam_pred  = \
                self.generate_segementation_by_sam(original_image, merged_auto_mask, uda_mask, confidence_mask, entropy_mask)
            sam_color = self.color_segmentation(sam_pred)
            prompt_color = self.color_segmentation(prompt_result)  # get by sam and uda_mask prompt
            prompt_color_bg = self.color_segmentation(prompt_result_bg)  # fill the holes

            # review here
            if config.save_mix_result:
                # save the fusion mask in trainid and color to the output folder
                mixed_color = cv2.addWeighted(original_image, self.mix_ratio, prompt_color, 1 - self.mix_ratio, 0)
                mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, prompt_color_bg, 1 - self.mix_ratio, 0)
                if self.resize_ratio != 1:
                    mixed_color = cv2.resize(mixed_color, (int(mixed_color.shape[1] * self.resize_ratio), int(mixed_color.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
                    mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(self.output_folder, 'trainID', image_name + self.mask_suffix), prompt_result)
                cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), prompt_result_bg)
                cv2.imwrite(os.path.join(self.output_folder, 'mixed', image_name + self.mask_suffix), mixed_color)
                cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)

            # get fusion result from 1, 2, 3, 4, 5
            fusion_trainid_bg_1, fusion_color_bg_1 = \
                self.fusion_mode_1(segmentation=uda_mask, sam_pred=sam_pred)
            fusion_trainid_bg_2, fusion_color_bg_2 = \
                self.fusion_mode_2(segmentation=uda_mask, sam_pred=sam_pred)
            fusion_trainid_bg_3, fusion_color_bg_3 = \
                self.fusion_mode_3(segmentation=uda_mask, sam_pred=sam_pred,
                                   fusion_trainid_0=fusion_trainid_bg_1,
                                   fusion_color_0=fusion_color_bg_1,
                                   confidence_mask=confidence_mask, entropy_mask=entropy_mask)
            fusion_trainid_bg_4, fusion_color_bg_4 = \
                self.fusion_mode_4(segmentation=uda_mask, sam_pred=sam_pred,
                                   fusion_trainid=fusion_trainid_bg_3, confidence_mask=confidence_mask)
            fusion_trainid_bg_5, fusion_color_bg_5 = \
                self.fusion_mode_5(segmentation=uda_mask, sam_pred=sam_pred,
                                   fusion_trainid=fusion_trainid_bg_3, entropy_mask=entropy_mask)

            miou_0, ious_0 = self.iou_cal.calculate_miou(uda_mask, gt)
            miou_5, ious_5 = self.iou_cal.calculate_miou(prompt_result_bg, gt)
            miou_1, ious_1 = self.iou_cal.calculate_miou(fusion_trainid_bg_1, gt)
            miou_2, ious_2 = self.iou_cal.calculate_miou(fusion_trainid_bg_2, gt)
            miou_3, ious_3 = self.iou_cal.calculate_miou(fusion_trainid_bg_3, gt)
            miou_4, ious_4 = self.iou_cal.calculate_miou(fusion_trainid_bg_4, gt)

            error_0 = self.get_error_image(uda_mask, gt, uda_color)
            error_5 = self.get_error_image(prompt_result_bg, gt, prompt_color_bg)
            error_1 = self.get_error_image(fusion_trainid_bg_1, gt, fusion_color_bg_1)
            error_2 = self.get_error_image(fusion_trainid_bg_2, gt, fusion_color_bg_2)
            error_3 = self.get_error_image(fusion_trainid_bg_3, gt, fusion_color_bg_3)
            error_4 = self.get_error_image(fusion_trainid_bg_4, gt, fusion_color_bg_4)

            # fusion.dis_imgs_horizontal([gt_color, sam_color, pred_color, fusion_color_bg_0], '{}_fusion0'.format(image_name.replace('_leftImg8bit', '')), miou_0)
            self.dis_imgs_horizontal(
                [original_image, gt_color, sam_color, uda_color, error_0,
                 prompt_color_bg, fusion_color_bg_1, fusion_color_bg_2, fusion_color_bg_3, fusion_color_bg_4,
                 error_5, error_1, error_2, error_3, error_4,
                 confidence_map, entropy_map, confidence_img, entropy_img],
                '{}'.format(image_name.replace('_leftImg8bit', '')),
                [(miou_0, ious_0), (miou_5, ious_5), (miou_1, ious_1),
                 (miou_2, ious_2), (miou_3, ious_3), (miou_4, ious_4)],
                [np.max(score_map), np.max(score_map)])
            self.save_ious(miou_0, ious_0, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3, miou_4, ious_4,
                           miou_5, ious_5, '{}'.format(image_name.replace('_leftImg8bit', '')))

            bar.update(1)

    def fusion_mode_1(self, segmentation, sam_pred):
        # initialize the fusion mask in trainid, fusion mask in color
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])
        fusion_trainid = np.ones_like(segmentation, dtype=np.uint8) * 255
        train_ids = np.unique(sam_pred)
        train_ids = train_ids[train_ids != 255]
        for train_id in train_ids:
            fusion_trainid[sam_pred == train_id] = train_id
        # fusion_color = self.color_segmentation(fusion_trainid)

        # use the segmentation result to fill the pixels in fusion_trainid whose trainid is 255
        fusion_trainid_bg = fusion_trainid.copy()
        indexs = np.where(fusion_trainid == 255)
        fusion_trainid_bg[indexs] = segmentation[indexs]      
        # use the corresponding color of segmentation result to fill the pixels in fusion_color whose trainid is 255
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg)
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)
        fusion_color_bg = fusion_color_bg.astype(np.uint8)

        return fusion_trainid_bg, fusion_color_bg

    def fusion_mode_2(self, segmentation, sam_pred):
        # initialize the fusion mask in trainid, fusion mask in color
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])
        fusion_trainid = np.ones_like(segmentation, dtype=np.uint8) * 255
        # fill the fusion result all with segmentation result
        fusion_trainid = segmentation.copy()

        # use sam_pred with self.sam_classes to cover the fusion_trainid
        sam_pred_ids = np.unique(sam_pred)
        sam_pred_ids = sam_pred_ids[sam_pred_ids != 255]
        for sam_class in self.sam_classes:
            if sam_class in sam_pred_ids:
                fusion_trainid[sam_pred == sam_class] = sam_class

        fusion_color = self.color_segmentation(fusion_trainid)

        return fusion_trainid, fusion_color

    def fusion_mode_3(self, segmentation, sam_pred, fusion_trainid_0=None, fusion_color_0=None, 
                      confidence_mask=None, entropy_mask=None):
        '''
        segmentation: [h, w, 3] or [h, w]
        sam_pred: [h, w]
        '''
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])
        if fusion_trainid_0 is None or fusion_color_0 is None:
            fusion_trainid_0, fusion_color_0 = self.fusion_mode_1(segmentation, sam_pred)
        else:
            # print('copy in fusion 3')
            fusion_trainid_0 = copy.deepcopy(fusion_trainid_0)
            fusion_color_0 = copy.deepcopy(fusion_color_0)
        fusion_ids = np.unique(fusion_trainid_0)
        # fusion_trainid_0: [h, w], fusion_color_0: [h, w, 3]
        # # 预测结果为road但是sam中和road对应的类别为sidewalk(分割成了同一个mask)，将预测结果改为road
        # mask_road = ((segmentation[:, :, 0] == 0) & (fusion_trainid_0 == 1))
        # 预测结果为siwalk但是sam中和siwalk对应的类别为road(分割成了同一个mask)，将预测结果改为siwalk
        mask_siwa = ((segmentation == 1) & (fusion_trainid_0 == 0))
        if confidence_mask is not None:
            mask_siwa = np.logical_and(mask_siwa, confidence_mask)
        # if entropy_mask is not None:
            # mask_siwa = np.logical_and(mask_siwa, entropy_mask)
        mask_buil = ((segmentation == 2) & (fusion_trainid_0 == 10))
        # 预测结果为fence但是sam中和fence对应的类别为building(分割成了同一个mask)，将预测结果改为fence
        mask_fenc = ((segmentation == 4) & (fusion_trainid_0 == 2))
        # 预测结果为pole但是sam中和pole对应的类别为building/light/sign(分割成了同一个mask)，将预测结果改为pole
        mask_pole = ((segmentation == 5) & (fusion_trainid_0 == 2))\
                    | ((segmentation == 5) & (fusion_trainid_0 == 6))\
                    | ((segmentation == 5) & (fusion_trainid_0 == 7))
        # 预测结果为ligh但是sam中和ligh对应的类别为building/pole/vegetation(分割成了同一个mask)，将预测结果改为ligh
        mask_ligh = ((segmentation == 6) & (fusion_trainid_0 == 2)) \
                    | ((segmentation == 6) & (fusion_trainid_0 == 5)) \
                    | ((segmentation == 6) & (fusion_trainid_0 == 8))
        # 预测结果为sign但是sam中和sign对应的类别为building/vegetation(分割成了同一个mask)，将预测结果改为sign
        mask_sign = ((segmentation == 7) & (fusion_trainid_0 == 2))\
                    | ((segmentation == 7) & (fusion_trainid_0 == 8))
        mask_sign_2 = ((segmentation == 7) & (fusion_trainid_0 == 5))  # [H, W]
        # 预测结果为car但是sam中和car对应的类别为vegetation(分割成了同一个mask)，将预测结果改为car
        mask_car = ((segmentation == 13) & (fusion_trainid_0 == 8))  #
        mask_bike = (segmentation == 18)
        if np.max(mask_sign_2):  # 如果mask_sign_2中有值
            # 注意要先试用np.newaxis将mask_sign_2的维度扩展为3维，
            # 再使用np.repeat将mask_sign_2的前两维在第三维复制3份
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)  # [h, w]->[3*h, w]
            # print('mask_sign_2.shape: ', mask_sign_2.shape, np.max(mask_sign_2), np.min(mask_sign_2))
            # mask_sign_2 = mask_sign_2.astype(np.uint8).repeat(3, axis=2)  # [h, w]->[h, w, 3]
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2

        # 预测结果为person但是sam中和person对应的类别为building(分割成了同一个mask)，将预测结果改为person
        mask_person = ((segmentation == 11) & (fusion_trainid_0 == 2))\

        # fusion_trainid_0[mask_road] = 0
        f0_road_mask = (fusion_trainid_0 == 0).astype(np.uint8)
        if f0_road_mask.any() and not inside_rect(cal_center(f0_road_mask), self.road_center_rect):
            fusion_trainid_0[mask_siwa] = 1
        fusion_trainid_0[mask_fenc] = 4
        fusion_trainid_0[mask_pole] = 5
        fusion_trainid_0[mask_ligh] = 6
        fusion_trainid_0[mask_sign] = 7
        fusion_trainid_0[mask_person] = 11
        fusion_trainid_0[mask_car] = 13
        # print(fusion_ids)
        # if 18 not in fusion_ids:
        fusion_trainid_0[mask_bike] = 18
        fusion_color_0 = self.color_segmentation(fusion_trainid_0)
        return fusion_trainid_0, fusion_color_0

    def fusion_mode_4(self, segmentation, sam_pred, fusion_trainid=None, confidence_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function:
            based on fusion_mode_3,
            use confidence_mask to select model segmentation to the fusion result
        input:
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            confidence_mask:[h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])

        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            # print('copy in fusion 4')
            fusion_trainid = copy.deepcopy(fusion_trainid)
        road_mask = (segmentation == 0) & (fusion_trainid == 1) & confidence_mask
        side_mask = (segmentation == 1) & (fusion_trainid == 0) & confidence_mask
        fusion_trainid[road_mask] = 0  # road
        fusion_trainid[side_mask] = 1  # sidewalk
        fusion_color = self.color_segmentation(fusion_trainid)

        return fusion_trainid, fusion_color

    def fusion_mode_5(self, segmentation, sam_pred, fusion_trainid=None, entropy_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function:
            based on fusion_mode_3,
            use entropy_mask to select model segmentation to the fusion result
        input:
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            entropy_mask:   [h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # [road, sidewalk]
        road_mask = (segmentation == 0) & (fusion_trainid == 1) & entropy_mask
        # [sidewalk, road]
        side_mask = (segmentation == 1) & (fusion_trainid == 0) & entropy_mask
        # [vegetation, sidewalk]
        vege_mask = (segmentation == 8) & (fusion_trainid == 1) & entropy_mask

        fusion_trainid[road_mask] = 0
        fusion_trainid[side_mask] = 1
        fusion_trainid[vege_mask] = 8
        fusion_color = self.color_segmentation(fusion_trainid)

        return fusion_trainid, fusion_color

    def fusion_mode_6(self, segmentation, sam_pred):
        # not so good
        if len(segmentation.shape) > 2:
            segmentation = copy.deepcopy(segmentation[:, :, 0])
        fusion_trainid, fusion_color = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]

        for class_id in unique_classes:
            # get the class mask in segmentation
            class_mask = (segmentation == class_id)
            # eroded the class mask in segmentation
            eroded_class_mask, area = shrink_region(class_mask, num_pixels=self.shrink_num)
            # assign the corresponding area in fusion_trainid with the class_id
            fusion_trainid[eroded_class_mask] = class_id
        fusion_trainid = fusion_trainid.astype(np.uint8)

        fusion_color = self.color_segmentation(fusion_trainid)
        return fusion_trainid, fusion_color

    def trainid2color(self, trainid):
        '''
        function: convert trainID to color in cityscapes
        input: trainid
        output: color
        '''
        # if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask

    def visualize_numpy(self, np_array):
        # 创建子图和画布
        fig, axis = plt.subplots(figsize=(8, 4))
        # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(8, 4))

        # 可视化预测置信度数组
        im1 = axis.imshow(np_array, cmap='viridis')
        # ax1.set_title('Confidence')
        axis.set_xlabel('Width')
        axis.set_ylabel('Height')
        axis.axis('off')
        # fig.colorbar(im1, ax=ax1)
        divider1 = make_axes_locatable(axis)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax1)
        # return the figure in numpy format
        # 将图像转换为NumPy数组
        fig.canvas.draw()
        bgr_image = np.array(fig.canvas.renderer._renderer)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        
        plt.clf()
        plt.close(fig)
        
        # 裁剪图像，去除边界
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bgr_image = bgr_image[y:y+h, x:x+w]
        
        return bgr_image

    def vis_np_higher_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array > threshold
        input:
        output: high_mask, image
        '''
        image = image.copy()
        high_mask = np_array > threshold
        image[high_mask] = [0, 255, 0]  # BGR
        return high_mask, image

    def vis_np_lower_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array < threshold
        input:
        output: low_mask, image
        '''
        image = image.copy()
        low_mask = np_array < threshold
        image[low_mask] = [0, 255, 0]  # BGR
        return low_mask, image

    def dis_imgs_horizontal(self, images, image_name, mious, thresholds):
        '''
        function:
            display the images horizontally and save the result
        input:
            images: a list of images, 3 * 4 = 12 images
                    [image, ground truth, sam seg, model seg, error_0
                    fusion_1_result, fusion_2_result, fusion_3_result, fusion_4_result, fusion_5_result,
                    error_1, error_2, error_3, error_4, error_5,
                    confidence_map, entropy_map]
            images_name: the name of the image
            mious: a list of miou and ious,
                    (miou_0, ious_0), (miou_1, ious_1), (miou_2, ious_2), 
                    (miou_3, ious_3), (miou_4, ious_4), (miou_5, ious_5),
            thresholds: a list of thresholds
                    [confidence_threshold, entropy_threshold]
        '''
        # 获取最大高度和总宽度
        # max_height = max(image.shape[0] for image in images)
        # total_width = sum(image.shape[1] for image in images)
        col = 5
        row = len(images) // col + 1 if len(images) % col != 0 else len(images) // col
        gap = 10  # the gap between two images horizontally
        new_height = self.display_size[0] * row
        new_total_width = (self.display_size[1] + gap) * col
        
        # 显示的文本列表
        texts = ['Image', 'Ground Truth', 'SAM', 'Pred, ', 'Error image of pred']
        for i, (miou, ious) in enumerate(mious):
            # cal the non-zero classes in ious
            unique_classes = np.sum(np.array(ious) != 0)
            mIOU2 = np.sum(np.array(ious)) / unique_classes
            if i == 0:
                texts[-2] += 'mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100)
            else:
                texts.append('f_{}, mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(i, self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100))
        for i in range(len(mious)-1):
            texts.append('Error image f_{}'.format(i + 1))
        texts.append('Confidence')
        texts.append('Entropy')
        texts.append('Confidence {}'.format(thresholds[0]))
        texts.append('Entropy {:.2f}'.format(thresholds[1]))

        # 创建一个新的空白画布
        output_image = np.zeros((new_height, new_total_width, 3), dtype=np.uint8)

        # 逐个将图像水平放置在画布上
        current_width = 0
        for i, image in enumerate(images):
            image = cv2.resize(image, (self.display_size[1], self.display_size[0]), \
                    interpolation=cv2.INTER_LINEAR)
            image = cv2.putText(image, texts[i], (20, 50), \
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, color=(0, 0, 255), thickness=2)
            # first row
            if i < col:
                output_image[0*image.shape[0]:1*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # second row
            elif col <= i < 2 * col:
                output_image[1*image.shape[0]:2*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # third row
            elif col * 2 <= i < 3 * col:
                output_image[2*image.shape[0]:3*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # fourth row
            else:
                output_image[3*image.shape[0]:4*image.shape[0], current_width:current_width+image.shape[1], :] = image
            current_width += (image.shape[1] + gap)
            current_width = current_width % new_total_width
        
        # 显示结果图像
        cv2.imwrite(os.path.join(self.output_folder, 'horizontal', image_name + self.mask_suffix), output_image)
        del output_image
        # cv2.imshow('Images', output_image)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()

    def save_ious(self, miou_0, ious_0, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3,
                  miou_4, ious_4, miou_5, ious_5, image_name):
        miou_diff_1_0 = round((miou_1 - miou_0) * 100, 2)
        miou_diff_2_0 = round((miou_2 - miou_0) * 100, 2)
        miou_diff_3_0 = round((miou_3 - miou_0) * 100, 2)
        miou_diff_4_0 = round((miou_4 - miou_0) * 100, 2)
        miou_diff_5_0 = round((miou_5 - miou_0) * 100, 2)
        iou_diff_1_0 = [round((ious_1[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_2_0 = [round((ious_2[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_3_0 = [round((ious_3[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_4_0 = [round((ious_4[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_5_0 = [round((ious_5[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'UDA seg':  [round(miou_0 * 100, 2)] + [round(ious_0[i] * 100, 2) for i in range(len(ious_0))],
            'Fusion 1': [round(miou_1 * 100, 2)] + [round(ious_1[i] * 100, 2) for i in range(len(ious_1))],
            'Fusion 2': [round(miou_2 * 100, 2)] + [round(ious_2[i] * 100, 2) for i in range(len(ious_2))],
            'Fusion 3': [round(miou_3 * 100, 2)] + [round(ious_3[i] * 100, 2) for i in range(len(ious_3))],
            'Fusion 4': [round(miou_4 * 100, 2)] + [round(ious_4[i] * 100, 2) for i in range(len(ious_4))],
            'Fusion 5': [round(miou_5 * 100, 2)] + [round(ious_5[i] * 100, 2) for i in range(len(ious_5))],
            'Differ_1_0': [miou_diff_1_0] + iou_diff_1_0,
            'Differ_2_0': [miou_diff_2_0] + iou_diff_2_0,
            'Differ_3_0': [miou_diff_3_0] + iou_diff_3_0,
            'Differ_4_0': [miou_diff_4_0] + iou_diff_4_0,
            'Differ_5_0': [miou_diff_5_0] + iou_diff_5_0,
        })

        # save the miou and class ious
        data.to_csv(os.path.join(self.output_folder, 'ious', image_name + '.csv'), index=False)

    def get_error_image(self, predicted, ground_truth, pred_color):
        '''
        function: get the error image
        input: predicted, ground_truth
            predicted: [H, W]
            ground_truth: [H, W]
            pred_color: [H, W, 3]
        output: error_image on pred_color
        '''
        if self.num_classes == 16:
            ground_truth[ground_truth==9] = 255
            ground_truth[ground_truth==14] = 255
            ground_truth[ground_truth==16] = 255
        error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
        # predicted_color = self.color_segmentation(predicted)
        # change the area of error mask in pred_color to white
        pred_color_copy = copy.deepcopy(pred_color)
        pred_color_copy[error_mask == 0] = [255, 255, 255]
        
        # error_mask[pred != gt] = 255
        return pred_color_copy

class Fusion_GTA():
    def __init__(self, mask_folder=None, segmentation_folder=None, confidence_folder=None, entropy_folder=None,
                 image_folder=None, gt_folder=None, num_classes=None, road_center_rect=None,
                 mix_ratio=None, resize_ratio=None, output_folder=None, mask_suffix=None,
                 segmentation_suffix=None, segmentation_suffix_noimg=None,
                 confidence_suffix=None, entropy_suffix=None, gt_suffix=None,
                 fusion_mode=None, sam_classes=None, shrink_num=None, display_size=(200, 400)):
        # the path to the sam mask
        self.mask_folder = mask_folder
        # the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        # the path to the confidence map
        self.confidence_folder = confidence_folder
        self.confidence_suffix = confidence_suffix
        # the path to the entropy map
        self.entropy_folder = entropy_folder
        self.entropy_suffix = entropy_suffix
        # the number of classes
        self.num_classes = num_classes
        # the rect of the road center
        self.road_center_rect = road_center_rect
        # the path to the ground truth folder
        self.gt_folder = gt_folder
        # self.gt_color_folder = self.gt_folder.replace('train_all', 'train_gt_color')
        # the path to the original image
        self.image_folder = image_folder
        # the mix ratio of the fusion result and original image
        self.mix_ratio = mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        # the path to the output folder
        self.output_folder = output_folder
        # the image suffix of the mask and segmentation result
        self.mask_suffix = mask_suffix
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        # the gt suffix
        self.gt_suffix = gt_suffix
        # the fusion mode
        self.fusion_mode = fusion_mode
        # the classes sam performs better
        self.sam_classes = sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = shrink_num
        # the size of the image
        self.display_size = display_size
        self.label_names = [trainid2label[train_id].name for train_id in range(19)]
        if self.num_classes == 16:
            self.label_names.remove('train')
            self.label_names.remove('truck')
            self.label_names.remove('terrain')
        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)
        self.image_names.sort()

        # make the folder to save the fusion result
        # the fusion result in trainID
        if self.output_folder is not None:
            self.check_and_make(os.path.join(self.output_folder, 'trainID'))
            # the fusion result in color
            # self.check_and_make(os.path.join(self.output_folder, 'color'))
            # the fusion result in color mixed with original image
            self.check_and_make(os.path.join(self.output_folder, 'mixed'))
            # make the folder to save the fusion result with segmentation result as the background
            # the fusion result in trainID with segmentation result as the background
            self.check_and_make(os.path.join(self.output_folder, 'trainID_bg'))
            # self.check_and_make(os.path.join(self.output_folder, 'color_bg'))
            # the fusion result in color with segmentation result as the background
            # the fusion result in color mixed with original image with segmentation 
            # result as the background
            self.check_and_make(os.path.join(self.output_folder, 'horizontal'))
            self.check_and_make(os.path.join(self.output_folder, 'mixed_bg'))
            self.check_and_make(os.path.join(self.output_folder, 'ious'))

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

    def get_sam_pred(self, image_name, segmentation, confidence_mask=None, entropy_mask=None):
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
        #现在已经按照mask的面积进行了从大到小的排序,序号从0开始
        mask_names = natsorted(mask_names)
        
        # 将mask按照面积从大到小进行排序,由于占用融合时间,目前已经离线完成
        # mask_areas = []
        # for mask_name in mask_names:
        #     mask_path = os.path.join(self.mask_folder, image_name, mask_name)
        #     mask = cv2.imread(mask_path)  # [h,w,3]
        #     mask_area = np.sum(mask[:, :, 0] == 255)
        #     mask_areas.append(mask_area)
        # mask_names = [mask_name for _, mask_name in sorted(zip(mask_areas, mask_names), reverse=True)]
        
        sam_mask = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        for index, mask_name in enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, image_name, mask_name)
            mask = cv2.imread(mask_path)  # [h,w,3]
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]  # [N,]
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n, ], [n1, n2, n3, ...]
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            most_freq_id = num_ids[0]
            
            if len(counts) >= 2:
                # [building, wall]
                if num_ids[0] == 2 and num_ids[1] == 3 and counts[1] / counts[0] >= 0.3:
                    most_freq_id = num_ids[1]
                # [building, fence]
                elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    most_freq_id = num_ids[1]
                # [building, pole]
                elif num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.15:
                    most_freq_id = num_ids[1]
                # [building, traffic sign]
                elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.1:
                    mask_shape = Mask_Shape(mask)
                    # if the mask is rectangular or triangular, then assign the mask with traffic sign
                    if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                        most_freq_id = num_ids[1]
                # [wall, fence]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    most_freq_id = num_ids[1]
                # [vegetation, terrain]
                elif num_ids[0] == 8 and num_ids[1] == 9 and counts[1] / counts[0] >= 0.05:
                    most_freq_id = num_ids[1]
                # [terrain, sidewalk]
                elif num_ids[0] == 9 and num_ids[1] == 1:
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], mask[:, :, 0] == 255), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], mask[:, :, 0] == 255), entropy_mask))
                    if num_id_1 > num_id_0:
                        most_freq_id = num_ids[1]
                # for synthia
                # [vegetation, building], 窗户被判断为vegetation
                elif num_ids[0] == 8 and num_ids[1] == 2:
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    if num_id_0 ==0 or num_id_1 / num_id_0 > 0.25:
                        most_freq_id = num_ids[1]
                # [road, sidewalk]
                # elif num_ids[0] == 0 and num_ids[1] == 1:
                #     if index == 0:
                #         most_freq_id = 0
                #     elif counts[1] / counts[0] >= 0.75:
                #         most_freq_id = num_ids[1]
                # [sidewalk, bicycle]
                elif num_ids[0] == 1 and num_ids[1] == 18 and counts[1] / counts[0] >= 0.15:
                    most_freq_id = num_ids[1]
                # elif (num_ids[0] == 1 and num_ids[1] == 0) or \
                #     (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                #     # [sidewalk, road]
                #     mask_center = cal_center(mask[:, :, 0])
                #     if inside_rect(mask_center, self.road_center_rect) or index == 0:
                #         most_freq_id = 0
                    
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id  # 存在重叠的问题
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
        return sam_mask
        
    def color_segmentation(self, segmentation):
        #get the color segmentation result, initial the color segmentation result with black (0,0,0)
        #input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        for train_id in train_ids:
            if self.num_classes == 16 and train_id in [9, 14, 16]:
                continue
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
        
    def fusion(self):
        bar = tqdm.tqdm(total=len(self.image_names))
        for image_name in self.image_names:
            #get the segmentation result
            prediction_path = os.path.join(self.segmentation_folder, image_name + self.segmentation_suffix)
            if self.segmentation_suffix_noimg:
                prediction_path = prediction_path.replace('_leftImg8bit', '')
            # print('load from: ', prediction_path)
            segmentation = cv2.imread(prediction_path) #[h, w, 3], 3 channels not 1 channel
            # print('prediction_path', prediction_path)
            
            #get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            
            #get the sam segmentation result using the mask
            sam_pred = self.get_sam_pred(image_name, segmentation)
            sam_color = self.color_segmentation(sam_pred)
            
            #get the mixed color image using the self.mix_ratio
            mixed_color = cv2.addWeighted(original_image, self.mix_ratio, sam_color, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color = cv2.resize(mixed_color, (int(mixed_color.shape[1] * self.resize_ratio), int(mixed_color.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            
            #save the sam mask in trainid and color to the output folder
            cv2.imwrite(os.path.join(self.output_folder, 'trainID', image_name + self.mask_suffix), sam_pred)
            # cv2.imwrite(os.path.join(self.output_folder, 'color', image_name + self.mask_suffix), fusion_color)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed', image_name + self.mask_suffix), mixed_color)

            #get the fusion result with the background
            if self.fusion_mode == 1:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 2:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_2(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 3:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
            # elif self.fusion_mode == 4:
            #     fusion_trainid_bg, fusion_color_bg = self.fusion_mode_4(segmentation=segmentation, sam_pred=sam_pred)
            else:
                # raise NotImplementedError
                raise NotImplementedError("This fusion mode has not been implemented yet.")
                             
            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), fusion_trainid_bg)
            # cv2.imwrite(os.path.join(self.output_folder, 'color_bg', image_name + self.mask_suffix), fusion_color_bg)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)
            # fusion_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            
            bar.update(1)
            # mask = cv2.imread(os.path.join(self.mask_folder, image_name + self.mask_suffix))
            
            # image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
            # mask = mask / 255
            # segmentation = segmentation / 255
            # mask = mask * self.mix_ratio
            # segmentation = segmentation * (1 - self.mix_ratio)
            # fusion = mask + segmentation
            # fusion = fusion * 255
            # fusion = fusion.astype('uint8')
            # fusion = cv2.cvtColor(fusion, cv2.COLOR_GRAY2BGR)
            # fusion = cv2.addWeighted(image, 0.5, fusion, 0.5, 0)
            # cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), fusion)

    def fusion_mode_1(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        train_ids = np.unique(sam_pred)
        train_ids = train_ids[train_ids != 255]
        for train_id in train_ids:
            fusion_trainid[sam_pred == train_id] = train_id
        # fusion_color = self.color_segmentation(fusion_trainid)
        
        #use the segmentation result to fill the pixels in fusion_trainid whose trainid is 255
        fusion_trainid_bg = fusion_trainid.copy()
        indexs = np.where(fusion_trainid == 255)
        fusion_trainid_bg[indexs] = segmentation[:, :, 0][indexs]      
        #use the corresponding color of segmentation result to fill the pixels in fusion_color whose trainid is 255
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg)
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)
        fusion_color_bg = fusion_color_bg.astype(np.uint8)
        
        return fusion_trainid_bg, fusion_color_bg

    def fusion_mode_2(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        fusion_trainid = segmentation[:, :, 0].copy() #fill the fusion result all with segmentation result
        
        #use sam_pred with self.sam_classes to cover the fusion_trainid
        sam_pred_ids = np.unique(sam_pred)
        sam_pred_ids = sam_pred_ids[sam_pred_ids != 255]
        for sam_class in self.sam_classes:
            if sam_class in sam_pred_ids:
                fusion_trainid[sam_pred == sam_class] = sam_class
        
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_3(self, segmentation, sam_pred, fusion_trainid=None, 
                      confidence_mask=None, entropy_mask=None):
        '''
        segmentation: [h, w, 3]
        sam_pred: [h, w]
        '''
        if fusion_trainid is None:
            fusion_trainid_0, _ = self.fusion_mode_1(segmentation, sam_pred)
        else:
            # print('copy in fusion 3')
            fusion_trainid_0 = copy.deepcopy(fusion_trainid)
        fusion_ids = np.unique(fusion_trainid_0)
        # fusion_trainid_0: [h, w], fusion_color_0: [h, w, 3]
        # # 预测结果为road但是sam中和road对应的类别为sidewalk(分割成了同一个mask)，将预测结果改为road
        # [road, sidewalk]
        mask_road = ((segmentation[:, :, 0] == 0) & (fusion_trainid_0 == 1))
        if confidence_mask is not None:
            mask_road = np.logical_and(mask_road, confidence_mask)
        # self.save_binary_mask('road before', mask_road)
        # self.save_binary_mask('confidence_mask', confidence_mask)
        # 预测结果为siwalk但是sam中和siwalk对应的类别为road(分割成了同一个mask)，将预测结果改为siwalk
        # [sidewalk, road]
        mask_siwa = ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 0)) \
                    | ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 9))
        if confidence_mask is not None:
            mask_siwa = np.logical_and(mask_siwa, confidence_mask)
        # if entropy_mask is not None:
            # mask_siwa = np.logical_and(mask_siwa, entropy_mask)
        # [building, sky], [building, fence]
        mask_buil = ((segmentation[:, :, 0] == 2) & (fusion_trainid_0 == 10))\
                    | ((segmentation[:, :, 0] == 2) & (fusion_trainid_0 == 4))
        if confidence_mask is not None:
            mask_buil = np.logical_and(mask_buil, confidence_mask)
        # fence, [building, vegetation]
        mask_wall = (segmentation[:, :, 0] == 3) & (fusion_trainid_0 == 8)
        if confidence_mask is not None:
            mask_wall = np.logical_and(mask_wall, confidence_mask)
        mask_fenc = ((segmentation[:, :, 0] == 4) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 4) & (fusion_trainid_0 == 8))
        # 预测结果为pole但是sam中和pole对应的类别为building/light/sign(分割成了同一个mask)，将预测结果改为pole
        mask_pole = ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 6))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 7))
        # 预测结果为ligh但是sam中和ligh对应的类别为building/pole/vegetation(分割成了同一个mask)，将预测结果改为ligh
        mask_ligh = ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 2)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 5)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 8))
        # traffic sign, [building, vegetation, traffic light]
        mask_sign = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 8))\
                    | ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 6))
        mask_sign_2 = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 5))  # [H, W]
        # [vegetation, terrain]
        mask_vege = ((segmentation[:, :, 0] == 8) & (fusion_trainid_0 == 9))\
                    | ((segmentation[:, :, 0] == 8) & (fusion_trainid_0 == 2))
        if confidence_mask is not None:
            mask_vege = np.logical_and(mask_vege, confidence_mask)
        # 预测结果为car但是sam中和car对应的类别为vegetation(分割成了同一个mask)，将预测结果改为car
        mask_car = ((segmentation[:, :, 0] == 13) & (fusion_trainid_0 == 8))  # 
        mask_bike = (segmentation[:, :, 0] == 18)
        if np.max(mask_sign_2):  # 如果mask_sign_2中有值
            # 注意要先试用np.newaxis将mask_sign_2的维度扩展为3维，
            # 再使用np.repeat将mask_sign_2的前两维在第三维复制3份
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)  # [h, w]->[3*h, w]
            # print('mask_sign_2.shape: ', mask_sign_2.shape, np.max(mask_sign_2), np.min(mask_sign_2))
            # mask_sign_2 = mask_sign_2.astype(np.uint8).repeat(3, axis=2)  # [h, w]->[h, w, 3]
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2

        # 预测结果为person但是sam中和person对应的类别为building(分割成了同一个mask)，将预测结果改为person
        mask_person = ((segmentation[:, :, 0] == 11) & (fusion_trainid_0 == 2))\
            
        fusion_trainid_0[mask_road] = 0
        # f0_road_mask = (fusion_trainid_0 == 0).astype(np.uint8)
        # if f0_road_mask.any() and not inside_rect(cal_center(f0_road_mask), self.road_center_rect):
        fusion_trainid_0[mask_siwa] = 1
        fusion_trainid_0[mask_buil] = 2
        fusion_trainid_0[mask_wall] = 3
        fusion_trainid_0[mask_fenc] = 4
        fusion_trainid_0[mask_pole] = 5
        fusion_trainid_0[mask_ligh] = 6
        fusion_trainid_0[mask_sign] = 7
        fusion_trainid_0[mask_vege] = 8
        fusion_trainid_0[mask_person] = 11
        fusion_trainid_0[mask_car] = 13
        # print(fusion_ids)
        # if 18 not in fusion_ids:
        fusion_trainid_0[mask_bike] = 18
        fusion_color_0 = self.color_segmentation(fusion_trainid_0)
        return fusion_trainid_0, fusion_color_0

    def fusion_mode_4(self, segmentation, sam_pred, fusion_trainid=None, confidence_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use confidence_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            confidence_mask:[h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            # print('copy in fusion 4')
            fusion_trainid = copy.deepcopy(fusion_trainid)
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & confidence_mask
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & confidence_mask
        buil_mask = (segmentation[:, :, 0] == 2) & (fusion_trainid == 7) & confidence_mask
        fusion_trainid[road_mask] = 0  # road
        fusion_trainid[side_mask] = 1  # sidewalk
        # newly added
        fusion_trainid[buil_mask] = 2  # building
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
    
    def fusion_mode_5(self, segmentation, sam_pred, fusion_trainid=None, entropy_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use entropy_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            entropy_mask:   [h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # [road, sidewalk]
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & entropy_mask
        # [sidewalk, road]
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & entropy_mask
        # [building, traffic sign]
        buil_mask = (segmentation[:, :, 0] == 2) & (fusion_trainid == 7) & entropy_mask
        # [vegetation, sidewalk]
        vege_mask = (segmentation[:, :, 0] == 8) & (fusion_trainid == 1) & entropy_mask
        
        fusion_trainid[road_mask] = 0
        fusion_trainid[side_mask] = 1
        # newly added
        fusion_trainid[buil_mask] = 2  # building
        fusion_trainid[vege_mask] = 8  # 
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_6(self, segmentation, sam_pred):
        #not so good
        fusion_trainid, fusion_color = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]
        
        for class_id in unique_classes:
            #get the class mask in segmentation
            class_mask = (segmentation == class_id)
            #eroded the class mask in segmentation
            eroded_class_mask, area = shrink_region(class_mask, num_pixels=self.shrink_num)
            #assign the corresponding area in fusion_trainid with the class_id
            fusion_trainid[eroded_class_mask] = class_id
        fusion_trainid = fusion_trainid.astype(np.uint8)            
        
        fusion_color = self.color_segmentation(fusion_trainid)
        return fusion_trainid, fusion_color

    def trainid2color(self, trainid):
        '''
        function: convert trainID to color in cityscapes
        input: trainid
        output: color
        '''
        #if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask
            
        
        
        # if type(trainid) == tuple: #a mask
        #     ###assign the color to the mask according to the trainid
        #     color_mask = np.zeros((trainid.shape[0], trainid.shape[1], 3), dtype=np.uint8)
        #     for i in range(trainid.shape[0]):
        #         for j in range(trainid.shape[1]):
        #             label_object = trainid2label[trainid[i, j]]
        #             color_mask[i, j] = label_object.color
        #     return color_mask
        # else: #one number
        #     label_object = trainid2label[trainid]
        #     return label_object.color   

    def visualize_numpy(self, np_array):
        # 创建子图和画布
        fig, axis = plt.subplots(figsize=(8, 4))
        # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(8, 4))

        # 可视化预测置信度数组
        im1 = axis.imshow(np_array, cmap='viridis')
        # ax1.set_title('Confidence')
        axis.set_xlabel('Width')
        axis.set_ylabel('Height')
        axis.axis('off')
        # fig.colorbar(im1, ax=ax1)
        divider1 = make_axes_locatable(axis)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax1)
        # return the figure in numpy format
        # 将图像转换为NumPy数组
        fig.canvas.draw()
        bgr_image = np.array(fig.canvas.renderer._renderer)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        
        plt.clf()
        plt.close(fig)
        
        # 裁剪图像，去除边界
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bgr_image = bgr_image[y:y+h, x:x+w]
        
        return bgr_image

    def vis_np_higher_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array > threshold
        input:
        output: high_mask, image
        '''
        image = image.copy()
        high_mask = np_array > threshold
        image[high_mask] = [0, 255, 0]  # BGR
        return high_mask, image

    def vis_np_lower_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array < threshold
        input:
        output: low_mask, image
        '''
        image = image.copy()
        low_mask = np_array < threshold
        image[low_mask] = [0, 255, 0]  # BGR
        return low_mask, image

    def dis_imgs_horizontal(self, images, image_name, mious, thresholds):
        '''
        function:
            display the images horizontally and save the result
        input:
            images: a list of images, 3 * 4 = 12 images
                    [image, ground truth, sam seg, model seg, error_0
                    fusion_1_result, fusion_2_result, fusion_3_result, fusion_4_result, fusion_5_result,
                    error_1, error_2, error_3, error_4, error_5,
                    confidence_map, entropy_map]
            images_name: the name of the image
            mious: a list of miou and ious,
                    (miou_0, ious_0), (miou_1, ious_1), (miou_2, ious_2), 
                    (miou_3, ious_3), (miou_4, ious_4), (miou_5, ious_5),
            thresholds: a list of thresholds
                    [confidence_threshold, entropy_threshold]
        '''
        # 获取最大高度和总宽度
        # max_height = max(image.shape[0] for image in images)
        # total_width = sum(image.shape[1] for image in images)
        col = 5
        row = len(images) // col + 1 if len(images) % col != 0 else len(images) // col
        gap = 10  # the gap between two images horizontally
        new_height = self.display_size[0] * row
        new_total_width = (self.display_size[1] + gap) * col
        
        # 显示的文本列表
        texts = ['Image', 'Ground Truth', 'SAM', 'Pred, ', 'Error image of pred']
        for i, (miou, ious) in enumerate(mious):
            # cal the non-zero classes in ious
            unique_classes = np.sum(np.array(ious) != 0)
            mIOU2 = np.sum(np.array(ious)) / unique_classes
            if i == 0:
                texts[-2] += 'mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100)
            else:
                texts.append('f_{}, mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(i, self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100))
        for i in range(len(mious)-1):
            texts.append('Error image f_{}'.format(i + 1))
        texts.append('Confidence')
        texts.append('Entropy')
        texts.append('Confidence {}'.format(thresholds[0]))
        texts.append('Entropy {:.2f}'.format(thresholds[1]))

        # 创建一个新的空白画布
        output_image = np.zeros((new_height, new_total_width, 3), dtype=np.uint8)

        # 逐个将图像水平放置在画布上
        current_width = 0
        for i, image in enumerate(images):
            image = cv2.resize(image, (self.display_size[1], self.display_size[0]), \
                    interpolation=cv2.INTER_LINEAR)
            image = cv2.putText(image, texts[i], (20, 50), \
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, color=(0, 0, 255), thickness=2)
            # first row
            if i < col:
                output_image[0*image.shape[0]:1*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # second row
            elif col <= i < 2 * col:
                output_image[1*image.shape[0]:2*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # third row
            elif col * 2 <= i < 3 * col:
                output_image[2*image.shape[0]:3*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # fourth row
            else:
                output_image[3*image.shape[0]:4*image.shape[0], current_width:current_width+image.shape[1], :] = image
            current_width += (image.shape[1] + gap)
            current_width = current_width % new_total_width
        
        # 显示结果图像
        cv2.imwrite(os.path.join(self.output_folder, 'horizontal', image_name + self.mask_suffix), output_image)
        # cv2.imshow('Images', output_image)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
    
    def save_ious(self, miou_0, ious_0, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3, 
                  miou_4, ious_4, miou_5, ious_5, image_name):
        miou_diff_1_0 = round((miou_1 - miou_0) * 100, 2)
        miou_diff_2_0 = round((miou_2 - miou_0) * 100, 2)
        miou_diff_3_0 = round((miou_3 - miou_0) * 100, 2)
        miou_diff_4_0 = round((miou_4 - miou_0) * 100, 2)
        miou_diff_5_0 = round((miou_5 - miou_0) * 100, 2)
        iou_diff_1_0 = [round((ious_1[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_2_0 = [round((ious_2[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_3_0 = [round((ious_3[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_4_0 = [round((ious_4[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_5_0 = [round((ious_5[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'UDA seg':  [round(miou_0 * 100, 2)] + [round(ious_0[i] * 100, 2) for i in range(len(ious_0))],
            'Fusion 1': [round(miou_1 * 100, 2)] + [round(ious_1[i] * 100, 2) for i in range(len(ious_1))],
            'Fusion 2': [round(miou_2 * 100, 2)] + [round(ious_2[i] * 100, 2) for i in range(len(ious_2))],
            'Fusion 3': [round(miou_3 * 100, 2)] + [round(ious_3[i] * 100, 2) for i in range(len(ious_3))],
            'Fusion 4': [round(miou_4 * 100, 2)] + [round(ious_4[i] * 100, 2) for i in range(len(ious_4))],
            'Fusion 5': [round(miou_5 * 100, 2)] + [round(ious_5[i] * 100, 2) for i in range(len(ious_5))],
            'Differ_1_0': [miou_diff_1_0] + iou_diff_1_0,
            'Differ_2_0': [miou_diff_2_0] + iou_diff_2_0,
            'Differ_3_0': [miou_diff_3_0] + iou_diff_3_0,
            'Differ_4_0': [miou_diff_4_0] + iou_diff_4_0,
            'Differ_5_0': [miou_diff_5_0] + iou_diff_5_0,
        })

        # save the miou and class ious
        data.to_csv(os.path.join(self.output_folder, 'ious', image_name + '.csv'), index=False)

    def get_error_image(self, predicted, ground_truth, pred_color):
        '''
        function: get the error image
        input: predicted, ground_truth
            predicted: [H, W]
            ground_truth: [H, W]
            pred_color: [H, W, 3]
        output: error_image on pred_color
        '''
        if self.num_classes == 16:
            ground_truth[ground_truth==9] = 255
            ground_truth[ground_truth==14] = 255
            ground_truth[ground_truth==16] = 255
        error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
        # predicted_color = self.color_segmentation(predicted)
        # change the area of error mask in pred_color to white
        pred_color_copy = copy.deepcopy(pred_color)
        pred_color_copy[error_mask == 0] = [255, 255, 255]
        
        # error_mask[pred != gt] = 255
        return pred_color_copy

    def save_binary_mask(self, image_name, mask):
        '''
        function: save the binary mask
        input: image_name, mask, mask_name
        output: save the mask in the output folder
        '''
        mask = mask.astype(np.uint8)
        mask = mask * 255
        cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), mask)
        
class Fusion_SJTU():
    def __init__(self, mask_folder=None, segmentation_folder=None, confidence_folder=None, entropy_folder=None,
                 image_folder=None, gt_folder=None, num_classes=None, road_center_rect=None,
                 mix_ratio=None, resize_ratio=None, output_folder=None, mask_suffix=None,
                 segmentation_suffix=None, segmentation_suffix_noimg=None,
                 confidence_suffix=None, entropy_suffix=None, gt_suffix=None,
                 fusion_mode=None, sam_classes=None, shrink_num=None, display_size=(200, 400)):
        # the path to the sam mask
        self.mask_folder = mask_folder
        # the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        # the path to the confidence map
        self.confidence_folder = confidence_folder
        self.confidence_suffix = confidence_suffix
        # the path to the entropy map
        self.entropy_folder = entropy_folder
        self.entropy_suffix = entropy_suffix
        # the number of classes
        self.num_classes = num_classes
        # the rect of the road center
        self.road_center_rect = road_center_rect
        # the path to the ground truth folder
        self.gt_folder = gt_folder
        # self.gt_color_folder = self.gt_folder.replace('train_all', 'train_gt_color')
        # the path to the original image
        self.image_folder = image_folder
        # the mix ratio of the fusion result and original image
        self.mix_ratio = mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        # the path to the output folder
        self.output_folder = output_folder
        # the image suffix of the mask and segmentation result
        self.mask_suffix = mask_suffix
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        # the gt suffix
        self.gt_suffix = gt_suffix
        # the fusion mode
        self.fusion_mode = fusion_mode
        # the classes sam performs better
        self.sam_classes = sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = shrink_num
        # the size of the image
        self.display_size = display_size
        self.label_names = [trainid2label[train_id].name for train_id in range(19)]
        if self.num_classes == 16:
            self.label_names.remove('train')
            self.label_names.remove('truck')
            self.label_names.remove('terrain')
        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)
        self.image_names.sort()

        # make the folder to save the fusion result
        # the fusion result in trainID
        if self.output_folder is not None:
            self.check_and_make(os.path.join(self.output_folder, 'trainID'))
            # the fusion result in color
            # self.check_and_make(os.path.join(self.output_folder, 'color'))
            # the fusion result in color mixed with original image
            self.check_and_make(os.path.join(self.output_folder, 'mixed'))
            # make the folder to save the fusion result with segmentation result as the background
            # the fusion result in trainID with segmentation result as the background
            self.check_and_make(os.path.join(self.output_folder, 'trainID_bg'))
            # self.check_and_make(os.path.join(self.output_folder, 'color_bg'))
            # the fusion result in color with segmentation result as the background
            # the fusion result in color mixed with original image with segmentation 
            # result as the background
            self.check_and_make(os.path.join(self.output_folder, 'horizontal'))
            self.check_and_make(os.path.join(self.output_folder, 'mixed_bg'))
            self.check_and_make(os.path.join(self.output_folder, 'ious'))

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

    def get_sam_pred(self, image_name, segmentation, confidence_mask=None, entropy_mask=None):
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
        #现在已经按照mask的面积进行了从大到小的排序,序号从0开始
        mask_names = natsorted(mask_names)
        
        # 将mask按照面积从大到小进行排序,由于占用融合时间,目前已经离线完成
        # mask_areas = []
        # for mask_name in mask_names:
        #     mask_path = os.path.join(self.mask_folder, image_name, mask_name)
        #     mask = cv2.imread(mask_path)  # [h,w,3]
        #     mask_area = np.sum(mask[:, :, 0] == 255)
        #     mask_areas.append(mask_area)
        # mask_names = [mask_name for _, mask_name in sorted(zip(mask_areas, mask_names), reverse=True)]
        
        sam_mask = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        for index, mask_name in enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, image_name, mask_name)
            mask = cv2.imread(mask_path)  # [h,w,3]
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]  # [N,]
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n, ], [n1, n2, n3, ...]
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            most_freq_id = num_ids[0]
            
            if len(counts) >= 2:
                # [building, wall]
                if num_ids[0] == 2 and num_ids[1] == 3 and counts[1] / counts[0] >= 0.3:
                    most_freq_id = num_ids[1]
                # [building, fence]
                elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    most_freq_id = num_ids[1]
                # [building, pole]
                elif num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.15:
                    most_freq_id = num_ids[1]
                # [building, traffic sign]
                elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.1:
                    mask_shape = Mask_Shape(mask)
                    # if the mask is rectangular or triangular, then assign the mask with traffic sign
                    if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                        most_freq_id = num_ids[1]
                # [wall, fence]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    most_freq_id = num_ids[1]
                # [vegetation, terrain]
                elif num_ids[0] == 8 and num_ids[1] == 9 and counts[1] / counts[0] >= 0.05:
                    most_freq_id = num_ids[1]
                # [terrain, sidewalk]
                elif num_ids[0] == 9 and num_ids[1] == 1:
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], mask[:, :, 0] == 255), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], mask[:, :, 0] == 255), entropy_mask))
                    if num_id_1 > num_id_0:
                        most_freq_id = num_ids[1]
                # for synthia
                # [vegetation, building], 窗户被判断为vegetation
                elif num_ids[0] == 8 and num_ids[1] == 2:
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], 
                                                    mask[:, :, 0] == 255), confidence_mask))
                    if num_id_0 ==0 or num_id_1 / num_id_0 > 0.25:
                        most_freq_id = num_ids[1]
                # [road, sidewalk]
                # elif num_ids[0] == 0 and num_ids[1] == 1:
                #     if index == 0:
                #         most_freq_id = 0
                #     elif counts[1] / counts[0] >= 0.75:
                #         most_freq_id = num_ids[1]
                # [sidewalk, bicycle]
                elif num_ids[0] == 1 and num_ids[1] == 18 and counts[1] / counts[0] >= 0.15:
                    most_freq_id = num_ids[1]
                # elif (num_ids[0] == 1 and num_ids[1] == 0) or \
                #     (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                #     # [sidewalk, road]
                #     mask_center = cal_center(mask[:, :, 0])
                #     if inside_rect(mask_center, self.road_center_rect) or index == 0:
                #         most_freq_id = 0
                    
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id  # 存在重叠的问题
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
        return sam_mask
        
    def color_segmentation(self, segmentation):
        #get the color segmentation result, initial the color segmentation result with black (0,0,0)
        #input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        for train_id in train_ids:
            if self.num_classes == 16 and train_id in [9, 14, 16]:
                continue
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
        
    def fusion(self):
        bar = tqdm.tqdm(total=len(self.image_names))
        for image_name in self.image_names:
            #get the segmentation result
            prediction_path = os.path.join(self.segmentation_folder, image_name + self.segmentation_suffix)
            if self.segmentation_suffix_noimg:
                prediction_path = prediction_path.replace('_leftImg8bit', '')
            # print('load from: ', prediction_path)
            segmentation = cv2.imread(prediction_path) #[h, w, 3], 3 channels not 1 channel
            # print('prediction_path', prediction_path)
            
            #get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            
            #get the sam segmentation result using the mask
            sam_pred = self.get_sam_pred(image_name, segmentation)
            sam_color = self.color_segmentation(sam_pred)
            
            #get the mixed color image using the self.mix_ratio
            mixed_color = cv2.addWeighted(original_image, self.mix_ratio, sam_color, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color = cv2.resize(mixed_color, (int(mixed_color.shape[1] * self.resize_ratio), int(mixed_color.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            
            #save the sam mask in trainid and color to the output folder
            cv2.imwrite(os.path.join(self.output_folder, 'trainID', image_name + self.mask_suffix), sam_pred)
            # cv2.imwrite(os.path.join(self.output_folder, 'color', image_name + self.mask_suffix), fusion_color)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed', image_name + self.mask_suffix), mixed_color)

            #get the fusion result with the background
            if self.fusion_mode == 1:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 2:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_2(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 3:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
            # elif self.fusion_mode == 4:
            #     fusion_trainid_bg, fusion_color_bg = self.fusion_mode_4(segmentation=segmentation, sam_pred=sam_pred)
            else:
                # raise NotImplementedError
                raise NotImplementedError("This fusion mode has not been implemented yet.")
                             
            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), fusion_trainid_bg)
            # cv2.imwrite(os.path.join(self.output_folder, 'color_bg', image_name + self.mask_suffix), fusion_color_bg)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)
            # fusion_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            
            bar.update(1)
            # mask = cv2.imread(os.path.join(self.mask_folder, image_name + self.mask_suffix))
            
            # image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
            # mask = mask / 255
            # segmentation = segmentation / 255
            # mask = mask * self.mix_ratio
            # segmentation = segmentation * (1 - self.mix_ratio)
            # fusion = mask + segmentation
            # fusion = fusion * 255
            # fusion = fusion.astype('uint8')
            # fusion = cv2.cvtColor(fusion, cv2.COLOR_GRAY2BGR)
            # fusion = cv2.addWeighted(image, 0.5, fusion, 0.5, 0)
            # cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), fusion)

    def fusion_mode_1(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        train_ids = np.unique(sam_pred)
        train_ids = train_ids[train_ids != 255]
        for train_id in train_ids:
            fusion_trainid[sam_pred == train_id] = train_id
        # fusion_color = self.color_segmentation(fusion_trainid)
        
        #use the segmentation result to fill the pixels in fusion_trainid whose trainid is 255
        fusion_trainid_bg = fusion_trainid.copy()
        indexs = np.where(fusion_trainid == 255)
        fusion_trainid_bg[indexs] = segmentation[:, :, 0][indexs]      
        #use the corresponding color of segmentation result to fill the pixels in fusion_color whose trainid is 255
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg)
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)
        fusion_color_bg = fusion_color_bg.astype(np.uint8)
        
        return fusion_trainid_bg, fusion_color_bg

    def fusion_mode_2(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        fusion_trainid = segmentation[:, :, 0].copy() #fill the fusion result all with segmentation result
        
        #use sam_pred with self.sam_classes to cover the fusion_trainid
        sam_pred_ids = np.unique(sam_pred)
        sam_pred_ids = sam_pred_ids[sam_pred_ids != 255]
        for sam_class in self.sam_classes:
            if sam_class in sam_pred_ids:
                fusion_trainid[sam_pred == sam_class] = sam_class
        
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_3(self, segmentation, sam_pred, fusion_trainid=None, 
                      confidence_mask=None, entropy_mask=None):
        '''
        segmentation: [h, w, 3]
        sam_pred: [h, w]
        '''
        if fusion_trainid is None:
            fusion_trainid_0, _ = self.fusion_mode_1(segmentation, sam_pred)
        else:
            # print('copy in fusion 3')
            fusion_trainid_0 = copy.deepcopy(fusion_trainid)
        fusion_ids = np.unique(fusion_trainid_0)
        # fusion_trainid_0: [h, w], fusion_color_0: [h, w, 3]
        # # 预测结果为road但是sam中和road对应的类别为sidewalk(分割成了同一个mask)，将预测结果改为road
        # [road, sidewalk]
        mask_road = ((segmentation[:, :, 0] == 0) & (fusion_trainid_0 == 1))
        if confidence_mask is not None:
            mask_road = np.logical_and(mask_road, confidence_mask)
        # self.save_binary_mask('road before', mask_road)
        # self.save_binary_mask('confidence_mask', confidence_mask)
        # 预测结果为siwalk但是sam中和siwalk对应的类别为road(分割成了同一个mask)，将预测结果改为siwalk
        # [sidewalk, road]
        mask_siwa = ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 0)) \
                    | ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 9))
        if confidence_mask is not None:
            mask_siwa = np.logical_and(mask_siwa, confidence_mask)
        # if entropy_mask is not None:
            # mask_siwa = np.logical_and(mask_siwa, entropy_mask)
        # [building, sky], [building, fence]
        mask_buil = ((segmentation[:, :, 0] == 2) & (fusion_trainid_0 == 10))\
                    | ((segmentation[:, :, 0] == 2) & (fusion_trainid_0 == 4))
        if confidence_mask is not None:
            mask_buil = np.logical_and(mask_buil, confidence_mask)
        # fence, [building, vegetation]
        mask_wall = (segmentation[:, :, 0] == 3) & (fusion_trainid_0 == 8)
        if confidence_mask is not None:
            mask_wall = np.logical_and(mask_wall, confidence_mask)
        mask_fenc = ((segmentation[:, :, 0] == 4) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 4) & (fusion_trainid_0 == 8))
        # 预测结果为pole但是sam中和pole对应的类别为building/light/sign(分割成了同一个mask)，将预测结果改为pole
        mask_pole = ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 6))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 7))
        # 预测结果为ligh但是sam中和ligh对应的类别为building/pole/vegetation(分割成了同一个mask)，将预测结果改为ligh
        mask_ligh = ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 2)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 5)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 8))
        # traffic sign, [building, vegetation, traffic light]
        mask_sign = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 8))\
                    | ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 6))
        mask_sign_2 = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 5))  # [H, W]
        # [vegetation, terrain]
        mask_vege = ((segmentation[:, :, 0] == 8) & (fusion_trainid_0 == 9))\
                    | ((segmentation[:, :, 0] == 8) & (fusion_trainid_0 == 2))
        if confidence_mask is not None:
            mask_vege = np.logical_and(mask_vege, confidence_mask)
        # 预测结果为car但是sam中和car对应的类别为vegetation(分割成了同一个mask)，将预测结果改为car
        mask_car = ((segmentation[:, :, 0] == 13) & (fusion_trainid_0 == 8))  # 
        mask_bike = (segmentation[:, :, 0] == 18)
        if np.max(mask_sign_2):  # 如果mask_sign_2中有值
            # 注意要先试用np.newaxis将mask_sign_2的维度扩展为3维，
            # 再使用np.repeat将mask_sign_2的前两维在第三维复制3份
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)  # [h, w]->[3*h, w]
            # print('mask_sign_2.shape: ', mask_sign_2.shape, np.max(mask_sign_2), np.min(mask_sign_2))
            # mask_sign_2 = mask_sign_2.astype(np.uint8).repeat(3, axis=2)  # [h, w]->[h, w, 3]
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2

        # 预测结果为person但是sam中和person对应的类别为building(分割成了同一个mask)，将预测结果改为person
        mask_person = ((segmentation[:, :, 0] == 11) & (fusion_trainid_0 == 2))\
            
        fusion_trainid_0[mask_road] = 0
        # f0_road_mask = (fusion_trainid_0 == 0).astype(np.uint8)
        # if f0_road_mask.any() and not inside_rect(cal_center(f0_road_mask), self.road_center_rect):
        fusion_trainid_0[mask_siwa] = 1
        fusion_trainid_0[mask_buil] = 2
        fusion_trainid_0[mask_wall] = 3
        fusion_trainid_0[mask_fenc] = 4
        fusion_trainid_0[mask_pole] = 5
        fusion_trainid_0[mask_ligh] = 6
        fusion_trainid_0[mask_sign] = 7
        fusion_trainid_0[mask_vege] = 8
        fusion_trainid_0[mask_person] = 11
        fusion_trainid_0[mask_car] = 13
        # print(fusion_ids)
        # if 18 not in fusion_ids:
        fusion_trainid_0[mask_bike] = 18
        fusion_color_0 = self.color_segmentation(fusion_trainid_0)
        return fusion_trainid_0, fusion_color_0

    def fusion_mode_4(self, segmentation, sam_pred, fusion_trainid=None, confidence_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use confidence_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            confidence_mask:[h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            # print('copy in fusion 4')
            fusion_trainid = copy.deepcopy(fusion_trainid)
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & confidence_mask
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & confidence_mask
        buil_mask = (segmentation[:, :, 0] == 2) & (fusion_trainid == 7) & confidence_mask
        fusion_trainid[road_mask] = 0  # road
        fusion_trainid[side_mask] = 1  # sidewalk
        # newly added
        fusion_trainid[buil_mask] = 2  # building
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
    
    def fusion_mode_5(self, segmentation, sam_pred, fusion_trainid=None, entropy_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use entropy_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            entropy_mask:   [h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # [road, sidewalk]
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & entropy_mask
        # [sidewalk, road]
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & entropy_mask
        # [building, traffic sign]
        buil_mask = (segmentation[:, :, 0] == 2) & (fusion_trainid == 7) & entropy_mask
        # [vegetation, sidewalk]
        vege_mask = (segmentation[:, :, 0] == 8) & (fusion_trainid == 1) & entropy_mask
        
        fusion_trainid[road_mask] = 0
        fusion_trainid[side_mask] = 1
        # newly added
        fusion_trainid[buil_mask] = 2  # building
        fusion_trainid[vege_mask] = 8  # 
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_6(self, segmentation, sam_pred):
        #not so good
        fusion_trainid, fusion_color = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]
        
        for class_id in unique_classes:
            #get the class mask in segmentation
            class_mask = (segmentation == class_id)
            #eroded the class mask in segmentation
            eroded_class_mask, area = shrink_region(class_mask, num_pixels=self.shrink_num)
            #assign the corresponding area in fusion_trainid with the class_id
            fusion_trainid[eroded_class_mask] = class_id
        fusion_trainid = fusion_trainid.astype(np.uint8)            
        
        fusion_color = self.color_segmentation(fusion_trainid)
        return fusion_trainid, fusion_color

    def trainid2color(self, trainid):
        '''
        function: convert trainID to color in cityscapes
        input: trainid
        output: color
        '''
        #if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask
            
        
        
        # if type(trainid) == tuple: #a mask
        #     ###assign the color to the mask according to the trainid
        #     color_mask = np.zeros((trainid.shape[0], trainid.shape[1], 3), dtype=np.uint8)
        #     for i in range(trainid.shape[0]):
        #         for j in range(trainid.shape[1]):
        #             label_object = trainid2label[trainid[i, j]]
        #             color_mask[i, j] = label_object.color
        #     return color_mask
        # else: #one number
        #     label_object = trainid2label[trainid]
        #     return label_object.color   

    def visualize_numpy(self, np_array):
        # 创建子图和画布
        fig, axis = plt.subplots(figsize=(8, 4))
        # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(8, 4))

        # 可视化预测置信度数组
        im1 = axis.imshow(np_array, cmap='viridis')
        # ax1.set_title('Confidence')
        axis.set_xlabel('Width')
        axis.set_ylabel('Height')
        axis.axis('off')
        # fig.colorbar(im1, ax=ax1)
        divider1 = make_axes_locatable(axis)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax1)
        # return the figure in numpy format
        # 将图像转换为NumPy数组
        fig.canvas.draw()
        bgr_image = np.array(fig.canvas.renderer._renderer)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        
        plt.clf()
        plt.close(fig)
        
        # 裁剪图像，去除边界
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bgr_image = bgr_image[y:y+h, x:x+w]
        
        return bgr_image

    def vis_np_higher_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array > threshold
        input:
        output: high_mask, image
        '''
        image = image.copy()
        high_mask = np_array > threshold
        image[high_mask] = [0, 255, 0]  # BGR
        return high_mask, image

    def vis_np_lower_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array < threshold
        input:
        output: low_mask, image
        '''
        image = image.copy()
        low_mask = np_array < threshold
        image[low_mask] = [0, 255, 0]  # BGR
        return low_mask, image

    def dis_imgs_horizontal(self, images, image_name, mious=None, thresholds=[0.99, 80]):
        '''
        function:
            display the images horizontally and save the result
        input:
            images: a list of images, 3 * 4 = 12 images
                    [image, ground truth, sam seg, model seg, error_0
                    fusion_1_result, fusion_2_result, fusion_3_result, fusion_4_result, fusion_5_result,
                    error_1, error_2, error_3, error_4, error_5,
                    confidence_map, entropy_map]
            images_name: the name of the image
            mious: a list of miou and ious,
                    (miou_0, ious_0), (miou_1, ious_1), (miou_2, ious_2), 
                    (miou_3, ious_3), (miou_4, ious_4), (miou_5, ious_5),
            thresholds: a list of thresholds
                    [confidence_threshold, entropy_threshold]
        '''
        # 获取最大高度和总宽度
        # max_height = max(image.shape[0] for image in images)
        # total_width = sum(image.shape[1] for image in images)
        col = 5
        row = len(images) // col + 1 if len(images) % col != 0 else len(images) // col
        gap = 10  # the gap between two images horizontally
        new_height = self.display_size[0] * row
        new_total_width = (self.display_size[1] + gap) * col
        
        # 显示的文本列表
        texts = ['Image', 'Ground Truth', 'SAM', 'Pred, ', 'Error image of pred']
        if mious != None:
            for i, (miou, ious) in enumerate(mious):
                # cal the non-zero classes in ious
                unique_classes = np.sum(np.array(ious) != 0)
                mIOU2 = np.sum(np.array(ious)) / unique_classes
                if i == 0:
                    texts[-2] += 'mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(self.num_classes, miou * 100,
                                            unique_classes, mIOU2 * 100)
                else:
                    texts.append('f_{}, mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(i, self.num_classes, miou * 100,
                                            unique_classes, mIOU2 * 100))
            for i in range(len(mious)-1):
                texts.append('Error image f_{}'.format(i + 1))
        else:
            for i in range(col):
                texts.append('mIoU f_{}'.format(i + 1))
            for i in range(col):
                texts.append('Error image f_{}'.format(i + 1))
        texts.append('Confidence')
        texts.append('Entropy')
        texts.append('Confidence {}'.format(thresholds[0]))
        texts.append('Entropy {:.2f}'.format(thresholds[1]))

        # 创建一个新的空白画布
        output_image = np.zeros((new_height, new_total_width, 3), dtype=np.uint8)

        # 逐个将图像水平放置在画布上
        current_width = 0
        for i, image in enumerate(images):
            if image is not None:
                image = cv2.resize(image, (self.display_size[1], self.display_size[0]), \
                        interpolation=cv2.INTER_LINEAR)
            else: # if image is None, create a black image
                image = np.zeros((self.display_size[0], self.display_size[1], 3), dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.putText(image, texts[i], (20, 50), \
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, color=(0, 0, 255), thickness=2)
            # first row
            if i < col:
                output_image[0*image.shape[0]:1*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # second row
            elif col <= i < 2 * col:
                output_image[1*image.shape[0]:2*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # third row
            elif col * 2 <= i < 3 * col:
                output_image[2*image.shape[0]:3*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # fourth row
            else:
                output_image[3*image.shape[0]:4*image.shape[0], current_width:current_width+image.shape[1], :] = image
            current_width += (image.shape[1] + gap)
            current_width = current_width % new_total_width
        
        # 显示结果图像
        cv2.imwrite(os.path.join(self.output_folder, 'horizontal', image_name + self.mask_suffix), output_image)
        # cv2.imshow('Images', output_image)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
    
    def save_ious(self, miou_0, ious_0, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3, 
                  miou_4, ious_4, miou_5, ious_5, image_name):
        miou_diff_1_0 = round((miou_1 - miou_0) * 100, 2)
        miou_diff_2_0 = round((miou_2 - miou_0) * 100, 2)
        miou_diff_3_0 = round((miou_3 - miou_0) * 100, 2)
        miou_diff_4_0 = round((miou_4 - miou_0) * 100, 2)
        miou_diff_5_0 = round((miou_5 - miou_0) * 100, 2)
        iou_diff_1_0 = [round((ious_1[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_2_0 = [round((ious_2[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_3_0 = [round((ious_3[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_4_0 = [round((ious_4[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_5_0 = [round((ious_5[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'UDA seg':  [round(miou_0 * 100, 2)] + [round(ious_0[i] * 100, 2) for i in range(len(ious_0))],
            'Fusion 1': [round(miou_1 * 100, 2)] + [round(ious_1[i] * 100, 2) for i in range(len(ious_1))],
            'Fusion 2': [round(miou_2 * 100, 2)] + [round(ious_2[i] * 100, 2) for i in range(len(ious_2))],
            'Fusion 3': [round(miou_3 * 100, 2)] + [round(ious_3[i] * 100, 2) for i in range(len(ious_3))],
            'Fusion 4': [round(miou_4 * 100, 2)] + [round(ious_4[i] * 100, 2) for i in range(len(ious_4))],
            'Fusion 5': [round(miou_5 * 100, 2)] + [round(ious_5[i] * 100, 2) for i in range(len(ious_5))],
            'Differ_1_0': [miou_diff_1_0] + iou_diff_1_0,
            'Differ_2_0': [miou_diff_2_0] + iou_diff_2_0,
            'Differ_3_0': [miou_diff_3_0] + iou_diff_3_0,
            'Differ_4_0': [miou_diff_4_0] + iou_diff_4_0,
            'Differ_5_0': [miou_diff_5_0] + iou_diff_5_0,
        })

        # save the miou and class ious
        data.to_csv(os.path.join(self.output_folder, 'ious', image_name + '.csv'), index=False)

    def get_error_image(self, predicted, ground_truth, pred_color):
        '''
        function: get the error image
        input: predicted, ground_truth
            predicted: [H, W]
            ground_truth: [H, W]
            pred_color: [H, W, 3]
        output: error_image on pred_color
        '''
        if self.num_classes == 16:
            ground_truth[ground_truth==9] = 255
            ground_truth[ground_truth==14] = 255
            ground_truth[ground_truth==16] = 255
        error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
        # predicted_color = self.color_segmentation(predicted)
        # change the area of error mask in pred_color to white
        pred_color_copy = copy.deepcopy(pred_color)
        pred_color_copy[error_mask == 0] = [255, 255, 255]
        
        # error_mask[pred != gt] = 255
        return pred_color_copy

    def save_binary_mask(self, image_name, mask):
        '''
        function: save the binary mask
        input: image_name, mask, mask_name
        output: save the mask in the output folder
        '''
        mask = mask.astype(np.uint8)
        mask = mask * 255
        cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), mask)
        
class Fusion_ACDC():
    def __init__(self, mask_folder=None, segmentation_folder=None, confidence_folder=None, entropy_folder=None,
                 image_folder=None, gt_folder=None, num_classes=None, sky_center_rect=None, road_center_rect=None,
                 mix_ratio=None, resize_ratio=None, output_folder=None, mask_suffix=None,
                 segmentation_suffix=None, segmentation_suffix_noimg=None,
                 confidence_suffix=None, entropy_suffix=None, gt_suffix=None, daytime_threshold=100, 
                 fusion_mode=None, sam_classes=None, shrink_num=None, display_size=(200, 400)):
        # the path to the sam mask
        self.mask_folder = mask_folder
        # the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        # the path to the confidence map
        self.confidence_folder = confidence_folder
        self.confidence_suffix = confidence_suffix
        # the path to the entropy map
        self.entropy_folder = entropy_folder
        self.entropy_suffix = entropy_suffix
        # the number of classes
        self.num_classes = num_classes
        # the rect of the sky center
        self.sky_center_rect = sky_center_rect
        # the rect of the road center
        self.road_center_rect = road_center_rect
        # the path to the ground truth folder
        self.gt_folder = gt_folder
        # self.gt_color_folder = self.gt_folder.replace('train_all', 'train_gt_color')
        # the path to the original image
        self.image_folder = image_folder
        # the mix ratio of the fusion result and original image
        self.mix_ratio = mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        # the path to the output folder
        self.output_folder = output_folder
        # the image suffix of the mask and segmentation result
        self.mask_suffix = mask_suffix
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        # the gt suffix
        self.gt_suffix = gt_suffix
        # daytime threshold
        self.daytime_threshold = daytime_threshold
        # the fusion mode
        self.fusion_mode = fusion_mode
        # the classes sam performs better
        self.sam_classes = sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = shrink_num
        # the size of the image
        self.display_size = display_size
        self.label_names = [trainid2label[train_id].name for train_id in range(19)]
        if self.num_classes == 16:
            self.label_names.remove('train')
            self.label_names.remove('truck')
            self.label_names.remove('terrain')
        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)  # GOPR0122_frame_000161_rgb_anon
        self.image_names.sort()
        self.daytime = True

        # make the folder to save the fusion result
        # the fusion result in trainID
        if self.output_folder is not None:
            self.check_and_make(os.path.join(self.output_folder, 'trainID'))
            # the fusion result in color
            # self.check_and_make(os.path.join(self.output_folder, 'color'))
            # the fusion result in color mixed with original image
            self.check_and_make(os.path.join(self.output_folder, 'mixed'))
            # make the folder to save the fusion result with segmentation result as the background
            # the fusion result in trainID with segmentation result as the background
            self.check_and_make(os.path.join(self.output_folder, 'trainID_bg'))
            # self.check_and_make(os.path.join(self.output_folder, 'color_bg'))
            # the fusion result in color with segmentation result as the background
            # the fusion result in color mixed with original image with segmentation 
            # result as the background
            self.check_and_make(os.path.join(self.output_folder, 'horizontal'))
            self.check_and_make(os.path.join(self.output_folder, 'mixed_bg'))
            self.check_and_make(os.path.join(self.output_folder, 'ious'))

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

    def get_sam_pred(self, image_name, segmentation, confidence_mask=None, entropy_mask=None):
        # DEBUG(cxy):　严老师，我给你加了一个新的参数original_image，原来的代码会报错，我不知道这样加对不对，现在可以把原始图像也传进来了
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
        #现在已经按照mask的面积进行了从大到小的排序,序号从0开始
        mask_names = natsorted(mask_names)
        
        # 将mask按照面积从大到小进行排序,由于占用融合时间,目前已经离线完成
        # mask_areas = []
        # for mask_name in mask_names:
        #     mask_path = os.path.join(self.mask_folder, image_name, mask_name)
        #     mask = cv2.imread(mask_path)  # [h,w,3]
        #     mask_area = np.sum(mask[:, :, 0] == 255)
        #     mask_areas.append(mask_area)
        # mask_names = [mask_name for _, mask_name in sorted(zip(mask_areas, mask_names), reverse=True)]
        
        sam_mask = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        for index, mask_name in enumerate(mask_names):
            mask_path = os.path.join(self.mask_folder, image_name, mask_name)
            mask = cv2.imread(mask_path)  # [h,w,3]
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]  # [N,]
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n, ], [n1, n2, n3, ...]
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            most_freq_id = num_ids[0]
            
            if len(counts) >= 2:
                if num_ids[0] == 2 and num_ids[1] == 3 and counts[1] / counts[0] >= 0.3:
                    # [building, wall]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    # [building, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.15:
                    # [building, pole]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.1:
                    # [building, traffic sign]
                    mask_shape = Mask_Shape(mask)
                    # if the mask is rectangular or triangular, then assign the mask with traffic sign
                    if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= 0.25:
                    # [wall, fence]
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 9 and num_ids[1] == 1:
                    # [terrain, sidewalk]
                    num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], mask[:, :, 0] == 255), entropy_mask))
                    num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], mask[:, :, 0] == 255), entropy_mask))
                    if num_id_1 > num_id_0:
                        most_freq_id = num_ids[1]
                # for acdc
                elif num_ids[0] == 0 and num_ids[1] == 1:
                    # [road, sidewalk]
                    if index == 0:
                        most_freq_id = 0
                    elif counts[1] / counts[0] >= 0.15:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 0 and num_ids[1] == 10:
                    # [road, sky]
                    if counts[1] / counts[0] >= 0.25:
                        most_freq_id = num_ids[1]
                elif num_ids[0] == 1 and num_ids[1] == 3:
                    # [sidewalk, wall]
                    if counts[1] / counts[0] >= 0.4:
                        most_freq_id = num_ids[1]
            if num_ids[0] == 8 and not self.daytime:  #夜间的天空总是被识别成vegetation
                mask_center = cal_center(mask[:, :, 0])
                if inside_rect(mask_center, self.sky_center_rect):
                    most_freq_id = 10
                    
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id  # 存在重叠的问题
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
        return sam_mask
        
    def color_segmentation(self, segmentation):
        #get the color segmentation result, initial the color segmentation result with black (0,0,0)
        #input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation)
        for train_id in train_ids:
            if self.num_classes == 16 and train_id in [9, 14, 16]:
                continue
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
        
    def fusion(self):
        bar = tqdm.tqdm(total=len(self.image_names))
        for image_name in self.image_names:
            #get the segmentation result
            prediction_path = os.path.join(self.segmentation_folder, image_name + self.segmentation_suffix)
            if self.segmentation_suffix_noimg:
                prediction_path = prediction_path.replace('_leftImg8bit', '')
            # print('load from: ', prediction_path)
            segmentation = cv2.imread(prediction_path) #[h, w, 3], 3 channels not 1 channel
            # print('prediction_path', prediction_path)
            
            #get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            
            #get the sam segmentation result using the mask
            sam_pred = self.get_sam_pred(image_name, segmentation)
            sam_color = self.color_segmentation(sam_pred)
            
            #get the mixed color image using the self.mix_ratio
            mixed_color = cv2.addWeighted(original_image, self.mix_ratio, sam_color, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color = cv2.resize(mixed_color, (int(mixed_color.shape[1] * self.resize_ratio), int(mixed_color.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            
            #save the sam mask in trainid and color to the output folder
            cv2.imwrite(os.path.join(self.output_folder, 'trainID', image_name + self.mask_suffix), sam_pred)
            # cv2.imwrite(os.path.join(self.output_folder, 'color', image_name + self.mask_suffix), fusion_color)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed', image_name + self.mask_suffix), mixed_color)

            #get the fusion result with the background
            if self.fusion_mode == 1:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 2:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_2(segmentation=segmentation, sam_pred=sam_pred)
            elif self.fusion_mode == 3:
                fusion_trainid_bg, fusion_color_bg = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
            # elif self.fusion_mode == 4:
            #     fusion_trainid_bg, fusion_color_bg = self.fusion_mode_4(segmentation=segmentation, sam_pred=sam_pred)
            else:
                # raise NotImplementedError
                raise NotImplementedError("This fusion mode has not been implemented yet.")
                             
            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), fusion_trainid_bg)
            # cv2.imwrite(os.path.join(self.output_folder, 'color_bg', image_name + self.mask_suffix), fusion_color_bg)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)
            # fusion_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            
            bar.update(1)
            # mask = cv2.imread(os.path.join(self.mask_folder, image_name + self.mask_suffix))
            
            # image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
            # mask = mask / 255
            # segmentation = segmentation / 255
            # mask = mask * self.mix_ratio
            # segmentation = segmentation * (1 - self.mix_ratio)
            # fusion = mask + segmentation
            # fusion = fusion * 255
            # fusion = fusion.astype('uint8')
            # fusion = cv2.cvtColor(fusion, cv2.COLOR_GRAY2BGR)
            # fusion = cv2.addWeighted(image, 0.5, fusion, 0.5, 0)
            # cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), fusion)

    def fusion_mode_1(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        train_ids = np.unique(sam_pred)
        train_ids = train_ids[train_ids != 255]
        for train_id in train_ids:
            fusion_trainid[sam_pred == train_id] = train_id
        # fusion_color = self.color_segmentation(fusion_trainid)
        
        #use the segmentation result to fill the pixels in fusion_trainid whose trainid is 255
        fusion_trainid_bg = fusion_trainid.copy()
        indexs = np.where(fusion_trainid == 255)
        fusion_trainid_bg[indexs] = segmentation[:, :, 0][indexs]      
        #use the corresponding color of segmentation result to fill the pixels in fusion_color whose trainid is 255
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg)
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)
        fusion_color_bg = fusion_color_bg.astype(np.uint8)
        
        return fusion_trainid_bg, fusion_color_bg

    def fusion_mode_2(self, segmentation, sam_pred):
        #initialize the fusion mask in trainid, fusion mask in color
        fusion_trainid = np.ones_like(segmentation[:, :, 0], dtype=np.uint8) * 255
        fusion_trainid = segmentation[:, :, 0].copy() #fill the fusion result all with segmentation result
        
        #use sam_pred with self.sam_classes to cover the fusion_trainid
        sam_pred_ids = np.unique(sam_pred)
        sam_pred_ids = sam_pred_ids[sam_pred_ids != 255]
        for sam_class in self.sam_classes:
            if sam_class in sam_pred_ids:
                fusion_trainid[sam_pred == sam_class] = sam_class
        
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_3(self, segmentation, sam_pred, fusion_trainid=None, 
                      confidence_mask=None, entropy_mask=None):
        '''
        segmentation: [h, w, 3]
        sam_pred: [h, w]
        '''
        if fusion_trainid is None:
            fusion_trainid_0, _ = self.fusion_mode_1(segmentation, sam_pred)
        else:
            # print('copy in fusion 3')
            fusion_trainid_0 = copy.deepcopy(fusion_trainid)
        fusion_ids = np.unique(fusion_trainid_0)
        # fusion_trainid_0: [h, w], fusion_color_0: [h, w, 3]
        # # 预测结果为road但是sam中和road对应的类别为sidewalk(分割成了同一个mask)，将预测结果改为road
        mask_road = ((segmentation[:, :, 0] == 0) & (fusion_trainid_0 == 1))
        # self.save_binary_mask('road before', mask_road)
        # self.save_binary_mask('confidence_mask', confidence_mask)
        if confidence_mask is not None:
            mask_road = np.logical_and(mask_road, confidence_mask)
        # 预测结果为siwalk但是sam中和siwalk对应的类别为road(分割成了同一个mask)，将预测结果改为siwalk
        mask_siwa = ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 0))
        if confidence_mask is not None:
            mask_siwa = np.logical_and(mask_siwa, confidence_mask)
        # if entropy_mask is not None:
            # mask_siwa = np.logical_and(mask_siwa, entropy_mask)
        mask_buil = ((segmentation[:, :, 0] == 2) & (fusion_trainid_0 == 10))
        # 预测结果为fence但是sam中和fence对应的类别为building(分割成了同一个mask)，将预测结果改为fence
        mask_fenc = ((segmentation[:, :, 0] == 4) & (fusion_trainid_0 == 2))
        # 预测结果为pole但是sam中和pole对应的类别为building/light/sign(分割成了同一个mask)，将预测结果改为pole
        mask_pole = ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 6))\
                    | ((segmentation[:, :, 0] == 5) & (fusion_trainid_0 == 7))
        # 预测结果为ligh但是sam中和ligh对应的类别为building/pole/vegetation(分割成了同一个mask)，将预测结果改为ligh
        mask_ligh = ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 2)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 5)) \
                    | ((segmentation[:, :, 0] == 6) & (fusion_trainid_0 == 8))
        # 预测结果为sign但是sam中和sign对应的类别为building/vegetation(分割成了同一个mask)，将预测结果改为sign
        mask_sign = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 2))\
                    | ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 8))
        mask_sign_2 = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 5))  # [H, W]
        # 预测结果为car但是sam中和car对应的类别为vegetation(分割成了同一个mask)，将预测结果改为car
        mask_car = ((segmentation[:, :, 0] == 13) & (fusion_trainid_0 == 8))  # 
        mask_bike = (segmentation[:, :, 0] == 18)
        if np.max(mask_sign_2):  # 如果mask_sign_2中有值
            # 注意要先试用np.newaxis将mask_sign_2的维度扩展为3维，
            # 再使用np.repeat将mask_sign_2的前两维在第三维复制3份
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)  # [h, w]->[3*h, w]
            # print('mask_sign_2.shape: ', mask_sign_2.shape, np.max(mask_sign_2), np.min(mask_sign_2))
            # mask_sign_2 = mask_sign_2.astype(np.uint8).repeat(3, axis=2)  # [h, w]->[h, w, 3]
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2

        # 预测结果为person但是sam中和person对应的类别为building(分割成了同一个mask)，将预测结果改为person
        mask_person = ((segmentation[:, :, 0] == 11) & (fusion_trainid_0 == 2))\
            
        fusion_trainid_0[mask_road] = 0
        # f0_road_mask = (fusion_trainid_0 == 0).astype(np.uint8)
        # if f0_road_mask.any() and not inside_rect(cal_center(f0_road_mask), self.road_center_rect):
        fusion_trainid_0[mask_siwa] = 1
        fusion_trainid_0[mask_fenc] = 4
        fusion_trainid_0[mask_pole] = 5
        fusion_trainid_0[mask_ligh] = 6
        fusion_trainid_0[mask_sign] = 7
        fusion_trainid_0[mask_person] = 11
        fusion_trainid_0[mask_car] = 13
        # print(fusion_ids)
        # if 18 not in fusion_ids:
        fusion_trainid_0[mask_bike] = 18
        fusion_color_0 = self.color_segmentation(fusion_trainid_0)
        return fusion_trainid_0, fusion_color_0

    def fusion_mode_4(self, segmentation, sam_pred, fusion_trainid=None, confidence_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use confidence_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            confidence_mask:[h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            # print('copy in fusion 4')
            fusion_trainid = copy.deepcopy(fusion_trainid)
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & confidence_mask
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & confidence_mask
        fusion_trainid[road_mask] = 0  # road
        fusion_trainid[side_mask] = 1  # sidewalk
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
    
    def fusion_mode_5(self, segmentation, sam_pred, fusion_trainid=None, entropy_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use entropy_mask to select model segmentation to the fusion result
        input: 
            segmentation:   [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            entropy_mask:   [h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        else:
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # [road, sidewalk]
        road_mask = (segmentation[:, :, 0] == 0) & (fusion_trainid == 1) & entropy_mask
        # [sidewalk, road]
        side_mask = (segmentation[:, :, 0] == 1) & (fusion_trainid == 0) & entropy_mask
        # [vegetation, sidewalk]
        vege_mask = (segmentation[:, :, 0] == 8) & (fusion_trainid == 1) & entropy_mask
        
        fusion_trainid[road_mask] = 0
        fusion_trainid[side_mask] = 1
        fusion_trainid[vege_mask] = 8
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_6(self, segmentation, sam_pred):
        #not so good
        fusion_trainid, fusion_color = self.fusion_mode_1(segmentation=segmentation, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]
        
        for class_id in unique_classes:
            #get the class mask in segmentation
            class_mask = (segmentation == class_id)
            #eroded the class mask in segmentation
            eroded_class_mask, area = shrink_region(class_mask, num_pixels=self.shrink_num)
            #assign the corresponding area in fusion_trainid with the class_id
            fusion_trainid[eroded_class_mask] = class_id
        fusion_trainid = fusion_trainid.astype(np.uint8)            
        
        fusion_color = self.color_segmentation(fusion_trainid)
        return fusion_trainid, fusion_color

    def trainid2color(self, trainid):
        '''
        function: convert trainID to color in cityscapes
        input: trainid
        output: color
        '''
        #if the input is a number in np.uint8, it means it is a trainid
        if type(trainid) == np.uint8:
            label_object = trainid2label[trainid]
            return label_object.color[::-1]
        else:
            color_mask = np.zeros((trainid.shape[0], 3), dtype=np.uint8)
            for i in range(trainid.shape[0]):
                label_object = trainid2label[trainid[i]]
                color_mask[i] = label_object.color[::-1]
            return color_mask
            
        
        
        # if type(trainid) == tuple: #a mask
        #     ###assign the color to the mask according to the trainid
        #     color_mask = np.zeros((trainid.shape[0], trainid.shape[1], 3), dtype=np.uint8)
        #     for i in range(trainid.shape[0]):
        #         for j in range(trainid.shape[1]):
        #             label_object = trainid2label[trainid[i, j]]
        #             color_mask[i, j] = label_object.color
        #     return color_mask
        # else: #one number
        #     label_object = trainid2label[trainid]
        #     return label_object.color   

    def visualize_numpy(self, np_array):
        # 创建子图和画布
        fig, axis = plt.subplots(figsize=(8, 4))
        # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(8, 4))

        # 可视化预测置信度数组
        im1 = axis.imshow(np_array, cmap='viridis')
        # ax1.set_title('Confidence')
        axis.set_xlabel('Width')
        axis.set_ylabel('Height')
        axis.axis('off')
        # fig.colorbar(im1, ax=ax1)
        divider1 = make_axes_locatable(axis)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax1)
        # return the figure in numpy format
        # 将图像转换为NumPy数组
        fig.canvas.draw()
        bgr_image = np.array(fig.canvas.renderer._renderer)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        
        plt.clf()
        plt.close(fig)
        
        # 裁剪图像，去除边界
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bgr_image = bgr_image[y:y+h, x:x+w]
        
        return bgr_image

    def vis_np_higher_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array > threshold
        input:
        output: high_mask, image
        '''
        image = image.copy()
        high_mask = np_array > threshold
        image[high_mask] = [0, 255, 0]  # BGR
        return high_mask, image

    def vis_np_lower_thres(self, np_array, image, threshold=0.0):
        '''
        function:
            mark the area on the image with np_array < threshold
        input:
        output: low_mask, image
        '''
        image = image.copy()
        low_mask = np_array < threshold
        image[low_mask] = [0, 255, 0]  # BGR
        return low_mask, image

    def dis_imgs_horizontal(self, images, image_name, mious, thresholds):
        '''
        function:
            display the images horizontally and save the result
        input:
            images: a list of images, 3 * 4 = 12 images
                    [image, ground truth, sam seg, model seg, error_0
                    fusion_1_result, fusion_2_result, fusion_3_result, fusion_4_result, fusion_5_result,
                    error_1, error_2, error_3, error_4, error_5,
                    confidence_map, entropy_map]
            images_name: the name of the image
            mious: a list of miou and ious,
                    (miou_0, ious_0), (miou_1, ious_1), (miou_2, ious_2), 
                    (miou_3, ious_3), (miou_4, ious_4), (miou_5, ious_5),
            thresholds: a list of thresholds
                    [confidence_threshold, entropy_threshold]
        '''
        # 获取最大高度和总宽度
        # max_height = max(image.shape[0] for image in images)
        # total_width = sum(image.shape[1] for image in images)
        col = 5
        row = len(images) // col + 1 if len(images) % col != 0 else len(images) // col
        gap = 10  # the gap between two images horizontally
        new_height = self.display_size[0] * row
        new_total_width = (self.display_size[1] + gap) * col
        
        # 显示的文本列表
        texts = ['Image', 'Ground Truth', 'SAM', 'Pred, ', 'Error image of pred']
        for i, (miou, ious) in enumerate(mious):
            # cal the non-zero classes in ious
            unique_classes = np.sum(np.array(ious) != 0)
            mIOU2 = np.sum(np.array(ious)) / unique_classes
            if i == 0:
                texts[-2] += 'mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100)
            else:
                texts.append('f_{}, mIoU{}: {:.2f} mIoU{}: {:.2f}'.format(i, self.num_classes, miou * 100,
                                        unique_classes, mIOU2 * 100))
        for i in range(len(mious)-1):
            texts.append('Error image f_{}'.format(i + 1))
        texts.append('Confidence')
        texts.append('Entropy')
        texts.append('Confidence {}'.format(thresholds[0]))
        texts.append('Entropy {:.2f}'.format(thresholds[1]))

        # 创建一个新的空白画布
        output_image = np.zeros((new_height, new_total_width, 3), dtype=np.uint8)

        # 逐个将图像水平放置在画布上
        current_width = 0
        for i, image in enumerate(images):
            image = cv2.resize(image, (self.display_size[1], self.display_size[0]), \
                    interpolation=cv2.INTER_LINEAR)
            image = cv2.putText(image, texts[i], (20, 50), \
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, color=(0, 0, 255), thickness=2)
            # first row
            if i < col:
                output_image[0*image.shape[0]:1*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # second row
            elif col <= i < 2 * col:
                output_image[1*image.shape[0]:2*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # third row
            elif col * 2 <= i < 3 * col:
                output_image[2*image.shape[0]:3*image.shape[0], current_width:current_width+image.shape[1], :] = image
            # fourth row
            else:
                output_image[3*image.shape[0]:4*image.shape[0], current_width:current_width+image.shape[1], :] = image
            current_width += (image.shape[1] + gap)
            current_width = current_width % new_total_width
        
        # 显示结果图像
        cv2.imwrite(os.path.join(self.output_folder, 'horizontal', image_name + self.mask_suffix), output_image)
        # cv2.imshow('Images', output_image)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
    
    def save_ious(self, miou_0, ious_0, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3, 
                  miou_4, ious_4, miou_5, ious_5, image_name):
        miou_diff_1_0 = round((miou_1 - miou_0) * 100, 2)
        miou_diff_2_0 = round((miou_2 - miou_0) * 100, 2)
        miou_diff_3_0 = round((miou_3 - miou_0) * 100, 2)
        miou_diff_4_0 = round((miou_4 - miou_0) * 100, 2)
        miou_diff_5_0 = round((miou_5 - miou_0) * 100, 2)
        iou_diff_1_0 = [round((ious_1[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_2_0 = [round((ious_2[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_3_0 = [round((ious_3[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_4_0 = [round((ious_4[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        iou_diff_5_0 = [round((ious_5[i] - ious_0[i]) * 100, 2) for i in range(len(ious_0))]
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'UDA seg':  [round(miou_0 * 100, 2)] + [round(ious_0[i] * 100, 2) for i in range(len(ious_0))],
            'Fusion 1': [round(miou_1 * 100, 2)] + [round(ious_1[i] * 100, 2) for i in range(len(ious_1))],
            'Fusion 2': [round(miou_2 * 100, 2)] + [round(ious_2[i] * 100, 2) for i in range(len(ious_2))],
            'Fusion 3': [round(miou_3 * 100, 2)] + [round(ious_3[i] * 100, 2) for i in range(len(ious_3))],
            'Fusion 4': [round(miou_4 * 100, 2)] + [round(ious_4[i] * 100, 2) for i in range(len(ious_4))],
            'Fusion 5': [round(miou_5 * 100, 2)] + [round(ious_5[i] * 100, 2) for i in range(len(ious_5))],
            'Differ_1_0': [miou_diff_1_0] + iou_diff_1_0,
            'Differ_2_0': [miou_diff_2_0] + iou_diff_2_0,
            'Differ_3_0': [miou_diff_3_0] + iou_diff_3_0,
            'Differ_4_0': [miou_diff_4_0] + iou_diff_4_0,
            'Differ_5_0': [miou_diff_5_0] + iou_diff_5_0,
        })

        # save the miou and class ious
        data.to_csv(os.path.join(self.output_folder, 'ious', image_name + '.csv'), index=False)

    def get_error_image(self, predicted, ground_truth, pred_color):
        '''
        function: get the error image
        input: predicted, ground_truth
            predicted: [H, W]
            ground_truth: [H, W]
            pred_color: [H, W, 3]
        output: error_image on pred_color
        '''
        if self.num_classes == 16:
            ground_truth[ground_truth==9] = 255
            ground_truth[ground_truth==14] = 255
            ground_truth[ground_truth==16] = 255
        error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
        # predicted_color = self.color_segmentation(predicted)
        # change the area of error mask in pred_color to white
        pred_color_copy = copy.deepcopy(pred_color)
        pred_color_copy[error_mask == 0] = [255, 255, 255]
        
        # error_mask[pred != gt] = 255
        return pred_color_copy

    def save_binary_mask(self, image_name, mask):
        '''
        function: save the binary mask
        input: image_name, mask, mask_name
        output: save the mask in the output folder
        '''
        mask = mask.astype(np.uint8)
        mask = mask * 255
        cv2.imwrite(os.path.join(self.output_folder, image_name + self.mask_suffix), mask)
    
    def is_daytime(self, image):
        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算图像的平均亮度值
        brightness = cv2.mean(gray_image)[0]

        # 判断亮度值是否超过阈值
        if brightness > self.daytime_threshold:
            self.daytime = True  # 白天
        else:
            self.daytime = False # 黑夜
        
def main():
    args = get_parse()
    fusion = Fusion(args)
    fusion.fusion()


if __name__ == '__main__':
    main()