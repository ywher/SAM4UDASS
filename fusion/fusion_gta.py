import argparse
import os
import cv2
from cityscapesscripts.helpers.labels import trainId2label as trainid2label
import numpy as np
import tqdm
from utils.shrink_mask import shrink_region
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt
from natsort import natsorted
from utils.mask_shape import Mask_Shape
from utils.cal_mask_center import cal_center, inside_rect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from segment_anything import sam_model_registry, SamPredictor
from tools.iou_perimg import SegmentationMetrics
from utils.segmentix import Segmentix
from utils.utils import AverageMeter

class Fusion_GTA():
    def __init__(self, 
                 mask_folder='', mask_folder_suffix='', mask_suffix='',
                 segmentation_folder='', segmentation_suffix='', segmentation_suffix_noimg=False,
                 confidence_folder='', confidence_suffix='', entropy_folder='', entropy_suffix='',
                 image_folder='', image_suffix='', gt_folder='', gt_suffix='',
                 output_folder='',
                 num_classes=19,
                 fusion_mode=1,
                 road_assumption=True,
                 road_center_rect=(740, 780, 1645, 995),
                 get_sam_mode=1,
                 use_sgml=True,
                 sam_alpha=0.2,
                 adaptive_ratio=False,
                 large_classes=[0, 1, 2, 8, 10, 13],
                 small_classes=[3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18],
                 sam_classes=[5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
                 shrink_num=2,
                 display_size=(200, 400),
                 mix_ratio=0.5,
                 resize_ratio=1.0,
                 time_process=True,
                 time_filename='time.txt',
                 save_sgml_process=False,
                 save_majority_process=False,
                 save_f1_process=False,
                 save_f2_process=False,
                 save_f3_process=False,
                 ):
        # the path to the sam mask
        self.mask_folder = mask_folder
        self.mask_folder_suffix = mask_folder_suffix
        self.mask_suffix = mask_suffix
        # the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        # the path to the confidence map
        self.confidence_folder = confidence_folder
        self.confidence_suffix = confidence_suffix
        # the path to the entropy map
        self.entropy_folder = entropy_folder
        self.entropy_suffix = entropy_suffix
        # the path to the original image
        self.image_folder = image_folder
        self.image_suffix = image_suffix
        self.gt_folder = gt_folder
        self.gt_suffix = gt_suffix
        # the path to the output folder
        self.output_folder = output_folder
        
        
        # the number of classes
        self.num_classes = num_classes
        
        ### SAM mask labeling process
        # get sam mode
        self.get_sam_mode = get_sam_mode
        
        # SGML params
        self.use_sgml = use_sgml
        self.small_classes = small_classes
        self.large_classes = large_classes
        
        self.sam_alpha = sam_alpha
        self.adaptive_ratio = adaptive_ratio
        
        # the rect of the road center
        self.road_assumption = road_assumption
        self.road_center_rect = road_center_rect
        
        
        # fusion params
        self.fusion_mode = fusion_mode
        # the classes sam performs better， for fusion mode 2
        self.sam_classes = sam_classes
        # the shrink num of segmentation mask
        self.shrink_num = shrink_num
        
            
        # display params
        self.display_size = display_size
        self.mix_ratio = mix_ratio
        # the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        
        
        # save params
        self.save_sgml_process = save_sgml_process
        self.save_majority_process = save_majority_process
        self.save_f1_process = save_f1_process
        self.save_f2_process = save_f2_process
        self.save_f3_process = save_f3_process
        
        
        self.label_names = [trainid2label[train_id].name for train_id in range(19)]
        if self.num_classes == 16:
            self.label_names.remove('train')
            self.label_names.remove('truck')
            self.label_names.remove('terrain')
        # one folder corresponds to one image name without suffix
        self.image_names = os.listdir(self.mask_folder)
        self.image_names.sort()
        
        
        self.time_process = time_process
        self.time_filename = time_filename
        if self.time_process:
            self.sample_num = 0
            self.total_majority_time = AverageMeter()
            self.total_sgml_time = AverageMeter()
            self.fusion1_time = AverageMeter()
            self.fusion2_time = AverageMeter()
            self.fusion3_time = AverageMeter()

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
            os.makedirs(path, exist_ok=True)

    def get_sam_pred(self, image_name, segmentation, confidence_mask=None, entropy_mask=None):
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        segmentation: [H, W]
        to do: add the confidence threshold of segmentation result
        '''
        self.sample_num += 1
        # get the mask names
        time_name = time.time()
        # The masks have now been sorted in descending order based on their area, with the index starting from 0.
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name + self.mask_folder_suffix)) if self.mask_suffix in name]
        mask_names = natsorted(mask_names)
        time_name = time.time() - time_name
        
        time_mask = time.time()
        sam_mask = np.ones_like(segmentation, dtype=np.uint8) * 255
        sam_mask_majority = copy.deepcopy(sam_mask)
        time_mask = time.time() - time_mask
        for index, mask_name in enumerate(mask_names):
            # get the binary mask
            mask_path = os.path.join(self.mask_folder, image_name + self.mask_folder_suffix, mask_name)
            mask = cv2.imread(mask_path, 0)  # [h,w,3]
            start_time = time.time()
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            
            # get the number of trainids in the segmentation result using the mask with value 255
            # Get train IDs where mask equals 255
            trainids = segmentation[mask == 255]  # [N,]

            # Find unique IDs and their counts
            num_ids, counts = np.unique(trainids, return_counts=True)  # [n,], [n1, n2, n3, ...]

            # Sort indices based on counts in descending order
            sorted_indices = np.argsort(counts)[::-1]

            # Sort num_ids and counts based on sorted indices
            num_ids = num_ids[sorted_indices]
            counts = counts[sorted_indices]

            # Most frequent ID is now the first element of sorted num_ids
            most_freq_id = num_ids[0]

            sam_mask_majority[mask == 255] = most_freq_id
            self.total_majority_time.update(time.time() - start_time) 
            
            if len(counts) >= 2:
                # large and small classes
                if num_ids[0] in self.large_classes and num_ids[1] in self.small_classes and counts[1] / counts[0] >= self.sam_alpha:
                    most_freq_id = num_ids[1]
                
                # specific class and alpha ratio
                '''
                # building, [wall, fence, pole, traffic sign]
                if num_ids[0] == 2 and num_ids[1] in [3,4,5,7] and counts[1] / counts[0] >= self.sam_alpha:
                    most_freq_id = num_ids[1]
                # [wall, fence]
                elif num_ids[0] == 3 and num_ids[1] == 4 and counts[1] / counts[0] >= self.sam_alpha:  # 0.25
                    most_freq_id = num_ids[1]
                # [vegetation, terrain]
                elif num_ids[0] == 8 and num_ids[1] == 9 and counts[1] / counts[0] >= self.sam_alpha:  # 0.05
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 9 and num_ids[1] == 1 and counts[1] / counts[0] >= self.sam_alpha:  # 0.05
                    most_freq_id = num_ids[1]
                elif num_ids[0] == 1 and num_ids[1] == 18 and counts[1] / counts[0] >= self.sam_alpha:  #0.15
                    most_freq_id = num_ids[1]
                '''
                
                # specific class and specific ratio
                '''
                # [building, wall]
                # if num_ids[0] == 2 and num_ids[1] == 3 and counts[1] / counts[0] >= self.sam_alpha:  # 0.3
                #     most_freq_id = num_ids[1]
                # [building, fence]
                # elif num_ids[0] == 2 and num_ids[1] == 4 and counts[1] / counts[0] >= self.sam_alpha:  # 0.25
                #     most_freq_id = num_ids[1]
                # [building, pole]
                # elif num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= self.sam_alpha:  # 0.15
                #     most_freq_id = num_ids[1]
                # [building, traffic sign]
                # elif num_ids[0] == 2 and num_ids[1] == 7 and counts[1] / counts[0] >= 0.05: #self.sam_alpha:  # 0.1 0.04 for sample aachen_000111
                #     # print(index, mask_name)
                #     # print('counts', counts[0], counts[1], counts[1] / counts[0])
                #     # print() 
                #     # mask_shape = Mask_Shape(mask)
                #     # # if the mask is rectangular or triangular, then assign the mask with traffic sign
                #     # if mask_shape.is_approx_rectangular() or mask_shape.is_approx_triangular():
                #         # print('true')
                #     most_freq_id = num_ids[1]
                # [terrain, sidewalk]
                # elif num_ids[0] == 9 and num_ids[1] == 1:
                #     num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation == num_ids[0], mask == 255), confidence_mask))
                #     num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation == num_ids[1], mask == 255), confidence_mask))
                #     if num_id_1 > num_id_0:
                #         most_freq_id = num_ids[1]
                    # num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation == num_ids[0], mask == 255), confidence_mask))
                    # num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation == num_ids[1], mask == 255), confidence_mask))
                    # if num_id_1 > num_id_0:
                    #     most_freq_id = num_ids[1]
                # for synthia
                # [vegetation, building], 窗户被判断为vegetation
                # elif num_ids[0] == 8 and num_ids[1] == 2:
                #     num_id_0 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[0], 
                #                                     mask[:, :, 0] == 255), confidence_mask))
                #     num_id_1 = np.sum(np.logical_and(np.logical_and(segmentation[:,:,0] == num_ids[1], 
                #                                     mask[:, :, 0] == 255), confidence_mask))
                #     if num_id_0 ==0 or num_id_1 / num_id_0 > 0.25:  # 0.25
                #         most_freq_id = num_ids[1]
                # # [road, sidewalk]
                # elif num_ids[0] == 0 and num_ids[1] == 1:
                #     if index == 0:
                #         most_freq_id = 0
                #     elif counts[1] / counts[0] >= self.sam_alpha:
                #         most_freq_id = num_ids[1]
                # [sidewalk, bicycle]
                # elif (num_ids[0] == 1 and num_ids[1] == 0) or \
                #     (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                #     # [sidewalk, road]
                #     mask_center = cal_center(mask[:, :, 0])
                #     if inside_rect(mask_center, self.road_center_rect) or index == 0:
                #         most_freq_id = 0
                '''
                
                # road assupmtion
                if self.road_assumption:
                    if (num_ids[0] == 1 and num_ids[1] == 0) or \
                    (len(counts) >= 3 and num_ids[0] == 1 and num_ids[2] == 0):
                        # [sidewalk, road]
                        mask_center = cal_center(mask[:, :])
                        if inside_rect(mask_center, self.road_center_rect) or index == 0:
                            most_freq_id = 0
                    
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask == 255] = most_freq_id  # 存在重叠的问题
            self.total_sgml_time.update(time.time() - start_time)
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
        # self.total_majority_time += (time_mask + time_name)
        # self.total_sgml_time += (time_mask + time_name)
        # print('time name {}, time mask {}'.format(time_name, time_mask))
        return sam_mask, sam_mask_majority
        
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
    
    def update_mask(self, base_mask, confidence_mask):
        return np.logical_and(base_mask, confidence_mask) if confidence_mask is not None else base_mask
    
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

    def fusion_mode_1(self, uda_pred, sam_pred, image_name=''):
        '''
        uda_pred: uda pseudo label
        sam_pred: sam pseudo label
        image_name: the name of the mask and image
        '''
        start_time = time.time()

        # Instead of using deep copy, directly modify sam_pred copy
        fusion_trainid_bg = sam_pred.copy()

        # Find indices where the value is 255 and replace them
        mask = (fusion_trainid_bg == 255)
        fusion_trainid_bg[mask] = uda_pred[mask]

        # Convert to uint8 only once at the end
        fusion_trainid_bg = fusion_trainid_bg.astype(np.uint8)

        # Update time
        self.fusion1_time.update(time.time() - start_time)

        # Get the color segmentation for the fusion result
        fusion_color_bg = self.color_segmentation(fusion_trainid_bg).astype(np.uint8)

        return fusion_trainid_bg, fusion_color_bg
    
    def fusion_mode_2(self, uda_pred, sam_pred, image_name=''):
        '''
        uda_pred: uda pseudo label
        sam_pred: sam pseudo label
        image_name: the name of the mask and image
        '''
        start_time = time.time()

        # Use shallow copy and avoid redundant type casting
        fusion_trainid = uda_pred.copy().astype(np.uint8)

        # Filter out the invalid class (255) and intersect with sam_classes
        sam_pred_ids = np.unique(sam_pred)
        valid_sam_classes = set(sam_pred_ids[sam_pred_ids != 255]) & set(self.sam_classes)

        # Update fusion_trainid based on valid sam_classes
        for sam_class in valid_sam_classes:
            fusion_trainid[sam_pred == sam_class] = sam_class

        # Update time
        self.fusion2_time.update(time.time() - start_time)

        # Generate fusion color
        fusion_color = self.color_segmentation(fusion_trainid)

        return fusion_trainid, fusion_color

    '''
    ### old version
    def fusion_mode_3(self, uda_pred, sam_pred, fusion_trainid=None, 
                      confidence_mask=0, entropy_mask=0, image_name=''):
        
        # uda_pred: [h, w]
        # sam_pred: [h, w]
        
        start_time = time.time()
        if fusion_trainid is None:
            fusion_1_trainid, _ = self.fusion_mode_1(uda_pred = uda_pred, sam_pred = sam_pred, image_name = image_name)
        else:
            fusion_1_trainid = copy.deepcopy(fusion_trainid)
        # fusion_ids = np.unique(fusion_1_trainid)
        # fusion_1_trainid: [h, w], fusion_color_0: [h, w, 3]
        
        # road, [sidewalk]
        mask_road = ((uda_pred == 0) & (fusion_1_trainid == 1))
        # if confidence_mask is not None:
        mask_road = np.logical_and(mask_road, confidence_mask)
        # self.save_binary_mask('road before', mask_road)
        # self.save_binary_mask('confidence_mask', confidence_mask)
        
        # sidewalk, [road, terrain]
        mask_siwa = ((uda_pred == 1) & (fusion_1_trainid == 0)) \
                    | ((uda_pred == 1) & (fusion_1_trainid == 9))
        # if confidence_mask is not None:
        mask_siwa = np.logical_and(mask_siwa, confidence_mask)
        # if entropy_mask is not None:
            # mask_siwa = np.logical_and(mask_siwa, entropy_mask)
        
        # building, [fence, sky]
        mask_buil = ((uda_pred == 2) & (fusion_1_trainid == 10))\
                    | ((uda_pred == 2) & (fusion_1_trainid == 4))
        # if confidence_mask is not None:
        mask_buil = np.logical_and(mask_buil, confidence_mask)
        
        # wall, [vegetation]
        mask_wall = (uda_pred == 3) & (fusion_1_trainid == 8)
        # if confidence_mask is not None:
        mask_wall = np.logical_and(mask_wall, confidence_mask)
        
        # fence, [building, vegetation]
        mask_fenc = ((uda_pred == 4) & (fusion_1_trainid == 2))\
                    | ((uda_pred == 4) & (fusion_1_trainid == 8))
        
        # pole, [building, traffic light, traffic sign]
        mask_pole = ((uda_pred == 5) & (fusion_1_trainid == 2))\
                    | ((uda_pred == 5) & (fusion_1_trainid == 6))\
                    | ((uda_pred == 5) & (fusion_1_trainid == 7))

        # traffic light, [building, pole, vegetation]
        mask_ligh = ((uda_pred == 6) & (fusion_1_trainid == 2)) \
                    | ((uda_pred == 6) & (fusion_1_trainid == 5)) \
                    | ((uda_pred == 6) & (fusion_1_trainid == 8))
                    
        # traffic sign, [building, vegetation, traffic light]
        mask_sign = ((uda_pred == 7) & (fusion_1_trainid == 2))\
                    | ((uda_pred == 7) & (fusion_1_trainid == 8))\
                    | ((uda_pred == 7) & (fusion_1_trainid == 6))
        # traffic sign, [pole]
        mask_sign_2 = ((uda_pred == 7) & (fusion_1_trainid == 5))  # [H, W]
        
        # vegetation, [terrain, building]
        mask_vege = ((uda_pred == 8) & (fusion_1_trainid == 9))\
                    | ((uda_pred == 8) & (fusion_1_trainid == 2))
        # if confidence_mask is not None:
        mask_vege = np.logical_and(mask_vege, confidence_mask)
        
        # person, [building]
        mask_person = ((uda_pred == 11) & (fusion_1_trainid == 2))
        # if confidence_mask is not None:
        mask_person = np.logical_and(mask_person, confidence_mask)
        
        # car, [vegetation]
        mask_car = ((uda_pred == 13) & (fusion_1_trainid == 8))
        
        # bike
        mask_bike = (uda_pred == 18)
        if np.max(mask_sign_2):  # 如果mask_sign_2中有值
            # np.newaxis, mask_sign_2 [H, W] -> [H, W, 1]
            # np.repeat, mask_sign_2 [H, W, 1] -> [H, W, 3]
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)  # [h, w]->[3*h, w]
            # print('mask_sign_2.shape: ', mask_sign_2.shape, np.max(mask_sign_2), np.min(mask_sign_2))
            # mask_sign_2 = mask_sign_2.astype(np.uint8).repeat(3, axis=2)  # [h, w]->[h, w, 3]
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2


        fusion_1_trainid[mask_road] = 0
        # f0_road_mask = (fusion_1_trainid == 0).astype(np.uint8)
        # if f0_road_mask.any() and not inside_rect(cal_center(f0_road_mask), self.road_center_rect):
        fusion_1_trainid[mask_siwa] = 1
        fusion_1_trainid[mask_buil] = 2
        fusion_1_trainid[mask_wall] = 3
        fusion_1_trainid[mask_fenc] = 4
        fusion_1_trainid[mask_pole] = 5
        fusion_1_trainid[mask_ligh] = 6
        fusion_1_trainid[mask_sign] = 7
        fusion_1_trainid[mask_vege] = 8
        fusion_1_trainid[mask_person] = 11
        fusion_1_trainid[mask_car] = 13
        fusion_1_trainid[mask_bike] = 18
        self.fusion3_time.update(time.time() - start_time)
        
        fusion_3_trainid = copy.deepcopy(fusion_1_trainid)
        fusion_3_color = self.color_segmentation(fusion_1_trainid)
        return fusion_3_trainid, fusion_3_color
    '''
    
    def fusion_mode_3(self, uda_pred, sam_pred, fusion_trainid=None, 
                  confidence_mask=0, entropy_mask=0, image_name=''):
        '''
        uda_pred: [h, w]
        sam_pred: [h, w]
        '''
        start_time = time.time()

        # Use fusion_mode_1 result if fusion_trainid is not provided
        fusion_1_trainid = self.fusion_mode_1(uda_pred, sam_pred, image_name)[0] if fusion_trainid is None else fusion_trainid.copy()

        # Initialize masks
        mask_road = self.update_mask((uda_pred == 0) & (fusion_1_trainid == 1), confidence_mask)
        mask_siwa = self.update_mask(((uda_pred == 1) & ((fusion_1_trainid == 0) | (fusion_1_trainid == 9))), confidence_mask)
        mask_buil = self.update_mask(((uda_pred == 2) & ((fusion_1_trainid == 10) | (fusion_1_trainid == 4))), confidence_mask)
        mask_wall = self.update_mask((uda_pred == 3) & (fusion_1_trainid == 8), confidence_mask)
        mask_fenc = ((uda_pred == 4) & ((fusion_1_trainid == 2) | (fusion_1_trainid == 8)))
        mask_pole = ((uda_pred == 5) & ((fusion_1_trainid == 2) | (fusion_1_trainid == 6) | (fusion_1_trainid == 7)))
        mask_ligh = ((uda_pred == 6) & ((fusion_1_trainid == 2) | (fusion_1_trainid == 5) | (fusion_1_trainid == 8)))
        mask_sign = ((uda_pred == 7) & ((fusion_1_trainid == 2) | (fusion_1_trainid == 8) | (fusion_1_trainid == 6)))
        mask_vege = self.update_mask(((uda_pred == 8) & ((fusion_1_trainid == 9) | (fusion_1_trainid == 2))), confidence_mask)
        mask_person = self.update_mask((uda_pred == 11) & (fusion_1_trainid == 2), confidence_mask)
        mask_car = (uda_pred == 13) & (fusion_1_trainid == 8)
        mask_bike = (uda_pred == 18)
        
        # Additional handling for mask_sign_2
        mask_sign_2 = ((uda_pred == 7) & (fusion_1_trainid == 5))
        if np.max(mask_sign_2):
            mask_sign_2_img = np.repeat(mask_sign_2.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
            mask_shape_sign = Mask_Shape(mask_sign_2_img)
            if mask_shape_sign.is_approx_circle:
                mask_sign = mask_sign | mask_sign_2

        # Update fusion_trainid based on masks
        fusion_1_trainid[mask_road] = 0
        fusion_1_trainid[mask_siwa] = 1
        fusion_1_trainid[mask_buil] = 2
        fusion_1_trainid[mask_wall] = 3
        fusion_1_trainid[mask_fenc] = 4
        fusion_1_trainid[mask_pole] = 5
        fusion_1_trainid[mask_ligh] = 6
        fusion_1_trainid[mask_sign] = 7
        fusion_1_trainid[mask_vege] = 8
        fusion_1_trainid[mask_person] = 11
        fusion_1_trainid[mask_car] = 13
        fusion_1_trainid[mask_bike] = 18

        # Update processing time
        self.fusion3_time.update(time.time() - start_time)

        # Generate the final output
        fusion_3_trainid = fusion_1_trainid.copy()
        fusion_3_color = self.color_segmentation(fusion_1_trainid)
        
        return fusion_3_trainid, fusion_3_color

    def fusion_mode_4(self, uda_pred, sam_pred, fusion_trainid=None, confidence_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use confidence_mask to select model uda_pred to the fusion result
        input: 
            uda_pred:       [h, w],     uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            confidence_mask:[h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(uda_pred=uda_pred, sam_pred=sam_pred)
        else:
            # print('copy in fusion 4')
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # road, [sidewalk]
        road_mask = (uda_pred == 0) & (fusion_trainid == 1) & confidence_mask
        # sidewalk, [road]
        side_mask = (uda_pred == 1) & (fusion_trainid == 0) & confidence_mask
        # building, [traffic sign]
        buil_mask = (uda_pred == 2) & (fusion_trainid == 7) & confidence_mask
        fusion_trainid[road_mask] = 0  # road
        fusion_trainid[side_mask] = 1  # sidewalk
        fusion_trainid[buil_mask] = 2  # building
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
    
    def fusion_mode_5(self, uda_pred, sam_pred, fusion_trainid=None, entropy_mask=None):
        '''
        author: weihao_yan
        date:   2023-6-26
        function: 
            based on fusion_mode_3, 
            use entropy_mask to select model uda_pred to the fusion result
        input: 
            uda_pred:       [h, w, 3],  uint8, from class 0 to 18
            sam_pred:       [h, w],     uint8, from class 0 to 18
            entropy_mask:   [h, w],     bool,
        output:
            fusion_trainid: [h, w],     uint8, from class 0 to 18
            fusion_color:   [h, w, 3],  uint8,
        '''
        if fusion_trainid is None:
            fusion_trainid, _ = self.fusion_mode_3(uda_pred=uda_pred, sam_pred=sam_pred)
        else:
            fusion_trainid = copy.deepcopy(fusion_trainid)
        # [road, sidewalk]
        road_mask = (uda_pred == 0) & (fusion_trainid == 1) & entropy_mask
        # [sidewalk, road]
        side_mask = (uda_pred == 1) & (fusion_trainid == 0) & entropy_mask
        # [building, traffic sign]
        buil_mask = (uda_pred == 2) & (fusion_trainid == 7) & entropy_mask
        # [vegetation, sidewalk]
        vege_mask = (uda_pred == 8) & (fusion_trainid == 1) & entropy_mask
        
        fusion_trainid[road_mask] = 0
        fusion_trainid[side_mask] = 1
        # newly added
        fusion_trainid[buil_mask] = 2  # building
        fusion_trainid[vege_mask] = 8  # 
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color

    def fusion_mode_6(self, uda_pred, sam_pred):
        # not so good
        fusion_trainid, fusion_color = self.fusion_mode_1(uda_pred=uda_pred, sam_pred=sam_pred)
        unique_classes = np.unique(fusion_trainid)
        unique_classes = unique_classes[unique_classes != 255]
        
        for class_id in unique_classes:
            #get the class mask in uda_pred
            class_mask = (uda_pred == class_id)
            #eroded the class mask in uda_pred
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
        # If input is a single trainid (np.uint8), return the corresponding color
        if type(trainid) == np.uint8:
            return trainid2label[trainid].color[::-1]

        # Vectorized operation for array inputs
        color_mask = np.array([trainid2label[tid].color[::-1] for tid in trainid], dtype=np.uint8)

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
            mIOU2 = np.sum(np.array(ious)) / (unique_classes + 1e-5)
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
    
    '''
    ### old version
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
    '''
    
    def save_ious(self, miou_values, ious_values, image_name):
        '''
        miou_values = [miou_0, miou_1, miou_2, miou_3, miou_4, miou_5]
        ious_values = [
            ious_0,  # List of IoUs for Fusion 0
            ious_1,  # List of IoUs for Fusion 1
            ious_2,  # List of IoUs for Fusion 2
            ious_3,  # List of IoUs for Fusion 3
            ious_4,  # List of IoUs for Fusion 4
            ious_5   # List of IoUs for Fusion 5
        ]
        '''
        # Calculate mIoU differences between Fusion results and UDA segmentation
        miou_diffs = [round((miou_values[i] - miou_values[0]) * 100, 2) for i in range(1, 4)]

        # Calculate IoU differences for each class between Fusion results and UDA segmentation
        iou_diffs = [
            [round((ious_values[i][j] - ious_values[0][j]) * 100, 2) for j in range(len(ious_values[0]))] 
            for i in range(1, 4)
        ]

        # Prepare data for saving into a DataFrame
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'UDA seg': [round(miou_values[0] * 100, 2)] + [round(ious_values[0][i] * 100, 2) for i in range(len(ious_values[0]))],
            'Fusion 1': [round(miou_values[1] * 100, 2)] + [round(ious_values[1][i] * 100, 2) for i in range(len(ious_values[1]))],
            'Fusion 2': [round(miou_values[2] * 100, 2)] + [round(ious_values[2][i] * 100, 2) for i in range(len(ious_values[2]))],
            'Fusion 3': [round(miou_values[3] * 100, 2)] + [round(ious_values[3][i] * 100, 2) for i in range(len(ious_values[3]))],
            # 'Fusion 4': [round(miou_values[4] * 100, 2)] + [round(ious_values[4][i] * 100, 2) for i in range(len(ious_values[4]))],
            # 'Fusion 5': [round(miou_values[5] * 100, 2)] + [round(ious_values[5][i] * 100, 2) for i in range(len(ious_values[5]))],
            'Differ_1_0': [miou_diffs[0]] + iou_diffs[0],
            'Differ_2_0': [miou_diffs[1]] + iou_diffs[1],
            'Differ_3_0': [miou_diffs[2]] + iou_diffs[2],
            # 'Differ_4_0': [miou_diffs[3]] + iou_diffs[3],
            # 'Differ_5_0': [miou_diffs[4]] + iou_diffs[4],
        })

        # Save the mIoU and class IoUs to a CSV file
        data.to_csv(os.path.join(self.output_folder, 'ious', f'{image_name}.csv'), index=False)


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
    
    def show_time_process(self):
        print('total sample:', self.sample_num)
        print(f'avg major time: {self.total_majority_time.avg:.4f}')
        print(f'avg sgml time: {self.total_sgml_time.avg:.4f}')
        print(f'avg fusion1 time: {self.fusion1_time.avg:.4f}')
        print(f'avg fusion2 time: {self.fusion2_time.avg:.4f}')
        print(f'avg fusion3 time: {self.fusion3_time.avg:.4f}')
        print(f'avg fusion1+fusion3 time: {self.fusion1_time.avg + self.fusion3_time.avg:.4f}')
        # save the time result to txt
        with open(os.path.join('outputs', self.time_filename), 'w') as f:
            f.write('total sample: {}\n'.format(self.sample_num))
            f.write(f'avg major time: {self.total_majority_time.avg:.4f}\n')
            f.write(f'avg sgml time: {self.total_sgml_time.avg:.4f}\n')
            f.write(f'avg fusion1 time: {self.fusion1_time.avg:.4f}\n')
            f.write(f'avg fusion2 time: {self.fusion2_time.avg:.4f}\n')
            f.write(f'avg fusion3 time: {self.fusion3_time.avg:.4f}\n')
            f.write(f'avg fusion1+fusion3 time: {self.fusion1_time.avg + self.fusion3_time.avg:.4f}\n')
        f.close()