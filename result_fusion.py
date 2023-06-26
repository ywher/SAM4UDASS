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
# from matplotlib import pyplot as plt
import pandas as pd
import copy
import matplotlib.pyplot as plt
from utils.mask_shape import Mask_Shape
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


class Fusion2():
    def __init__(self, mask_folder, segmentation_folder, confidence_folder, entropy_folder,
                 image_folder, gt_folder, mix_ratio,
                 resize_ratio, output_folder, mask_suffix,
                 segmentation_suffix, segmentation_suffix_noimg,
                 confidence_suffix, entropy_suffix,
                 gt_suffix, fusion_mode, sam_classes,
                 shrink_num, display_size=(200, 400)):
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
        # the path to the ground truth folder
        self.gt_folder = gt_folder
        self.gt_color_folder = self.gt_folder.replace('train_all', 'train_gt_color')
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

    def get_sam_pred(self, image_name, segmentation):
        '''
        use the mask from sam and the prediction from uda
        output the trainid and color mask
        to do: add the confidence threshold of segmentation result
        '''
        # get the mask names
        mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
        mask_names.sort()
        
        # sort the mask names accrording to the mask area from large to small
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
            # print('mask name', mask_name)
            # cv2.imshow('mask', cv2.resize(mask, (512,256)))
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # get the number of trainids in the segmentation result using the mask with value 255
            trainids = segmentation[:, :, 0][mask[:, :, 0] == 255]
            num_ids, counts = np.unique(trainids, return_counts=True)
            # sort the num_ids according to the counts
            num_ids = [num_id for _, num_id in sorted(zip(counts, num_ids), reverse=True)]
            counts = sorted(counts, reverse=True)
            # print the top 3 classes
            # print('image_name: ', image_name)
            # print('class', num_ids[:3])
            # print('class names', [self.label_names[num_id] for num_id in num_ids[:3]])
            # print('counts', counts[:3])
            # get the most frequent trainid
            most_freq_id = num_ids[0]
            
            if len(counts) >= 2:
                if num_ids[0] == 2 and num_ids[1] == 5 and counts[1] / counts[0] >= 0.1:
                    # [building, pole]
                    # if the building is the first class and the pole is the second class, 
                    # and the ratio of pole to building is larger than 0.25
                    # then assign the mask with pole
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
            
            # fill the sam mask using the most frequent trainid in segmentation
            sam_mask[mask[:, :, 0] == 255] = most_freq_id
            # print('mask_name {}, most_freq_id{}'.format(mask_name, most_freq_id))
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

    def fusion_mode_3(self, segmentation, sam_pred):
        '''
        segmentation: [h, w, 3]
        sam_pred: [h, w]
        '''
        fusion_trainid_0, fusion_color_0 = self.fusion_mode_1(segmentation, sam_pred)
        # fusion_trainid_0: [h, w], fusion_color_0: [h, w, 3]
        # # 预测结果为road但是sam中和road对应的类别为sidewalk(分割成了同一个mask)，将预测结果改为road
        # mask_road = ((segmentation[:, :, 0] == 0) & (fusion_trainid_0 == 1))
        # 预测结果为siwalk但是sam中和siwalk对应的类别为road(分割成了同一个mask)，将预测结果改为siwalk
        mask_siwa = ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 0))\
                    | ((segmentation[:, :, 0] == 1) & (fusion_trainid_0 == 0))
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
        mask_sign_2 = ((segmentation[:, :, 0] == 7) & (fusion_trainid_0 == 5))  # [H, W]'
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
            
        # fusion_trainid_0[mask_road] = 0
        fusion_trainid_0[mask_siwa] = 1
        fusion_trainid_0[mask_fenc] = 4
        fusion_trainid_0[mask_pole] = 5
        fusion_trainid_0[mask_ligh] = 6
        fusion_trainid_0[mask_sign] = 7
        fusion_trainid_0[mask_person] = 11
        fusion_color_0 = self.color_segmentation(fusion_trainid_0)
        return fusion_trainid_0, fusion_color_0

    def fusion_mode_4(self, segmentation, sam_pred, confidence_mask):
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
        fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        fusion_trainid[confidence_mask] = segmentation[:, :, 0][confidence_mask]
        fusion_color = self.color_segmentation(fusion_trainid)
        
        return fusion_trainid, fusion_color
    
    def fusion_mode_5(self, segmentation, sam_pred, entropy_mask):
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
        fusion_trainid, _ = self.fusion_mode_3(segmentation=segmentation, sam_pred=sam_pred)
        fusion_trainid[entropy_mask] = segmentation[:, :, 0][entropy_mask]
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
        
        # 裁剪图像，去除边界
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bgr_image = bgr_image[y:y+h, x:x+w]
        return bgr_image

    def visualize_numpy_higher_threshold(self, np_array, image, threshold=0.0):
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

    def visualize_numpy_lower_threshold(self, np_array, image, threshold=0.0):
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

    def display_images_horizontally(self, images, image_name, mious, thresholds):
        '''
        function:
            display the images horizontally and save the result
        input:
            images: a list of images, 3 * 4 = 12 images
                    [image, ground truth, sam seg, model seg,
                    fusion_1_result, fusion_2_result, fusion_3_result, fusion_4_result,
                    error_1, error_2, error_3, error_4,
                    confidence_map, entropy_map]
            images_name: the name of the image
            mious: a list of miou and ious,
                    (miou_1, ious_1), (miou_2, ious_2),(miou_3, ious_3), (miou_4, ious_4)
            thresholds: a list of thresholds
                    [confidence_threshold, entropy_threshold]
        '''
        # 获取最大高度和总宽度
        # max_height = max(image.shape[0] for image in images)
        # total_width = sum(image.shape[1] for image in images)
        col = 4
        row = len(images) // col + 1 if len(images) % col != 0 else len(images) // col
        gap = 10  # the gap between two images horizontally
        new_height = self.display_size[0] * row
        new_total_width = (self.display_size[1] + gap) * col
        
        # 显示的文本列表
        texts = ['Image', 'Ground Truth', 'SAM', 'Segmentation']
        for i, (miou, ious) in enumerate(mious):
            # cal the non-zero classes in ious
            unique_classes = np.sum(np.array(ious) != 0)
            mIOU2 = np.sum(np.array(ious)) / unique_classes
            if i > 0:
                texts.append('f_{}, mIoU19: {:.2f} mIoU{}: {:.2f}'.format(i + 2, miou * 100,
                                        unique_classes, mIOU2 * 100))
            else:
                texts.append('f_{}, mIoU19: {:.2f} mIoU{}: {:.2f}'.format(i + 1, miou * 100,
                                        unique_classes, mIOU2 * 100))
        for i in range(len(mious)):
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
    
    def save_ious(self, miou_1, ious_1, miou_2, ious_2, miou_3, ious_3, miou_4, ious_4, image_name):
        miou_diff_2_1 = round((miou_2 - miou_1) * 100, 2)
        miou_diff_3_1 = round((miou_3 - miou_1) * 100, 2)
        miou_diff_4_1 = round((miou_4 - miou_1) * 100, 2) 
        iou_diff_2_1 = [round((ious_2[i] - ious_1[i]) * 100, 2) for i in range(len(ious_1))]
        iou_diff_3_1 = [round((ious_3[i] - ious_1[i]) * 100, 2) for i in range(len(ious_1))]
        iou_diff_4_1 = [round((ious_4[i] - ious_1[i]) * 100, 2) for i in range(len(ious_1))]
        data = pd.DataFrame({
            'class': ['mIoU'] + [name for name in self.label_names],
            'Fusion 1': [round(miou_1 * 100, 2)] + [round(ious_1[i] * 100, 2) for i in range(len(ious_1))],
            'Fusion 2': [round(miou_2 * 100, 2)] + [round(ious_2[i] * 100, 2) for i in range(len(ious_2))],
            'Fusion 3': [round(miou_3 * 100, 2)] + [round(ious_3[i] * 100, 2) for i in range(len(ious_3))],
            'Fusion 4': [round(miou_4 * 100, 2)] + [round(ious_4[i] * 100, 2) for i in range(len(ious_4))],
            'Differ_2_1': [miou_diff_2_1] + iou_diff_2_1,
            'Differ_3_1': [miou_diff_3_1] + iou_diff_3_1,
            'Differ_4_1': [miou_diff_4_1] + iou_diff_4_1,
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
        
        error_mask = np.where((predicted != ground_truth) & (ground_truth != 255), 0, 255).astype(np.uint8)
        # predicted_color = self.color_segmentation(predicted)
        # change the area of error mask in pred_color to white
        pred_color_copy = copy.deepcopy(pred_color)
        pred_color_copy[error_mask == 0] = [255, 255, 255]
        
        # error_mask[pred != gt] = 255
        return pred_color_copy


def main():
    args = get_parse()
    fusion = Fusion(args)
    fusion.fusion()


if __name__ == '__main__':
    main()