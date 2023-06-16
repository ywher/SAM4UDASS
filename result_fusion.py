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

def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mask_folder', type=str, help='the path to the segment anything result',
                       default='/media/yons/pool1/ywh/projects/Segmentation/segment-anything/outputs/cityscapes/train')
    parse.add_argument('--segmentation_folder', type=str, help='the path to the model prediction',
                       default='/media/yons/pool1/ywh/projects/UDA/MIC/seg/work_dirs/local-exp80/230422_0820_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_3e-05_s0_21197/pred_trainid')
    parse.add_argument('--image_folder', type=str, help='the path to the original image',
                       default='/media/yons/pool1/ywh/dataset/cityscapes/leftImg8bit/train_all')
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
        #the path to the sam mask
        self.mask_folder = args.mask_folder
        #the path to the uda prediction
        self.segmentation_folder = args.segmentation_folder
        #the path to the original image
        self.image_folder = args.image_folder
        #the mix ratio of the fusion result and origianl image
        self.mix_ratio = args.mix_ratio
        #the resize ratio of the mix image
        self.resize_ratio = args.resize_ratio
        #the path to the output folder
        self.output_folder = args.output_folder
        #the image suffix of the mask and segmentation result
        self.mask_suffix = args.mask_suffix
        self.segmentation_suffix = args.segmentation_suffix
        self.segmentation_suffix_noimg = args.segmentation_suffix_noimg
        #the fusion mode
        self.fusion_mode = args.fusion_mode
        #the classes sam performs better
        self.sam_classes = args.sam_classes
        #the shrink num of segmentation mask
        self.shrink_num = args.shrink_num
        
        self.image_names = os.listdir(self.mask_folder) #one folder corresponds to one image name without suffix
        self.image_names.sort()
        
        #make the folder to save the fusion result
        self.check_and_make(os.path.join(self.output_folder, 'trainID')) #the fusion result in trainID
        # self.check_and_make(os.path.join(self.output_folder, 'color')) #the fusion result in color
        self.check_and_make(os.path.join(self.output_folder, 'mixed')) #the fusion result in color mixed with original image
        #make the folder to save the fusion result with segmenation result as the background
        self.check_and_make(os.path.join(self.output_folder, 'trainID_bg')) #the fusion result in trainID with segmenation result as the background
        # self.check_and_make(os.path.join(self.output_folder, 'color_bg')) #the fusion result in color with segmenation result as the background
        self.check_and_make(os.path.join(self.output_folder, 'mixed_bg')) #the fusion result in color mixed with original image with segmenation result as the background
        
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

class Fusion2():
    def __init__(self, mask_folder, segmentation_folder, image_folder, mix_ratio,
                 resize_ratio, output_folder, mask_suffix, segmentation_suffix, segmentation_suffix_noimg,
                 fusion_mode, sam_classes, shrink_num):
        #the path to the sam mask
        self.mask_folder = mask_folder
        #the path to the uda prediction
        self.segmentation_folder = segmentation_folder
        #the path to the original image
        self.image_folder = image_folder
        #the mix ratio of the fusion result and origianl image
        self.mix_ratio = mix_ratio
        #the resize ratio of the mix image
        self.resize_ratio = resize_ratio
        #the path to the output folder
        self.output_folder = output_folder
        #the image suffix of the mask and segmentation result
        self.mask_suffix = mask_suffix
        self.segmentation_suffix = segmentation_suffix
        self.segmentation_suffix_noimg = segmentation_suffix_noimg
        #the fusion mode
        self.fusion_mode = fusion_mode
        #the classes sam performs better
        self.sam_classes = sam_classes
        #the shrink num of segmentation mask
        self.shrink_num = shrink_num
        
        self.image_names = os.listdir(self.mask_folder) #one folder corresponds to one image name without suffix
        self.image_names.sort()
        
        #make the folder to save the fusion result
        self.check_and_make(os.path.join(self.output_folder, 'trainID')) #the fusion result in trainID
        # self.check_and_make(os.path.join(self.output_folder, 'color')) #the fusion result in color
        self.check_and_make(os.path.join(self.output_folder, 'mixed')) #the fusion result in color mixed with original image
        #make the folder to save the fusion result with segmenation result as the background
        self.check_and_make(os.path.join(self.output_folder, 'trainID_bg')) #the fusion result in trainID with segmenation result as the background
        # self.check_and_make(os.path.join(self.output_folder, 'color_bg')) #the fusion result in color with segmenation result as the background
        self.check_and_make(os.path.join(self.output_folder, 'mixed_bg')) #the fusion result in color mixed with original image with segmenation result as the background
        
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



def main():
    args = get_parse()
    fusion = Fusion(args)
    fusion.fusion()

if __name__ == '__main__':
    main()