import os
import cv2
import pandas as pd
import numpy as np
from config import config_sam_syn as config
from cityscapesscripts.helpers.labels import trainId2label as trainid2label
from segment_anything import sam_model_registry, SamPredictor  # SamAutomaticMaskGenerator, SamPredictor
import tqdm
import copy
from natsort import natsorted
from tools.iou_perimg import SegmentationMetrics
from utils.segmentix import Segmentix


class SAM_FUSION():
    def __init__(self, mask_folder=None, segmentation_folder=None, confidence_folder=None, entropy_folder=None,
                 image_folder=None, gt_folder=None, num_classes=None, road_center_rect=None,
                 mix_ratio=None, resize_ratio=None, output_folder=None, mask_suffix=None,
                 segmentation_suffix=None, segmentation_suffix_noimg=None,
                 confidence_suffix=None, entropy_suffix=None, gt_suffix=None,
                 fusion_mode=None, sam_classes=None, shrink_num=None, display_size=(200, 400),
                 sam_model_path = "./models/sam_vit_h_4b8939.pth"):
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

        model_type = "vit_h" #vit_b, vit_h, vit_l
        device = "cuda:0"

        sam = sam_model_registry[model_type](checkpoint=sam_model_path)
        sam.to(device=device)
        print('load model successfully')

        #set predictor
        self.predictor = SamPredictor(sam)
        self.segmtrix = Segmentix()

        self.iou_cal = SegmentationMetrics(num_classes=num_classes)

    def check_and_make(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('the path is already exist')

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

    def mask_merge_by_stability(self, prompt_masks, scores):
        """
        func: merge the masks by the stability, one pixel may belong to many masks
        prompt_masks : dict{mask_id : np.array}
        score : dict{mask_id : float}
        """
        merged_mask = (np.zeros_like(list(prompt_masks.values())[0]) + 1000)
        score_map = np.zeros_like(merged_mask, dtype=float)

        for mask_id in prompt_masks:
            higher_score_region = (prompt_masks[mask_id] > 0) & (score_map < scores[mask_id])
            merged_mask[higher_score_region] = mask_id
            score_map[higher_score_region] = scores[mask_id]
            # print(f"mask id is {class_names[mask_id]}, mask score is {scores[mask_id]}")

        return merged_mask, score_map

    def mask_merge(self, prompt_masks, scores, possibilities):
        """ prompt_masks : dict{mask_id : np.array}
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
        merged_mask[merged_mask == 255] = blank_region[merged_mask == 255]
        return merged_mask
    
    def get_prompt_point(self, bin_mask_image):
        # 给uda中每个类别的mask, 对每个联通区域计算它们的单个质心作为点的prompt
        # 进行连通区域提取
        connectivity = 8  # 连通性，4代表4连通，8代表8连通
        output = cv2.connectedComponentsWithStats(bin_mask_image, connectivity, cv2.CV_32S)

        # 获取连通区域的数量
        num_labels = output[0]

        # 获取连通区域的属性
        labels = output[1]
        stats = output[2]

        cps = []
        
        # 循环遍历每个连通区域
        for i in range(1, num_labels):
            # 获取连通区域的左上角坐标和宽高
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
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

    def generate_segementation_by_sam(self, image, autogenerator_mask, uda_mask):
        """
        func: generate pseudo label for sam by referring to uda_mask
            image : np.array
            autogenrator_mask : np.array, uint8
            uda_mask : np.array, uint8
        """ 

        # Load image to extract image embedding
        self.predictor.set_image(image)
        sam_mask_result = np.zeros_like(autogenerator_mask).astype(np.uint8) + 255

        # Give auto generated mask corresponding semantic label
        # can update to the get_sam_pred function in result_fusion.py
        for submask_id in np.unique(autogenerator_mask):
            if submask_id == 1000:
                break
            submask = autogenerator_mask == submask_id
            max_iou = 0
            final_mask_id = 0
            for semantic_id in np.unique(uda_mask):
                iou = np.sum(submask & (uda_mask == semantic_id))
                if max_iou < iou:
                    max_iou = iou
                    final_mask_id = semantic_id 
            sam_mask_result[submask] = final_mask_id

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

            mask, score, logit = self.predictor.predict(
                point_coords=np.array(pos + neg),
                point_labels=np.array([1]*len(pos) + [0]*len(neg)),
                mask_input = mask_point_prompts[semantic_id]["mask"],
                multimask_output=False,
                )

            score_result[semantic_id] = score[0]  # stability score, only one value, like confidence score
            mask_result[semantic_id] = mask[0]  # 
            # possibility_result shape is [H, W]
            possibility_result[semantic_id] = self.segmtrix.turn_logits_to_possibility(logit, (image.shape[1], image.shape[0]))

        merged_mask = self.mask_merge(mask_result, score_result, possibility_result)
        return merged_mask, sam_mask_result

    def color_segmentation(self, segmentation):
        #get the color segmentation result, initial the color segmentation result with black (0,0,0)
        #input: segmentation [h, w]
        color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        train_ids = np.unique(segmentation).astype(np.uint8)
        for train_id in train_ids:
            color_segmentation[segmentation == train_id] = self.trainid2color(train_id)
        return color_segmentation
    
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
            
            if len(image.shape) == 2:
                image = image[:,:,None]
            # first row
            if i < col:
                try:
                    output_image[0*image.shape[0]:1*image.shape[0], current_width:current_width+image.shape[1], :] = image
                except:
                    import pdb; pdb.set_trace()
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

    def fusion(self):
        bar = tqdm.tqdm(total=len(self.image_names))
        for image_name in self.image_names:
            # get the segmentation result
            if self.segmentation_suffix_noimg:
                prediction_path = os.path.join(self.segmentation_folder, image_name.replace('_leftImg8bit', '') + self.segmentation_suffix)
            else:
                prediction_path = os.path.join(self.segmentation_folder, image_name + self.segmentation_suffix)
            # import pdb; pdb.set_trace()
            # print('load from: ', prediction_path)
            uda_mask = cv2.imread(prediction_path, 0)  # [h, w, 3], 3 channels not 1 channel
            # import pdb; pdb.set_trace()
            uda_color = self.color_segmentation(uda_mask)

            # get the ground truth
            gt_path = os.path.join(self.gt_folder, \
                image_name.replace('_leftImg8bit', '') + self.gt_suffix)
            gt = cv2.imread(gt_path, 0)  # [h, w, 3]
            gt_color = self.color_segmentation(gt)
            # print(np.unique(gt))

            # get the original image
            original_image = cv2.imread(os.path.join(self.image_folder, image_name + self.mask_suffix))

            # get the mask names
            mask_names = [name for name in os.listdir(os.path.join(self.mask_folder, image_name)) if self.mask_suffix in name]
            mask_names = natsorted(mask_names)  # order the mask names according to the id, from big to small

            autogenerator_masks = {}
            masks_stability_score = {}
            meta_data = pd.read_csv(os.path.join(self.mask_folder, image_name,"metadata.csv"))  

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
            fused_result, auto_mask  = self.generate_segementation_by_sam(original_image, merged_auto_mask, uda_mask)
            sam_color = self.color_segmentation(auto_mask)
            fusion_color_bg = self.color_segmentation(fused_result)

            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            if self.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * self.resize_ratio), int(mixed_color_bg.shape[0] * self.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.output_folder, 'trainID_bg', image_name + self.mask_suffix), fused_result)
            # cv2.imwrite(os.path.join(self.output_folder, 'color_bg', image_name + self.mask_suffix), fusion_color_bg)
            cv2.imwrite(os.path.join(self.output_folder, 'mixed_bg', image_name + self.mask_suffix), mixed_color_bg)
            # fusion_color_bg = cv2.addWeighted(original_image, self.mix_ratio, fusion_color_bg, 1 - self.mix_ratio, 0)
            
            miou_0, ious_0 = self.iou_cal.calculate_miou(uda_mask, gt)
            miou_1, ious_1 = self.iou_cal.calculate_miou(fused_result, gt)

            error_0 = self.get_error_image(uda_mask, gt, uda_color)
            error_1 = self.get_error_image(fused_result, gt, fusion_color_bg)

            # fusion.dis_imgs_horizontal([gt_color, sam_color, pred_color, fusion_color_bg_0], '{}_fusion0'.format(image_name.replace('_leftImg8bit', '')), miou_0)
            self.dis_imgs_horizontal(
                [original_image, gt_color, sam_color, uda_color, error_0, \
                fusion_color_bg, fusion_color_bg, fusion_color_bg, fusion_color_bg, fusion_color_bg, \
                error_1, error_1, error_1, error_1, error_1, \
                score_map, score_map, score_map, score_map], \
                '{}'.format(image_name.replace('_leftImg8bit', '')), \
                [(miou_0, ious_0), (miou_1, ious_1), (miou_1, ious_1), \
                (miou_1, ious_1), (miou_1, ious_1), (miou_1, ious_1)], \
                [np.max(score_map), np.max(score_map)])
            self.save_ious(miou_0, ious_0, miou_1, ious_1, miou_1, ious_1, miou_1, ious_1, miou_1, ious_1, \
                            miou_1, ious_1, '{}'.format(image_name.replace('_leftImg8bit', '')))

            bar.update(1)

if __name__ == "__main__":

    # define the folder path and parameters
    # train2 is the folder of the generated mask, preciser
    # train is the folder of the generated mask, default sam params
    mask_folder = config.mask_folder
    # the path to the model prediction
    # segmentation_root = '/media/ywh/pool1/yanweihao/projects/uda/DAFormer/work_dirs'
    segmentation_folder = config.segmentation_folder
    confidence_folder = config.confidence_folder
    entropy_folder = config.entropy_folder
    # segmentation_folder = '/media/ywh/pool1/yanweihao/projects/uda/MIC/seg/work_dirs/local-basic/230509_1455_gtaHR2csHR_mic_hrda_s2_108c1/pred_trainid'
    # the path to the original image
    image_folder = config.image_folder
    # the path to the ground truth
    gt_folder = config.gt_folder

    # 
    mix_ratio = config.mix_ratio
    # 
    resize_ratio = config.resize_ratio
    # 
    output_folder = config.output_folder #这是去掉了mask按照名称排序的过程
    # 
    mask_suffix = config.mask_suffix
    # 
    # segmentation_suffix = '_gtFine_labelTrainIds.png'
    # segmentation_suffix = '_leftImg8bittrainID.png'
    segmentation_suffix = config.segmentation_suffix
    #
    segmentation_suffix_noimg = config.segmentation_suffix_noimg
    #
    confidence_suffix = config.confidence_suffix
    entropy_suffix = config.entropy_suffix
    confidence_threshold = config.confidence_threshold  # absolute value
    entropy_ratio = config.entropy_ratio  # relative value, lowest 70% entropy
    #
    gt_suffix = config.gt_suffix

    # fusion mode = 1
    # fusion_mode = 0
    fusion_mode = config.fusion_mode
    # 
    sam_classes = config.sam_classes  # 11 classes, 5, 6, 7, 
    # 
    shrink_num = config.shrink_num
    # 
    display_size = config.display_size
    #
    road_center_rect = config.road_center_rect
    # whether to save the mixed result
    save_mix_result = config.save_mix_result
    save_all_fusion = config.save_all_fusion
    #num of classes
    num_classes = config.num_classes
    #
    debug_num = config.debug_num # 2975
    begin_index = config.begin_index # 0
    fuse = SAM_FUSION(mask_folder, segmentation_folder, confidence_folder, entropy_folder, \
                image_folder, gt_folder, num_classes, road_center_rect, \
                mix_ratio, resize_ratio, output_folder, mask_suffix, \
                segmentation_suffix, segmentation_suffix_noimg, \
                confidence_suffix, entropy_suffix, gt_suffix, \
                fusion_mode, sam_classes, shrink_num, display_size)
    
    fuse.fusion()