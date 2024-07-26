# import the needed packages
# from result_fusion import Fusion
from fusion.fusion_syn import Fusion_SYN
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tools.iou_perimg import SegmentationMetrics
import tqdm
from config import config_syn as config

if __name__ == '__main__':
    # folder pathes
    mask_folder = config.mask_folder
    mask_suffix = config.mask_suffix
    # the path to the model prediction
    segmentation_folder = config.segmentation_folder
    segmentation_suffix = config.segmentation_suffix
    segmentation_suffix_noimg = config.segmentation_suffix_noimg
    # 
    confidence_folder = config.confidence_folder
    confidence_suffix = config.confidence_suffix
    entropy_folder = config.entropy_folder
    entropy_suffix = config.entropy_suffix

    # the path to the original image
    image_folder = config.image_folder
    image_suffix = config.image_suffix
    # the path to the ground truth
    gt_folder = config.gt_folder
    gt_suffix = config.gt_suffix
    # the path to the output folder
    output_folder = config.output_folder

    # num of classes
    num_classes = config.num_classes

    ### fusion parameters
    # the fusion mode
    fusion_mode = config.fusion_mode
    road_assumption = config.road_assumption
    road_center_rect = config.road_center_rect
    use_sam = config.use_sam
    sam_model_type = config.sam_model_type
    sam_model_path = config.sam_model_path
    device = config.device
    get_sam_mode= config.get_sam_mode
    use_sgml = config.use_sgml
    sam_alpha = config.sam_alpha
    large_classes = config.large_classes  # [0, 1, 2, 8, 10, 13]
    small_classes = config.small_classes  # [3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18]
    sam_classes = config.sam_classes  # 11 classes, 5, 6, 7, 
    shrink_num = config.shrink_num
    
    if 'BiSeNet-uda' in config.segmentation_folder:
        confidence_threshold = config.confidence_threshold_tufl
        entropy_ratio = config.entropy_ratio_tufl
    elif 'HRDA' in config.segmentation_folder:
        confidence_threshold = config.confidence_threshold_hrda
        entropy_ratio = config.entropy_ratio_hrda
    elif 'MIC' in config.segmentation_folder:
        confidence_threshold = config.confidence_threshold_mic
        entropy_ratio = config.entropy_ratio_mic
    else:
        confidence_threshold = config.confidence_threshold_daformer
        entropy_ratio = config.entropy_ratio_daformer

    ### display parameters
    display_size = config.display_size
    mix_ratio = config.mix_ratio
    resize_ratio = config.resize_ratio

    ### save params
    save_mix_result = config.save_mix_result
    save_sam_result = config.save_sam_result
    save_all_fusion = config.save_all_fusion

    save_majority_process = config.save_majority_process
    save_sgml_process = config.save_sgml_process
    save_f1_process = config.save_f1_process
    save_f2_process = config.save_f2_process
    save_f3_process = config.save_f3_process

    ### time setting
    time_process = config.time_process
    time_filename = config.time_filename

    # 
    debug_num = config.debug_num # 2975
    begin_index = config.begin_index # 0

    fusion = Fusion_SYN(mask_folder,
                        mask_suffix,
                        segmentation_folder,
                        segmentation_suffix,
                        segmentation_suffix_noimg,
                        confidence_folder,
                        confidence_suffix,
                        entropy_folder,
                        entropy_suffix,
                        image_folder, 
                        image_suffix,
                        gt_folder, 
                        gt_suffix,
                        output_folder,
                        num_classes, 
                        fusion_mode,
                        road_assumption,
                        road_center_rect,
                        use_sam,
                        sam_model_type,
                        sam_model_path,
                        device,
                        get_sam_mode,
                        sam_alpha,
                        large_classes,
                        small_classes,
                        sam_classes,
                        shrink_num,
                        display_size,
                        mix_ratio, 
                        resize_ratio,
                        time_process,
                        time_filename,
                        save_sgml_process,
                        save_majority_process,
                        save_f1_process, 
                        save_f2_process, 
                        save_f3_process)
    
    index_range = list(range(begin_index, begin_index + debug_num))
    iou_cal = SegmentationMetrics(num_classes=num_classes)

    if save_all_fusion:
        f1_output_folder = os.path.join(output_folder, 'fusion1_trainid')
        f1_color_output_folder = os.path.join(output_folder, 'fusion1_color')
        f2_output_folder = os.path.join(output_folder, 'fusion2_trainid')
        f2_color_output_folder = os.path.join(output_folder, 'fusion2_color')
        f3_output_folder = os.path.join(output_folder, 'fusion3_trainid')
        f3_color_output_folder = os.path.join(output_folder, 'fusion3_color')
        # f4_output_folder = os.path.join(output_folder, 'fusion4_trainid')
        # f5_output_folder = os.path.join(output_folder, 'fusion5_trainid')
        fusion.check_and_make(f1_output_folder)
        fusion.check_and_make(f1_color_output_folder)
        fusion.check_and_make(f2_output_folder)
        fusion.check_and_make(f2_color_output_folder)
        fusion.check_and_make(f3_output_folder)
        fusion.check_and_make(f3_color_output_folder)
        # fusion.check_and_make(f4_output_folder)
        # fusion.check_and_make(f5_output_folder)

    if save_sam_result:
        sam_majority_output_folder = os.path.join(output_folder, 'sam_majority_trainid')
        sam_majority_color_output_folder = os.path.join(output_folder, 'sam_majority_color')
        sam_sgml_output_folder = os.path.join(output_folder, 'sam_sgml_trainid')
        sam_sgml_color_output_folder = os.path.join(output_folder, 'sam_sgml_color')
        fusion.check_and_make(sam_majority_output_folder)
        fusion.check_and_make(sam_majority_color_output_folder)
        fusion.check_and_make(sam_sgml_output_folder)
        fusion.check_and_make(sam_sgml_color_output_folder)
        

    bar = tqdm.tqdm(total=debug_num)
    for i in index_range:
        image_name = fusion.image_names[i]  # aachen_000000_000019_leftImg8bit
        image_name = image_name.replace('_leftImg8bit', '') # aachen_000000_000019
        
        # get the prediction
        prediction_path = os.path.join(fusion.segmentation_folder, image_name + fusion.segmentation_suffix)
        if fusion.segmentation_suffix_noimg:
            prediction_path = prediction_path.replace('_leftImg8bit', '')
        uda_pred = cv2.imread(prediction_path, 0) #[h, w]
        uda_pred_color = fusion.color_segmentation(uda_pred)
        
        # get the confidence map
        confidence_path = os.path.join(fusion.confidence_folder, image_name + fusion.confidence_suffix)
        pred_confidence = np.load(confidence_path, allow_pickle=True) #[h, w]
        
        # get the entropy map
        entropy_path = os.path.join(fusion.entropy_folder, image_name + fusion.entropy_suffix)
        pred_entropy = np.load(entropy_path, allow_pickle=True) #[h, w]
        
        # get the ground truth
        gt_path = os.path.join(fusion.gt_folder, image_name + fusion.gt_suffix)
        gt = cv2.imread(gt_path, 0) # [h, w, 3]
        gt_color = fusion.color_segmentation(gt)

        # get the original image
        original_image = cv2.imread(os.path.join(fusion.image_folder, image_name + fusion.image_suffix))

        # get the confidence and entropy map
        pred_confidence = pred_confidence.astype(np.float32)
        confidence_map = fusion.visualize_numpy(pred_confidence)
        confidence_mask, confidence_img = fusion.vis_np_higher_thres(pred_confidence, original_image, confidence_threshold)
        
        pred_entropy = pred_entropy.astype(np.float32)
        entropy_map = fusion.visualize_numpy(pred_entropy)
        entropy_threshold = np.percentile(pred_entropy, entropy_ratio)
        entropy_mask, entropy_img = fusion.vis_np_lower_thres(pred_entropy, original_image, entropy_threshold)

        #get the sam segmentation result using the mask
        sam_pred_sgml, sam_pred_majority = fusion.get_sam_pred(image_name, uda_pred, confidence_mask, entropy_mask)  # [h,w]
        sam_color_sgml = fusion.color_segmentation(sam_pred_sgml)  # [h,w,3]
        
        if use_sgml:
            sam_pred = sam_pred_sgml
        else:
            sam_pred = sam_pred_majority
            
        # save the sam result
        image_filename = image_name + fusion.mask_suffix
        if save_sam_result:
            sam_majority_color = fusion.color_segmentation(sam_pred_majority) # [h,w,3]
            cv2.imwrite(os.path.join(sam_majority_output_folder, image_filename), sam_pred_majority)
            cv2.imwrite(os.path.join(sam_majority_color_output_folder, image_filename), sam_majority_color)
            cv2.imwrite(os.path.join(sam_sgml_output_folder, image_filename), sam_pred_sgml)
            cv2.imwrite(os.path.join(sam_sgml_color_output_folder, image_filename), sam_color_sgml)

        # initialize the fusion color results and error images with black image
        black_img = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
        fusion_color_bg_1, fusion_color_bg_2, fusion_color_bg_3, fusion_color_bg_4, fusion_color_bg_5 = black_img, black_img, black_img, black_img, black_img
        error_1, error_2, error_3, error_4, error_5 = black_img, black_img, black_img, black_img, black_img
        
        # initialize fusion trainid with 255 one-channel image
        ignore_lb = np.ones((original_image.shape[0], original_image.shape[1]), dtype=np.uint8) * 255
        fusion_trainid_bg_1, fusion_trainid_bg_2, fusion_trainid_bg_3, fusion_trainid_bg_4, fusion_trainid_bg_5 = ignore_lb, ignore_lb, ignore_lb, ignore_lb, ignore_lb

        # get fusion result from 1, 2, 3, 4, 5
        fusion_trainid_bg_1, fusion_color_bg_1 = \
            fusion.fusion_mode_1(uda_pred=uda_pred, sam_pred=sam_pred)
        fusion_trainid_bg_2, fusion_color_bg_2 = \
            fusion.fusion_mode_2(uda_pred=uda_pred, sam_pred=sam_pred)
        fusion_trainid_bg_3, fusion_color_bg_3 = \
            fusion.fusion_mode_3(uda_pred=uda_pred, sam_pred=sam_pred, fusion_trainid=fusion_trainid_bg_1,
                                confidence_mask=confidence_mask, entropy_mask=entropy_mask, image_name=image_name)
        # fusion_trainid_bg_4, fusion_color_bg_4 = \
        #     fusion.fusion_mode_4(uda_pred=uda_pred, sam_pred=sam_pred, \
        #                          fusion_trainid=fusion_trainid_bg_3, confidence_mask=confidence_mask)
        # fusion_trainid_bg_5, fusion_color_bg_5 = \
        #     fusion.fusion_mode_5(uda_pred=uda_pred, sam_pred=sam_pred, \
            
        miou_0, miou_1, miou_2, miou_3, miou_4, miou_5 = -1, -1, -1, -1, -1, -1
        ious_0, ious_1, ious_2, ious_3, ious_4, ious_5 = [], [], [], [], [], []
        
        miou_0, ious_0 = iou_cal.calculate_miou(uda_pred, gt)
        miou_1, ious_1 = iou_cal.calculate_miou(fusion_trainid_bg_1, gt)
        miou_2, ious_2 = iou_cal.calculate_miou(fusion_trainid_bg_2, gt)
        miou_3, ious_3 = iou_cal.calculate_miou(fusion_trainid_bg_3, gt)
        # miou_4, ious_4 = iou_cal.calculate_miou(fusion_trainid_bg_4, gt)
        # miou_5, ious_5 = iou_cal.calculate_miou(fusion_trainid_bg_5, gt)
        
        error_0 = fusion.get_error_image(uda_pred, gt, uda_pred_color)
        error_1 = fusion.get_error_image(fusion_trainid_bg_1, gt, fusion_color_bg_1)
        error_2 = fusion.get_error_image(fusion_trainid_bg_2, gt, fusion_color_bg_2)
        error_3 = fusion.get_error_image(fusion_trainid_bg_3, gt, fusion_color_bg_3)
        # error_4 = fusion.get_error_image(fusion_trainid_bg_4, gt, fusion_color_bg_4)
        # error_5 = fusion.get_error_image(fusion_trainid_bg_5, gt, fusion_color_bg_5)
        
        # fusion.dis_imgs_horizontal([gt_color, sam_color_sgml, uda_pred_color, fusion_color_bg_0], '{}_fusion0'.format(image_name.replace('_leftImg8bit', '')), miou_0)
        fusion.dis_imgs_horizontal(
            [original_image, gt_color, sam_color_sgml, uda_pred_color, error_0, \
            fusion_color_bg_1, fusion_color_bg_2, fusion_color_bg_3, fusion_color_bg_4, fusion_color_bg_5, \
            error_1, error_2, error_3, error_4, error_5, \
            confidence_map, entropy_map, confidence_img, entropy_img], \
            '{}'.format(image_name), \
            [(miou_0, ious_0), (miou_1, ious_1), (miou_2, ious_2), \
            (miou_3, ious_3), (miou_4, ious_4), (miou_5, ious_5)], \
            [confidence_threshold, entropy_threshold])
        
        # save the mious and ious
        miou_values = [miou_0, miou_1, miou_2, miou_3, miou_4, miou_5]
        ious_values = [ious_0, ious_1, ious_2, ious_3, ious_4, ious_5]
        fusion.save_ious(miou_values, ious_values, '{}'.format(image_name))

        # save all fusion results
        if save_all_fusion:
            cv2.imwrite(os.path.join(f1_output_folder, image_filename), fusion_trainid_bg_1)
            cv2.imwrite(os.path.join(f1_color_output_folder, image_filename), fusion_color_bg_1)
            cv2.imwrite(os.path.join(f2_output_folder, image_filename), fusion_trainid_bg_2)
            cv2.imwrite(os.path.join(f2_color_output_folder, image_filename), fusion_color_bg_2)
            cv2.imwrite(os.path.join(f3_output_folder, image_filename), fusion_trainid_bg_3)
            cv2.imwrite(os.path.join(f3_color_output_folder, image_filename), fusion_color_bg_3)
            # cv2.imwrite(os.path.join(f4_output_folder, image_filename), fusion_trainid_bg_4)
            # cv2.imwrite(os.path.join(f5_output_folder, image_filename), fusion_trainid_bg_5)
            
        # save mix results
        if save_mix_result:
            # get the sam mixed color image using the fusion.mix_ratio
            sam_mixed_color = cv2.addWeighted(original_image, fusion.mix_ratio, sam_color_sgml, 1 - fusion.mix_ratio, 0)
            if fusion.resize_ratio != 1:
                new_h = int(sam_mixed_color.shape[0] * fusion.resize_ratio)
                new_w = int(sam_mixed_color.shape[1] * fusion.resize_ratio)
                sam_mixed_color = cv2.resize(sam_mixed_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # save the sam mask in trainid and color to the output folder
            cv2.imwrite(os.path.join(fusion.output_folder, 'trainID', image_filename), sam_pred)
            # cv2.imwrite(os.path.join(fusion.output_folder, 'color', image_filename), fusion_color)
            cv2.imwrite(os.path.join(fusion.output_folder, 'mixed', image_filename), sam_mixed_color)
            
            # make the fusion results to list for easy use    
            fusion_trainid_bgs = [fusion_trainid_bg_1, fusion_trainid_bg_2, fusion_trainid_bg_3, fusion_trainid_bg_4, fusion_trainid_bg_5]
            fusion_color_bgs = [fusion_color_bg_1, fusion_color_bg_2, fusion_color_bg_3, fusion_color_bg_4, fusion_color_bg_5]
            
            mode = fusion.fusion_mode - 1
            if mode in range(len(fusion_trainid_bgs)):
                fusion_trainid_bg, fusion_color_bg = fusion_trainid_bgs[mode], fusion_color_bgs[mode]
            else:
                raise NotImplementedError("This fusion mode has not been implemented yet.")
        
            #save the fusion mask in trainid and color to the output folder
            mixed_color_bg = cv2.addWeighted(original_image, fusion.mix_ratio, fusion_color_bg, 1 - fusion.mix_ratio, 0)
            if fusion.resize_ratio != 1:
                mixed_color_bg = cv2.resize(mixed_color_bg, (int(mixed_color_bg.shape[1] * fusion.resize_ratio), int(mixed_color_bg.shape[0] * fusion.resize_ratio)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(fusion.output_folder, 'trainID_bg', image_filename), fusion_trainid_bg)
            # cv2.imwrite(os.path.join(fusion.output_folder, 'color_bg', image_filename), fusion_color_bg)
            cv2.imwrite(os.path.join(fusion.output_folder, 'mixed_bg', image_filename), mixed_color_bg)

        bar.update(1)
    bar.close()
    if time_process:
        fusion.show_time_process()