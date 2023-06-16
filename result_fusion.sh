
# for city in augsburg bayreuth duisburg heidelberg konigswinter muhlheim-ruhr saarbrucken wuppertal bad-honnef dortmund erlangen heilbronn konstanz nuremberg schweinfurt wurzburg bamberg dresden freiburg karlsruhe mannheim oberhausen troisdorf
# do
#     python result_fusion.py \
#     --mask_folder '/home/ywh/Downloads/cityscapes/unsup/'$city \
#     --output_folder 'outputs/cityscapes/unsup_fusion/'$city \
#     --segmentation_folder '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/GTA5_deeplab_BiSeNet_20kunsup_adapt_focal_0.8_0.01/output_unsup/pred_trainid/'$city \
#     --image_folder '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/unsup_single/'$city \
#     --mask_suffix '.png' \
#     --mix_ratio 0.5 
# done

#gta
CUDA_VISIBLE_DEVICES=1
python result_fusion.py \
--mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2' \
--output_folder 'outputs/cityscapes/train_fusion_gta_daformer_base2' \
--segmentation_folder '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/trainid' \
--image_folder '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all' \
--mask_suffix '.png' \
--segmentation_suffix 'trainID.png' \
--mix_ratio 0.5 \
--fusion_mode 1 \
--resize_ratio 0.5 \

# --resize_ratio 0.5 \
# --segmentation_suffix_noimg \

# '_gtFine_labelTrainIds.png'

# gta part darmstadt
# CUDA_VISIBLE_DEVICES=1
# python result_fusion.py \
# --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train_darmstadt' \
# --output_folder 'outputs/cityscapes/train_fusion_darmstadt_1' \
# --segmentation_folder '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/230524_2319_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_6283e/pred_trainid_darmstadt' \
# --image_folder '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_darmstadt' \
# --mask_suffix '.png' \
# --segmentation_suffix '_gtFine_labelTrainIds.png' \
# --mix_ratio 0.5 \
# --fusion_mode 1 \
# --resize_ratio 0.25 \
# --segmentation_suffix_noimg \

#synthia
# CUDA_VISIBLE_DEVICES=1
# python result_fusion.py \
# --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2' \
# --output_folder 'outputs/cityscapes/train_fusion_syn_mic_base' \
# --segmentation_folder '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-basic/synHR2csHR_mic_hrda_be9a4/pred_trainid' \
# --image_folder '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all' \
# --mask_suffix '.png' \
# --segmentation_suffix '_trainID.png' \
# --mix_ratio 0.5 \
# --fusion_mode 1 \
# --resize_ratio 0.5 \

#acdc
# CUDA_VISIBLE_DEVICES=0
# python result_fusion.py \
# --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/train2' \
# --output_folder 'outputs/ACDC/train_fusion2' \
# --segmentation_folder '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp8/230527_0645_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f753f/pred_trainid' \
# --image_folder '/media/ywh/1/yanweihao/dataset/acdc/rgb_anon/train' \
# --mask_suffix '.png' \
# --segmentation_suffix 'trainID.png' \
# --mix_ratio 0.5 \
# --fusion_mode 1 \
# --resize_ratio 0.5 \
# --segmentation_suffix_noimg \

#gtFine_labelTrainIds.png
# --segmentation_suffix_noimg \ segmentation suffix without _leftImg8bit, replace '_leftImg8bit' with ''

#_leftImg8bittrainID.png _gtFine_labelTrainIds