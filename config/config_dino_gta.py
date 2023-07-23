
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
dino_folder = '/media/ywh/1/XAC_Learning/projects/Grounded-Segment-Anything/Gray_outputs_train_all'

# the path to the model prediction
# daformer
segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/pred_trainid'
confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/pred_confidence'
entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/pred_entropy'

# sepico
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_entropy'

# mic
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230716_1343_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_ea911/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230716_1343_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_ea911/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230716_1343_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_ea911/pred_entropy'

# tufl stage 1
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/GTA5_deeplab_BiSeNet_20kunsup_adapt_focal_0.8_0.01/output_train/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/GTA5_deeplab_BiSeNet_20kunsup_adapt_focal_0.8_0.01/output_train/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/GTA5_deeplab_BiSeNet_20kunsup_adapt_focal_0.8_0.01/output_train/pred_entropy'

# the path to the original image
image_folder = '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all'

# the path to the ground truth
gt_folder = '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/gtFine/train_all'

#num of classes, 16 for synthia, 19 for gta5
num_classes = 19

# 
mix_ratio = 0.5
# 
resize_ratio = 1

# output folder
output_folder = 'outputs/cityscapes/train_fusion_gta_dino4' #这是去掉了mask按照名称排序的过程

# 
dino_suffix = '.png'
# 
# segmentation_suffix = '_gtFine_labelTrainIds.png'
segmentation_suffix = '_leftImg8bittrainID.png'
# segmentation_suffix = '_leftImg8bit.png'
#
segmentation_suffix_noimg=False

# 接在aachen_000000_000019_leftImg8bit之后的后缀
confidence_suffix = '_confi.npy'
entropy_suffix = '_entro.npy'
gt_suffix = '_gtFine_labelTrainIds.png'
confidence_threshold_mic = 0.9
entropy_ratio_mic = 90  # relative value, lowest 90% entropy
confidence_threshold_daformer = 0.85  # absolute value
entropy_ratio_daformer = 60  # relative value, lowest 60% entropy
confidence_threshold_tufl = 0.9  # absolute value
entropy_ratio_tufl = 80  # relative value, lowest 60% entropy

# fusion mode = 1
# fusion_mode = 0
fusion_mode = 3
# 
dino_classes = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # 11 classes, 5, 6, 7, 
# 
# 
display_size = (350, 700)
#
# whether to save the mixed result
save_mix_result = False
save_all_fusion = True

# num of images to process
debug_num = 2975 # 2975 - 1087 = 1888
begin_index = 0 # 804 + 283 = 1087