
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
mask_folder = '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/sjtu1'

# the path to the model prediction
# daformer
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/entropy'

# sepico
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_entropy'

# mic
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_entropy'

# tufl stage 1
segmentation_folder = '/media/ywh/Elements/projects/UDA/BiSeNet-uda/outputs/tufl_sjtu_focal_10k/pred_trainid/sjtu1'
confidence_folder = '/media/ywh/Elements/projects/UDA/BiSeNet-uda/outputs/tufl_sjtu_focal_10k/pred_confidence/sjtu1'
entropy_folder = '/media/ywh/Elements/projects/UDA/BiSeNet-uda/outputs/tufl_sjtu_focal_10k/pred_entropy/sjtu1'

# the path to the original image
image_folder = '/media/ywh/1/yanweihao/dataset/sjtu/image/train/sjtu1'
# the path to the ground truth
gt_folder = '/media/ywh/1/yanweihao/dataset/sjtu/label/train/sjtu1'
# gt_folder = '/home/cyber-fx/ywh/dataset/sjtu/label/train/sjtu9'

# output folder
output_folder = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/sjtu1_fusion' #这是去掉了mask按照名称排序的过程

# 
mix_ratio = 0.5
# 
resize_ratio = 1

# 
mask_suffix = '.png'
# 
# segmentation_suffix = '_gtFine_labelTrainIds.png'
segmentation_suffix = '.png'
#
segmentation_suffix_noimg=True

# 接在aachen_000000_000019_leftImg8bit之后的后缀
confidence_suffix = '_confi.npy'
entropy_suffix = '_entro.npy'
confidence_threshold_mic = 0.9
entropy_ratio_mic = 90  # relative value, lowest 90% entropy
confidence_threshold_daformer = 0.99  # absolute value
entropy_ratio_daformer = 60  # relative value, lowest 60% entropy
confidence_threshold_tufl = 0.99  # absolute value
entropy_ratio_tufl = 60  # relative value, lowest 60% entropy

#
# gt_suffix = '_gtFine_labelTrainIds.png'
gt_suffix = None

# fusion mode = 1
# fusion_mode = 0
fusion_mode = 3
# 
sam_classes = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # 11 classes, 5, 6, 7, 
# 
shrink_num = 2
# 
display_size = (350, 700)
#sam_alpha
sam_alpha = 0.15
#
road_center_rect = (740, 780, 1645, 995)  # no need
# whether to save the mixed result
save_mix_result = True
save_all_fusion = True

#num of classes, 16 for synthia, 19 for gta5
num_classes = 19

# num of images to process
debug_num = 2399 # 2033 - 249 = 1784
begin_index = 0 # 0