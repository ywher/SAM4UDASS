
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
mask_folder = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2'
# the path to the model prediction
# segmentation_root = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs'

# pred folder for gta
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/entropy'

# sepico pred entro and confidence folder for gta 
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629/pred_entropy'

segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_trainid'
confidence_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_confidence'
entropy_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0133_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_8caed/pred_entropy'

# daformer pred folder for synthia
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_trainid_new'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_entropy'
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-basic/230509_1455_gtaHR2csHR_mic_hrda_s2_108c1/pred_trainid'

# the path to the original image
image_folder = '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all'
# the path to the ground truth
gt_folder = '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/gtFine/train_all'

# 
mix_ratio = 0.5
# 
resize_ratio = 1
# 
output_folder = 'outputs/cityscapes/mic_gta_new' #这是去掉了mask按照名称排序的过程
# 
mask_suffix = '.png'
# 
# segmentation_suffix = '_gtFine_labelTrainIds.png'
# segmentation_suffix = '_leftImg8bittrainID.png'
segmentation_suffix = '_leftImg8bittrainID.png'
#
segmentation_suffix_noimg=False
#
confidence_suffix = '_confi.npy'
entropy_suffix = '_entro.npy'
confidence_threshold_mic = 0.9
entropy_ratio_mic = 90  # relative value, lowest 90% entropy
confidence_threshold_daformer = 0.99  # absolute value
entropy_ratio_mic_daformer = 60  # relative value, lowest 60% entropy
#
gt_suffix = '_gtFine_labelTrainIds.png'

# fusion mode = 1
# fusion_mode = 0
fusion_mode = 3
# 
sam_classes = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # 11 classes, 5, 6, 7, 
# 
shrink_num = 2
# 
display_size = (350, 700)
#
road_center_rect = (740, 780, 1645, 995)  # (740, 810, 1625, 995)
# whether to save the mixed result
save_mix_result = False
save_all_fusion = True

#num of classes, 16 for synthia, 19 for gta5
num_classes = 19

# num of images to process
debug_num = 2975 # 2975 - 1648 = 1327
begin_index = 0 # 0