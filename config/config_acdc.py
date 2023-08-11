
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
mask_folder = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/train2'
# the path to the model prediction
# segmentation_root = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs'

# pred folder for gta daformer
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp8/230527_0645_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f753f/trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp8/230527_0645_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f753f/confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp8/230527_0645_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f753f/entropy'

# pred folder for mic gta
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0405_csHR2acdcHR_1024x1024_dacs_a999_fdthings_rcs001-20_m64-07-sep_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_0b9fc/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0405_csHR2acdcHR_1024x1024_dacs_a999_fdthings_rcs001-20_m64-07-sep_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_0b9fc/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230713_0405_csHR2acdcHR_1024x1024_dacs_a999_fdthings_rcs001-20_m64-07-sep_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_0b9fc/pred_entropy'

# pred folder for tufl acdc
segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/acdc/CityScapes_ACDC_BiSeNet_20kunsup_focal_0.8_0.05_paste_mode_Single_1/pred_train/pred_trainid'
confidence_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/acdc/CityScapes_ACDC_BiSeNet_20kunsup_focal_0.8_0.05_paste_mode_Single_1/pred_train/pred_confidence'
entropy_folder = '/media/ywh/1/yanweihao/projects/uda/BiSeNet-uda/outputs/acdc/CityScapes_ACDC_BiSeNet_20kunsup_focal_0.8_0.05_paste_mode_Single_1/pred_train/pred_entropy'

#pred folder for synthia
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_trainid_new'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_entropy'
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-basic/230509_1455_gtaHR2csHR_mic_hrda_s2_108c1/pred_trainid'

# the path to the original image
image_folder = '/media/ywh/1/yanweihao/dataset/acdc/rgb_anon/train'
# the path to the ground truth
gt_folder = '/media/ywh/1/yanweihao/dataset/acdc/gt/train'

#
mix_ratio = 0.5
#
resize_ratio = 0.5
#
output_folder = 'outputs/ACDC/tufl_acdc' # 这是去掉了mask按照名称排序的过程
#
mask_suffix = '.png'
#
# segmentation_suffix = '_gtFine_labelTrainIds.png'
# segmentation_suffix = '_leftImg8bittrainID.png'
# segmentation_suffix = 'trainID.png'  # mic
segmentation_suffix = '.png'  # tufl GOPR0122_frame_000161_rgb_anon + 
#
segmentation_suffix_noimg=True
#
confidence_suffix = '_confi.npy'
entropy_suffix = '_entro.npy'
confidence_threshold = 0.99  # absolute value 0.99
entropy_ratio = 80  # relative value, lowest 70% entropy 60
#
gt_suffix = '_gt_labelTrainIds.png'

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
sky_center_rect = (600, 110, 1580, 440)
road_center_rect = (740, 780, 1645, 995)  # (740, 810, 1625, 995)
# 
daytime_threshold = 70
# whether to save the mixed result
save_mix_result = False
save_all_fusion = True

# num of classes, 16 for synthia, 19 for gta5
num_classes = 19

# num of images to process
debug_num = 1600  # 2975 - 1648 = 1327
begin_index = 0  # 0