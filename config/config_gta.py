
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
mask_root = '/media/ywh/pool1/yanweihao/projects/segmentation/segment-anything'
mask_folder = f'{mask_root}/outputs/cityscapes/train_vith_2'
mask_folder_suffix = '_leftImg8bit'
mask_suffix = '.png'

# the path to the model prediction
# daformer
uda_root = '/media/ywh/pool1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659'

segmentation_folder = f'{uda_root}/pred_trainid'
confidence_folder = f'{uda_root}/pred_confidence'
entropy_folder = f'{uda_root}/pred_entropy'

segmentation_suffix = '_leftImg8bittrainID.png'
# segmentation_suffix = '_leftImg8bit.png'
#
segmentation_suffix_noimg = False

# 接在aachen_000000_000019_leftImg8bit之后的后缀
confidence_suffix = '_leftImg8bit_confi.npy'
entropy_suffix = '_leftImg8bit_entro.npy'

# sepico
# uda_root = '/media/ywh/pool1/yanweihao/projects/uda/SePiCo/work_dirs/local-exp1/230707_0324_dlv2_proj_r101v1c_sepico_DistCL-reg-w1.0-start-iter3000-tau100.0-l3-w1.0_rcs0.01_cpl_self_adamw_6e-05_pmT_poly10warm_1x2_40k_gta2cs_seed76_36629'
# segmentation_folder = f'{uda_root}/pred_trainid'
# confidence_folder = f'{uda_root}/pred_confidence'
# entropy_folder = f'{uda_root}/pred_entropy'

# mic
# best
# uda_root = '/media/ywh/pool1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230716_1343_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_ea911'
# segmentation_folder = f'{uda_root}/pred_trainid'
# confidence_folder = f'{uda_root}/pred_confidence'
# entropy_folder = f'{uda_root}/pred_entropy'

# baseline
# uda_root = '/media/ywh/pool1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230711_0741_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_139ce'
# segmentation_folder = f'{uda_root}/pred_trainid'
# confidence_folder = f'{uda_root}/pred_confidence'
# entropy_folder = f'{uda_root}/pred_entropy'

# tufl stage 1
# uda_root = '/media/ywh/pool1/yanweihao/projects/uda/BiSeNet-uda/outputs/GTA5_deeplab_BiSeNet_20kunsup_adapt_focal_0.8_0.01'
# segmentation_folder = f'{uda_root}/output_train/pred_trainid'
# confidence_folder = f'{uda_root}/output_train/pred_confidence'
# entropy_folder = f'{uda_root}/output_train/pred_entropy'

data_root = '/media/ywh/pool1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest'
# the path to the original image
image_folder = f'{data_root}/leftImg8bit/train_all'
image_suffix = '_leftImg8bit.png'
# the path to the ground truth
gt_folder = f'{data_root}/gtFine/train_all'
gt_suffix = '_gtFine_labelTrainIds.png'

# output folder
output_folder = 'outputs/cityscapes/daformer/daformer_gta_tmp2' #这是去掉了mask按照名称排序的过程

# SAM mask labeling process
get_sam_mode = 1
use_sgml=True

# C_s and C_l classes
# small_classes = [3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18] # [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # 11 classes, 5, 6, 7, 
small_classes = [3, 4, 5, 6, 7, 12, 14, 17, 18]
large_classes = [0, 1, 2, 8, 10, 13]
# large_classes = [0, 1, 2, 8, 9, 10, 11, 13, 15, 16]

# sam_alpha
sam_alpha = 0.2
adaptive_ratio = False

# road assumption
road_assumption = True
road_center_rect = (740, 780, 1645, 995)  # (740, 810, 1625, 995)

# fusion pseudo labels
# fusion mode
fusion_mode = 3  # 0, 1, 2

# sam classes, the classes sam performs better
sam_classes = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

# 
shrink_num = 2

# confidence threshold and entropy ratio
confidence_threshold_mic = 0.9
entropy_ratio_mic = 90  # relative value, lowest 90% entropy
confidence_threshold_daformer = 0.99  # absolute value
entropy_ratio_daformer = 60  # relative value, lowest 60% entropy
confidence_threshold_tufl = 0.9  # absolute value
entropy_ratio_tufl = 80  # relative value, lowest 60% entropy

### display params
display_size = (350, 700)
mix_ratio = 0.5
resize_ratio = 1

### save params
save_sam_result = False
save_mix_result = False
save_all_fusion = True

save_majority_process = False
save_sgml_process = False
save_f1_process = False
save_f2_process = False
save_f3_process = False

### time setting
time_process = True
time_filename = '2975_time2.txt'

# num of classes, 16 for synthia, 19 for gta5
num_classes = 19

# num of images to process
debug_num = 2975 # 2975 - 1648 = 1327
begin_index = 0 # 0