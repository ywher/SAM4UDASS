
# define the folder path and parameters
# train2 is the folder of the generated mask, preciser
# train is the folder of the generated mask, default sam params
mask_folder = '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2'
# the path to the model prediction
# segmentation_root = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs'

#pred folder for gta
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/entropy'

#pred folder for synthia
segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_trainid_new'
confidence_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_confidence'
entropy_folder = '/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/syn/230526_1633_syn2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_103e3/pred_entropy'
# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-basic/230509_1455_gtaHR2csHR_mic_hrda_s2_108c1/pred_trainid'

# segmentation_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230610_0813_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_85394/pred_trainid'
# confidence_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230610_0813_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_85394/pred_confidence'
# entropy_folder = '/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230610_0813_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_85394/pred_entropy'

# the path to the original image
image_folder = '/media/ywh/pool1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all'
# the path to the ground truth
gt_folder = '/media/ywh/pool1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/gtFine/train_all'

#
sam_model_type = 'vit_h'
sam_model_path = './models/sam_vit_h_4b8939.pth'
device = 'cuda:1'
#
mix_ratio = 0.5
#
resize_ratio = 0.5
#
output_folder = 'outputs/cityscapes/debug_syn_vis'  # 这是去掉了mask按照名称排序的过程
#
mask_suffix = '.png'
#
# segmentation_suffix = '_gtFine_labelTrainIds.png'
# segmentation_suffix = '_leftImg8bittrainID.png'
segmentation_suffix = '_leftImg8bittrainID.png'
#
segmentation_suffix_noimg = False
#
confidence_suffix = '_confi.npy'
entropy_suffix = '_entro.npy'
confidence_threshold = 0.9  # absolute value
entropy_ratio = 60  # relative value, lowest 70% entropy
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
save_sam_color = True
save_uda_color = True
save_fusion_color = True
save_majority_process = True
save_sgml_process = True
save_f1_process = True
save_f2_process = True
save_f3_process = True

# num of classes, 16 for synthia, 19 for gta5
num_classes = 16

# num of images to process
debug_num = 1 # 2975 - 106 = 869
begin_index = 2373 # 0