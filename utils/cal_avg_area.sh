### gta5 label path
# label_path="/media/ywh/1/yanweihao/dataset/GTA5/labels_trainid"
# input_width=1914
# input_height=1052
# label_suffix=".png"
# output_filename="gta_avg_area.csv"

### synthia label path
# label_path="/media/ywh/1/yanweihao/dataset/synthia/GT/LABELS_trainid"

### city label path
# label_path="/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/gtFine/train_all"

# daformer city pseudo label
# label_path="/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/pred_trainid"
# label_suffix=".png"
# output_filename="pass_val_avg_area.csv"

# mic city pseudo label
# label_path="/media/ywh/1/yanweihao/projects/uda/MIC/seg/work_dirs/local-exp80/230711_0741_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_daformer_sepaspp_sl_mitb5_poly10warm_s0_139ce/pred_trainid"

### acdc
# acdc label path
# label_path="/media/ywh/1/yanweihao/dataset/acdc/rgb_anon/train"
# label_suffix="_rgb_anon.png"
# output_filename="acdc_avg_area.csv"

# daformer acdc pseudo label
# label_path="/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp8/230527_0645_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f753f/trainid"
# label_suffix="_rgb_anontrainID.png"
# output_filename="acdc_da_avg_area.csv"

### pass
# passv2 pseudo label
# label_path="/media/ywh/1/yanweihao/projects/uda/Trans4PASS/adaptations/pseudo_DensePASS_Trans4PASS_v2_ms_full/pred_label"
# label_path="/media/ywh/1/yanweihao/projects/uda/Trans4PASS/adaptations/pseudo_DensePASS_val_Trans4PASS_v2_ms_full/pred_label"
# label_path="/media/ywh/1/yanweihao/dataset/DensePASS/gtFine/val"
# label_suffix=".png"
# output_filename="pass_val_avg_area.csv"

output_root="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/utils/output_avg_area"
output_path="${output_root}/${output_filename}"

python cal_avg_area.py \
--input_folder "${label_path}" \
--label_suffix "${label_suffix}" \
--num_class 19 \
--output_csv "${output_path}"
