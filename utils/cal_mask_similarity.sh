input_mask_path="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/sam_masks/train_vith_32_86_92" # subfolder: aachen_000000_000019_leftImg8bit/1.png
input_label_path="/media/ywh/1/yanweihao/projects/uda/DAFormer/work_dirs/local-exp7/gta/230522_2312_gta2cs_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_ea659/pred_trainid" # hanover_000000_027998_leftImg8bittrainID.png
input_image_folder="/media/ywh/1/yanweihao/dataset/city_tmp/leftImg8bit"

python cal_mask_similarity.py \
--input_folder_masks "${input_mask_path}" \
--mask_suffix ".png" \
--input_folder_labels "${input_label_path}" \
--label_suffix "_leftImg8bittrainID.png" \
--input_image_path "${input_image_folder}" \
--image_suffix "_leftImg8bit.png" \
--num_classes 19 \
--ratio_thres 0.3 \
--output_csv1 "/media/ywh/1/yanweihao/projects/segmentation/segment-anything/utils/output_similarity/id_info0.3.csv" \
--output_csv2 "/media/ywh/1/yanweihao/projects/segmentation/segment-anything/utils/output_similarity/similarity_matrix0.3.csv" \
# --show_mask
