
date_root="/media/ywh/pool1/yanweihao/projects/segmentation/segment-anything/outputs"
dataset="cityscapes"
for scene in "train_vith_2" # 'sjtu1' 'sjtu7' 'sjtu9' 
do
    python sort_mask.py \
    --folder_path "${date_root}/${dataset}/${scene}" \
    --output_file "${date_root}/${dataset}/${scene}.json"
done