
for scene in 'sjtu2' # 'sjtu1' 'sjtu7' 'sjtu9' 
do
    python sort_mask.py \
    --folder_path '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene \
    --output_file '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene'.json'
done