
for scene in 'sjtu1' 'sjtu2' 'sjtu7' 'sjtu9'
do
    python sort_mask.py \
    --folder_path '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene \
    --output_file '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene'.json'
done