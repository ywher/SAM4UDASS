# python img_2_video.py \
# --input_folder '/media/ywh/1/yanweihao/dataset/kyxz/image/train/scene1' \
# --frame_rate 2 \
# --data_ratio 0.2 \
# --output_file '/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/kyxz_mix/scene1/scene1_original.mp4' \

# --input_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/kyxz_mix/scene3/rgb' \
#/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all
#/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/school1/rgb
# --input_folder "/media/ywh/1/yanweihao/dataset/miyuan_parking/${dir}" \

for dir in front left right rear
do
    python img_2_video.py \
    --input_folder "/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/miyuan_parking/${dir}/rgb" \
    --frame_rate 2 \
    --data_ratio 0.5 \
    --output_file "/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/miyuan_parking/${dir}/${dir}.mp4"
done