###dianyuan
# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/dianyuan' \
# --image_folder '/media/ywh/1/yanweihao/dataset/dianyuan_driving/image_2' \
# --output_folder 'outputs/dianyuan_mix' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'


###cityscapes
# python vis_mask.py \
# --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/train2' \
# --image_folder '/media/ywh/1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all' \
# --output_folder 'outputs/cityscapes_mix2' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'



###acdc
python vis_mask.py \
--mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ACDC/train2' \
--image_folder '/media/ywh/1/yanweihao/dataset/acdc/rgb_anon/train' \
--output_folder 'outputs/acdc_mix2' \
--mix_ratio 0.5 \
--mask_suffix '.png' \
--img_suffix '.png'

#kyxz
# for scene in scene2 scene3
# do
#     python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/outputs/kyxz/'$scene \
#     --image_folder '/media/ywh/1/yanweihao/dataset/kyxz/image/train/'$scene --output 'outputs/kyxz/'$scene \
#     --output_folder 'outputs/kyxz_mix/'$scene \
#     --mix_ratio 0.5 \
#     --mask_suffix '.png' \
#     --img_suffix '.png'
# done


# #miyuan parking
# for dir in front left right rear
# do
#     python vis_mask.py \
#     --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/outputs/miyuan_parking/'$dir \
#     --image_folder '/media/ywh/1/yanweihao/dataset/miyuan_parking/'$dir \
#     --output_folder 'outputs/miyuan_parking/'$dir \
#     --mix_ratio 0.5 \
#     --mask_suffix '.png' \
#     --img_suffix '.jpg'
# done

###
# python vis_mask.py \
# --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/outputs/fenxian/132' \
# --image_folder '/media/ywh/1/yanweihao/projects/MTMC/AIC21-MTMC/datasets_fenxian/detection/132_crop' \
# --output_folder 'outputs/fenxian/132' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'

###fisheye
# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/fish-sjtu/train' \
# --image_folder '/media/ywh/Elements/dataset/SurroundView-train_val_test/gtFine_trainvaltest/leftImg8bit/train/sjtu-train' \
# --output_folder 'outputs/fish_mix' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'


###NTHU
# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/NTHU/Rio' \
# --image_folder '/media/ywh/1/yanweihao/dataset/NTHU_Datasets/Rio/Images/Train' \
# --output_folder 'outputs/NTHU/Rio_mix' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.jpg'

# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/NTHU/Rome' \
# --image_folder '/media/ywh/1/yanweihao/dataset/NTHU_Datasets/Rome/Images/Train' \
# --output_folder 'outputs/NTHU/Rome_mix' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.jpg'

# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/NTHU/Taipei' \
# --image_folder '/media/ywh/1/yanweihao/dataset/NTHU_Datasets/Taipei/Images/Train' \
# --output_folder 'outputs/NTHU/Taipei_mix' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.jpg'

###school
# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/school1' \
# --image_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/tools/school1' \
# --output_folder 'outputs/school1' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'

#ivfc zhuitong
# python vis_mask.py --mask_folder '/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/ivfc/zhuitong' \
# --image_folder '/media/ywh/1/yanweihao/dataset/ivfc/zhuitong/output_frames' \
# --output_folder 'outputs/ivfc/zhuitong' \
# --mix_ratio 0.5 \
# --mask_suffix '.png' \
# --img_suffix '.png'


###panoramic
