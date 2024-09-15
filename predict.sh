### dianyuan ###
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/dianyuan_driving/image_2' --output 'outputs/dianyuan'

### cityscapes ###
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/unsup_single/augsburg' --output 'outputs/cityscapes/unsup_single/augsburg'

# --model-type "vit_l" "vit_b"
# --checkpoint "sam_vit_l_0b3195.pth" "sam_vit_h_4b8939.pth" "sam_vit_b_01ec64.pth"

CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'vit_h' --checkpoint 'models/sam_vit_h_4b8939.pth' \
--input '/media/ywh/pool1/yanweihao/dataset/cityscapes_original/gtFine_trainvaltest/leftImg8bit/train_all' \
--output 'outputs/cityscapes/train_vith_2' \
--num_samples 2975 \
--begin_index 0 \
--points-per-side 32 \
--pred-iou-thresh 0.86 \
--stability-score-thresh 0.92 \
--crop-n-layers 1 \
--crop-n-points-downscale-factor 2 \
--min-mask-region-area 100 \
# --count_time True

### DensePASS ###
# for folder in 'Canberra' 'Melbourne' 'Nottingham' 'Amsterdam' 'Manila' 'Capetown' 'Edinburgh' 'Jakarta' 'Zagreb' 'Auckland' 'Bangkok' 'Osaka' 'Saopaulo' 'Florence' 'Yokohama' 'Chicago' 'Glasgow' 'Helsinki' 'Turin' 'Singapore' 'Toronto' 'Oslo' 'Seoul' 'Barcelona' 'Lisbon' 'Sandiego' 'Buenosaires' 'Dublin' 'Moscow' 'Athens' 'Copenhagen' 'Montreal' 'Istanbul' 'Mexicocity' 'Stockholm' 'Marseille' 'Brussel' 'Bremen' 'Zurich' 'Hochiminhcity'
# do
#     CUDA_VISIBLE_DEVICES=1 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
#     --input "/media/ywh/pool1/yanweihao/dataset/DensePASS/leftImg8bit/${folder}" \
#     --output "outputs/DensePASS/${folder}" \
#     --num_samples 50 \
#     --points-per-side 32 \
#     --pred-iou-thresh 0.86 \
#     --stability-score-thresh 0.92 \
#     --crop-n-layers 1 \
#     --crop-n-points-downscale-factor 2 \
#     --min-mask-region-area 100
# done

# CUDA_VISIBLE_DEVICES=1 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input "/media/ywh/pool1/yanweihao/dataset/DensePASS/leftImg8bit/val" \
# --output "outputs/DensePASS_val/vith_masks" \
# --num_samples 100 \
# --points-per-side 32 \
# --pred-iou-thresh 0.86 \
# --stability-score-thresh 0.92 \
# --crop-n-layers 1 \
# --crop-n-points-downscale-factor 2 \
# --min-mask-region-area 100

# --count_time false \
###new added from where?

# 64 0.86 0.92 1 2 100

# for folder in sjtu7 sjtu9
# do
#     CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
#     --input '/home/cyber-fx/ywh/dataset/sjtu/image/train/'$folder \
#     --output 'outputs/'$folder \
#     --points-per-side 64 \
#     --pred-iou-thresh 0.86 \
#     --stability-score-thresh 0.92 \
#     --crop-n-layers 1 \
#     --crop-n-points-downscale-factor 2 \
#     --min-mask-region-area 100
# done

### sjtu ###
# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/sjtu/image/train/sjtu1' \
# --output '/media/ywh/Elements/projects/sam/outputs/sjtu1' \
# --points-per-side 64 \
# --pred-iou-thresh 0.86 \
# --stability-score-thresh 0.92 \
# --crop-n-layers 1 \
# --crop-n-points-downscale-factor 2 \
# --min-mask-region-area 100

# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/sjtu/image/train/sjtu7' \
# --output '/media/ywh/Elements/projects/sam/outputs/sjtu7' \
# --points-per-side 64 \
# --pred-iou-thresh 0.86 \
# --stability-score-thresh 0.92 \
# --crop-n-layers 1 \
# --crop-n-points-downscale-factor 2 \
# --min-mask-region-area 100

# Requires open-cv to run post-processing

#--checkpoint sam_vit_h_4b8939, sam_vit_l_0b3195, sam_vit_b_01ec64
# --mode-type 'vit_l' 'vit_b'

### acdc ###
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/acdc/rgb_anon/train' --output 'outputs/ACDC/train'

# CUDA_VISIBLE_DEVICES=1 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/acdc/rgb_anon/train' \
# --output 'outputs/ACDC/train2' \
# --points-per-side 32 \
# --pred-iou-thresh 0.86 \
# --stability-score-thresh 0.92 \
# --crop-n-layers 1 \
# --crop-n-points-downscale-factor 2 \
# --min-mask-region-area 100 \

###fisheye
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/Elements/dataset/SurroundView-train_val_test/gtFine_trainvaltest/leftImg8bit/train/sjtu-train' --output 'outputs/fish-sjtu/train'

###mapillary
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/mapillary/leftImg8bit/train' --output 'outputs/mapillary/train'

###NTHU
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/NTHU_Datasets/Rio/Images/Train' --output 'outputs/NTHU/Rio'

# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/NTHU_Datasets/Rome/Images/Train' --output 'outputs/NTHU/Rome'

# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/NTHU_Datasets/Taipei/Images/Train' --output 'outputs/NTHU/Taipei'

# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/NTHU_Datasets/Tokyo/Images/Train' --output 'outputs/NTHU/Tokyo'

###school1
# python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/projects/segmentation/segment-anything/tools/school1' --output 'outputs/school1'

###kyxz
# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/kyxz/image/train/scene1' --output 'outputs/kyxz/scene1'

# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/kyxz/image/train/scene2' --output 'outputs/kyxz/scene2'

# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/kyxz/image/train/scene3' --output 'outputs/kyxz/scene3'

### miyuan parking ###
# for dir in front left right rear
# do
#     CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
#     --input '/media/ywh/pool1/yanweihao/dataset/miyuan_parking/'$dir --output 'outputs/miyuan_parking/'$dir
# done

### fenxian camera ###
# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/projects/MTMC/AIC21-MTMC/datasets_fenxian/detection/132_crop' --output 'outputs/fenxian/132'

### ivfc_zhuitong ###
# CUDA_VISIBLE_DEVICES=0 python scripts/amg.py --model-type 'default' --checkpoint 'models/sam_vit_h_4b8939.pth' \
# --input '/media/ywh/pool1/yanweihao/dataset/ivfc/zhuitong/output_frames' --output 'outputs/ivfc/zhuitong'