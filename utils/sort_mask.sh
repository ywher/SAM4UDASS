
### SJTU
# for scene in 'sjtu2' # 'sjtu1' 'sjtu7' 'sjtu9' 
# do
#     python sort_mask.py \
#     --folder_path '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene \
#     --output_file '/media/ywh/Elements/projects/segmentation/segment-anything/outputs/'$scene'.json'
# done

### cityscapes
# data_root="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes"
# folder_name="train_vitb_default"
# mask_folder_path="${data_root}/${folder_name}"
# echo "${folder_name} begin"
# python sort_mask.py \
# --folder_path ${mask_folder_path} \
# --output_file "${data_root}/mapping_json/${folder_name}.json"

data_root="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/cityscapes/other_sam"
for folder_name in "samhq"
do
    mask_folder_path="${data_root}/${folder_name}"
    echo "${folder_name} begin"
    python sort_mask.py \
    --folder_path ${mask_folder_path} \
    --output_file "${data_root}/mapping_json/${folder_name}.json"
done


### DensePASS
# data_root="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/DensePASS"
# for folder in 'Canberra' 'Melbourne' 'Nottingham' 'Amsterdam' 'Manila' 'Capetown' 'Edinburgh' 'Jakarta' 'Zagreb' 'Auckland' 'Bangkok' 'Osaka' 'Saopaulo' 'Florence' 'Yokohama' 'Chicago' 'Glasgow' 'Helsinki' 'Turin' 'Singapore' 'Toronto' 'Oslo' 'Seoul' 'Barcelona' 'Lisbon' 'Sandiego' 'Buenosaires' 'Dublin' 'Moscow' 'Athens' 'Copenhagen' 'Montreal' 'Istanbul' 'Mexicocity' 'Stockholm' 'Marseille' 'Brussel' 'Bremen' 'Zurich' 'Hochiminhcity'
# do
#     mask_folder_path="${data_root}/${folder}"
#     python sort_mask.py \
#     --folder_path ${mask_folder_path} \
#     --output_file "${data_root}/mapping_json/${folder_name}.json"
# done

# DensePASS val
# data_root="/media/ywh/1/yanweihao/projects/segmentation/segment-anything/outputs/DensePASS_val"
# folder_name="vith"
# mask_folder_path="${data_root}/${folder_name}"
# python sort_mask.py \
# --folder_path ${mask_folder_path} \
# --output_file "${data_root}/mapping_json/${folder_name}.json"