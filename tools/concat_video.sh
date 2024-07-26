# for scene in scene2 scene3
# do
#     python concat_video.py \
#     --inputs "/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/kyxz_mix/${scene}/${scene}_original.mp4" \
#     "/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/kyxz_mix/${scene}/${scene}.mp4" \
#     --interval 5 \
#     --output "/media/ywh/1/yanweihao/projects/segmentation/segment-anything-old/tools/outputs/kyxz_mix/${scene}/${scene}_concat.mp4"
# done

python concat_video.py \
--inputs "/home/ywh/Videos/v1.mp4" "/home/ywh/Videos/v2.mp4" "/home/ywh/Videos/v3.mp4" \
--interval 50 \
--output "/home/ywh/Videos/v_concat.mp4"