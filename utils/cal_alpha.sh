# 计算数据集中的alpha分布，一个mask中包含了两类物体，其中数量更少的物体和数量更多的物体的像素数量比例范围为(0,1]
python cal_alpha.py \
--csv_file "output_similarity/id_info.csv" \
--column_name "2_div_1" \
--percentage_interval 5 \
--alpha_thres -1 \
--pdf_output "output_alpha/distribution.pdf" \
--txt_output "output_alpha/data.txt" \
--x_label "" \
--y_label ""