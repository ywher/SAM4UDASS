data1="city"
data2="city_da"
col_name="avg_area" #"occur_freq"
python plot_diff2.py \
--csv_file1 "${data1}_avg_area.csv" \
--label1 "ground truth" \
--csv_file2 "${data2}_avg_area.csv" \
--label2 "pseudo label" \
--iou_csv "table_data.csv" \
--output_pdf "output_pdf/${data1}/${data1}_${col_name}_diff.pdf" \
--col_name "${col_name}" \
--bar_width 0.4 \
--width 2048 \
--height 1024 \
--divide_img_area \
--selected_classes 3 4 5 6 7 9 11 12 14 15 16 17 18

# --divide_img_area \

# pass
# data1="pass_val"
# data2="pass_val_passv2"
# python plot_diff2.py \
# --csv_file1 "${data1}_avg_area.csv" \
# --label1 "${data1}" \
# --csv_file2 "${data2}_avg_area.csv" \
# --label2 "${data2}" \
# --output_pdf "output_pdf/${data1}/${data1}_diff.pdf" \
# --bar_width 0.4 \
# --width 2048 \
# --height 400 \
# --selected_classes 3 4 5 6 7 9 11 12 14 15 16 17 18