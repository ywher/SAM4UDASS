
# city  (2048, 1024)
# acdc  (1920, 1080)
# gta   (1914, 1052)
# syn   (1280, 760)
# pass  (2048, 400)

for data_set in "city_da" # "city_da" "city" "acdc" "acdc_da" "syn" "gta" "pass_passv2"
do
    file_csv="${data_set}_avg_area.csv"

    python plot.py \
    --csv_file ${file_csv} \
    --output_pdf "output_pdf/${data_set}/${data_set}.pdf" \
    --width 2048 \
    --height 1024
done

# data1="city"
# data2="city_da"
# python plot_diff2.py \
# --csv_file1 "${data1}_avg_area.csv" \
# --label1 "${data1}" \
# --csv_file2 "${data2}_avg_area.csv" \
# --label2 "${data2}" \
# --output_pdf "output_pdf/${data1}/${data1}_diff.pdf" \
# --bar_width 0.5 \
# --width 2048 \
# --height 1024 \
# --selected_classes 3 4 5 6 7 9 11 12 14 15 16 17 18

# --bar_width 0.0 \