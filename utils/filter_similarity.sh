threshold=0.1
python filter_similarity.py \
--input "output_similarity/id_info.csv" \
--output "output_similarity/similarity_matrix_${threshold}.csv" \
--column "2_div_1" \
--threshold $threshold \
--num_categories 19 \
--stat_column_1 "id1" \
--stat_column_2 "id2"