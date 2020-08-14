<<comment
1.data prepareing
comment
python3 ../../preprocess/createdata.py --filename NFE2_K562_NF-E2_Yale_AC.seq
<<comment
2.run ezgeno code
comment
python3 ../../ezgeno/ezgeno.py --task TFBind --cuda -1 --train_pos_data_path ./NFE2_positive_data.fa --train_neg_data_path ./NFE2_dinucleotide_negative_data.fa --test_pos_data_path ./NFE2_positive_test.fa --test_neg_data_path ./NFE2_negative_test.fa --save example.model
<<comment
3.visualize
comment
python3 ../../ezgeno/visualize.py --show_seq all --load example.model --data_path ./NFE2_positive_test.fa --dataName NFE2 --target_layer_names "[2]"

