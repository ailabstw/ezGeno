<<comment
1.data prepareing
comment
python3 ../../preprocess/createdata.py --filename NFE2_K562_NF-E2_Yale_AC.seq --neg_type dinucleotide --outputprefix NFE2_training
python3 ../../preprocess/createdata.py --filename NFE2_K562_NF-E2_Yale_B.seq --outputprefix NFE2_testing --reverse False
<<comment
2.run ezgeno code
comment
python3 ../../ezgeno/ezgeno.py --task TFBind --trainFileList NFE2_training.sequence --trainLabel NFE2_training.label --testFileList NFE2_testing.sequence --testLabel NFE2_testing.label --cuda 0  --save example.model
<<comment
3.visualize
comment
python3 ../../ezgeno/visualize.py --show_seq all --load example.model --data_path ./NFE2_positive_test.fa --dataName NFE2 --target_layer_names "[2]" --use_cuda True

