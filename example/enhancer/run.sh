<<comment
1.You can download dataset from https://drive.google.com/file/d/1qLk48r1tbmfhXsEiQhhz9kpYwuoVvJEQ/view?usp=sharing
comment
<<comment
2.run ezgeno code
comment
python3 ../../ezgeno/ezgeno.py --task epigenome --cuda 0 --train_dNase_path ./h1hesc_dnase.training.score --train_seq_path ./h1hesc_dnase.training_input_seq 
--train_label_path ./h1hesc_dnase.training_label --test_dNase_path ./h1hesc_dnase.validation.score --test_seq_path ./h1hesc_dnase.validation_input_seq --test_label_path ./h1hesc_dnase.validation_label
<<comment