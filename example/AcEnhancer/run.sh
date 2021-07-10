<<comment
1.You can download dataset from https://drive.google.com/file/d/1qLk48r1tbmfhXsEiQhhz9kpYwuoVvJEQ/view?usp=sharing
comment
<<comment
2.run ezgeno code
comment
python3 ../../ezgeno/ezgeno.py --trainFileList ./h1hesc_dnase.training.score,./h1hesc_dnase.training_input.sequence  --trainLabel ./h1hesc_dnase.training_label --testFileList ./h1hesc_dnase.validation.score,./h1hesc_dnase.validation_input.sequence --testLabel ./h1hesc_dnase.validation_label --cuda 0  --save example.model
<<comment
