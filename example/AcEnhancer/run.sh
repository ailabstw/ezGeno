<<comment
1.You can download dataset from https://drive.google.com/file/d/1qLk48r1tbmfhXsEiQhhz9kpYwuoVvJEQ/view?usp=sharing
comment
<<comment
2.run ezgeno code
comment
python3 ../../ezgeno/ezgeno.py --trainFileList /volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.training.score,/volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.training_input.sequence  --trainLabel /volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.training_label --testFileList /volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.validation.score,/volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.validation_input.sequence --testLabel /volume/tsungting/ezgeno/eNAS/epigenome/data/h1hesc_dnase.validation_label --cuda 0  --save example.model
<<comment
