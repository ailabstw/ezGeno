# ezGeno 
ezGeno is an implementation of the efficient neural architecture search algorithm specifically tailored for genomic sequence categorization, for example predicting transcription factor (TF) binding sites and histone modifications. 

This repository contains a pytorch implementation of an eNAS algorithm, where parameters can be altered and adjusted by users. Here, we used two examples to demonstrate how ezGeno can be applied to employ deep learning on genomic data categorization:
* predicting TF binding. The basic architecture of this idea was built based on DeepBind (https://github.com/kundajelab/deepbind).
* predicting activity of enhancers. The basic architecture of this idea was built based on accuEnhancer.

## workflow

### Contents

*  **network.py**          : Model file, implementation of the model. Flexible construction according to hyperparameters provided.
*  **ezgeno.py**           : Users can pass parameters to the train model.
*  **utils.py**            : file for helper functions.
*  **controller.py**       : Controller(Agent) will use reinforcemnt learning to learn which architecture is better from the available search space.
*  **dataset.py**          : define ezgeno input file data formats.
*  **epigenomeDataset**    : define epigenome input file data formats.
*  **trainer**             : define training steps.
*  **visualize.py**        : visualize sequence position importance and output sub-sequence data whose score surpasses threshold.

### Requirements

#### Packages (refer from requirements.txt)
* biopython==1.77
* cycler==0.10.0
* future==0.18.2
* joblib==0.16.0
* kiwisolver==1.2.0
* matplotlib==3.3.0
* numpy==1.19.0
* opencv-python==4.3.0.36
* pandas==1.0.5
* Pillow==7.2.0
* pyparsing==2.4.7
* python-dateutil==2.8.1
* pytz==2020.1
* scikit-learn==0.23.1
* scipy==1.5.1
* seaborn==0.10.1
* six==1.15.0
* sklearn==0.0
* threadpoolctl==2.1.0
* torch==1.5.1
* torchvision==0.6.1
* tqdm==4.48.0
* utils==1.0.1

* Bedtools (Tested on 2.28.0)

#### Input data
* Pre-processed peak files 

## Usage
```bash
usage: ezgeno.py  [-h help] 
                  [--task TASK]
                  
                  [--train_pos_data_path POSITIVE_TRAINING_DATA] 
                  [--train_neg_data_path NEGATIVE_TRAINING_DATA]
                  [--test_pos_data_path POSITIVE_TESTING_DATA] 
                  [--test_neg_data_path NEGATIVE_TESTING_DATA]
                  
                  [--train_dNase_path TRAINING_dNase_DATA ]
                  [--train_seq_path TRAINING_seq_DATA]
                  [--train_label_path TRAINING_label_DATA]
                  [--test_dNase_path Testing_dNase_DATA]
                  [--test_seq_path Testing_seq_DATA]
                  [--test_label_path Testing_label_DATA]

                  [--batch_size BATCH_SIZE] [--optimizer OPTIMIZER]
                  [--epochs EPOCHS] [--learning_rate LEARNING_RATE] 
                  [--supernet_learning_rate SUPERNET_LEARNING_RATE] [--supernet_epochs SUPERNET_EPOCHS] 
                  [--controller_learning_rate CONTROLLER_LEARNING_RATE] [--controller_optimizer CONTROLLER_OPTIMIZER] [--cstep CSTEP]
                  [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] 
                             
                  [--layers LAYERS] [--feature_dim FEATURE_DIM]         
                  [--conv_filter_size_list CONV_FILTER_SIZE_LIST]
                  [--dNase_layers DNASE_LAYERS] [--dNase_feature_dim DNASE_FEATURE_DIM]         
                  [--dNase_conv_filter_size_list DNASE_CONV_FILTER_SIZE_LIST]

                  [--cuda CUDA]
                  [--eval EVAL]
                  [--load MODEL_NAME]
                  [--save MODEL_NAME ]
                  
Required arguments:
1.TFBind
  --train_pos_data_path    
                        positive training data path. 
                        [Type: String]  
  --train_neg_data_path
                        negative training data path. 
                        [Type: String]  
  --test_pos_data_path    
                        positive testing data path. 
                        [Type: String]  
  --test_neg_data_path
                        negative testing data path. 
                        [Type: String]
 2.epigenome
  --train_dNase_path    
                        training data dNase path. A file with a 25*1 vector per row. The elements in the vector are of type int, which represent DNA expression. 
                        [Type: String]  
  --train_seq_path
                        training data sequence path. A file with 200 base pairs per row. 
                        [Type: String]  
                        
  --train_label_path
                        training data label path. Each row is 1 or 0, denoting whether the TF will bind or not.
                        [Type: String]
  
  --test_dNase_path    
                        testing data dNase path. It is a file with 25*1 vector per row. The elements in the vector are of type int, which represent DNA expression. 
                        [Type: String]   
  --test_seq_path
                        testing data sequence path. A file with 200 base pairs per row. 
                        [Type: String]          
                
  --test_label_path
                        testing data label path. Each row is 1 or 0, denoting whether the TF will bind or not.
                        [Type: String]
                        
Optional arguments:
  -h, --help            
                        Show this help message and exit
  --task 
                        "TFBind": predicting TF binding
                        "epigenome": predicting activity of enhancers
                        [Type: String, default: "TFBind", options: "TFBind, epigenome"]
                        
  --negative_data_method NEGATIVE_DATA_METHOD  
                        If not given the negative training data, ezGeno will generate 
                        negative data based on the selected methods.
                        "random": random sampling from the human genome.
                        "dinucl": generate negative sequence based on the same dinucleotide
                                  composition of the positive training data.
                        [Type: String, Default:"dinucl", options: "random, dinucl"]
                        
  --epochs EPOCHS
                        Number of epochs for training searched model. 
                        [Type: Int, default: 100]
                      
  --supernet_epochs SUPERNET_EPOCHS
                        Number of epochs for training supernet. 
                        [Type: Int, default: 50]
  --cstep
                        Number of steps for training controller.

  --batch_size BATCH_SIZE
                        Batch size for each training iterations. 
                        [Type: Int, default: 128]
  --learning_rate LEARNING_RATE         
                        Learning rate for training searched model. 
                        [Type: Float, default: 0.001]
  --supernet_learning_rate SUPERNET_LEARNING_RATE         
                        Learning rate for training supernet. 
                        [Type: Float, default: 0.01]
  --controller_learning_rate CONTROLLER_LEARNING_RATE         
                        Learning rate for training controller. 
                        [Type: Float, default: 0.1]
  --momentum MOMENTUM
                        Learning rate for training searched model. 
                        [Type: Float, default: 0.9]
  --weight_decay WEIGHT_DECAY
                        Weight decay. 
                        [Type: Float, default: 0.0005]  
  --optimizer OPTIMIZER
                        Optimizer used for training models. 
                        [Type: String, default: "sgd", options: "sgd, adam, adagrad"]
  --controller_optimizer CONTROLLER_OPTIMIZER
                        Optimizer used for training controller. 
                        [Type: String, default: "adam", options: "sgd, adam, adagrad"]
  
  --weight_init WEIGHT_INITILIZATION
                        Method used for weight initilization. 
                        [Type: String, default: "Normal"]
                        
  --layers 
                        1. In TFBind task, we use this parameter to determine the layers of convolution units.
                        2. In epigenome task, we use this parameter to determine the layers of convolution units in sequence module.
                        [Type: int, default: 3]
  --feature_dim
                        1. In TFBind task, we use this parameter to determine the number of convolution filters.
                        2. In epigenome task, we use this parameter to determine the number of convolution filters in sequence module.
                        [Type: int, default: 64]
  --conv_filter_size_list
                        1. In TFBind task, we use this parameter to determine the filter size list of convolution filters. Our purposed method will 
                        find the best filter size from this list by reinforcement learning.
                        2. In epigenome task, we use this parameter to determine the filter size list of convolution filters in sequence module.
                        Our purposed method will find the best filter size from this list by reinforcement learning.
                        [Type: List, default: [3,7,11,15,19] ]
  --dNase_layers
                        The parameter is only work on epigenome task. We use this parameter to determine the layers of convolution units in dNase modules.
                        [Type: int, default: 6]
  --dNase_feature_dim
                        The parameter is only work on epigenome task. We use this parameter to determine the number of convolution filters in dNase module.
                        [Type: int, default: 64]
  --dNase_conv_filter_size_list
                        The parameter is only works on epigenome tasks. We use this parameter to determine the filter size list of convolution filters in dNase
                        module.      
                        [Type: List, default: [3,7,11] ]
                                      
  --cuda 
                        We use this parameter to determine to use cuda or not. If you want to use gpu, you can type in gpu index, e.g.: 0.
                        If you want to use cpu only, you can type -1.
                        [Type: Int, default: -1 ]

  --eval 
                        This flag is used to predict testing data directly. 
                        It is usually used with "load" parameter.
                        [Type: Bool, default: False]
                       
  --load 
                        This parameter is treated as loaded path. We will load modules from this path.
                        [Type: str, default: './model.t7']
  --save 
                        This parameter is treated as saved path. We will save trained modules to this path after training.
                        [Type: str, default: './model.t7']           

```


## Installation
1) Download/Clone ezGeno
``` bash
git clone https://github.com/ailabstw/ezGeno.git

cd ezGeno

```

2) Install required packages
``` bash
apt-get install bedtools
apt-get install python3
apt-get install python3-distutils
apt-get install libglib2.0-0
apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install -r requirements.txt
```

### Dataset
1)TFBind: We downloaded data transcription factors ChIP-seq called peaks from the curated database of deepBind. Alipanahi et al. (2015) from the ENCODE database.

2)Epigenome:The portal now makes available over 13000 datasets and their accompanying metadata and can be accessed at: https://www.encodeproject.org/ .

### Models
**./models** contains links of previous trained models.


### Model Archietcture


## Example1 - TFBind:
users can run a sample dataset with the following: "./example/tfbind/run.sh".
### 1. preprocesing
Please refer to the ReadMe file in the preprocessing folder
### 2. eNAS
```python
 python3 ezgeno.py --task TFBind --cuda 0 --train_pos_data_path ../SUZ12/SUZ12_positive_training.fa --train_neg_data_path ../SUZ12/SUZ12_negative_training.fa  --test_pos_data_path ../SUZ12/SUZ12_positive_test.fa --test_neg_data_path ../SUZ12/SUZ12_negative_test.fa
 ```
#### (optional) modify layers parameters 
```python
 python3 ezgeno.py --layers 6 --task TFBind --cuda 0 --train_pos_data_path ../SUZ12/SUZ12_positive_training.fa --train_neg_data_path ../SUZ12/SUZ12_negative_training.fa  --test_pos_data_path ../SUZ12/SUZ12_positive_test.fa --test_neg_data_path ../SUZ12/SUZ12_negative_test.fa 
 ```
#### (optional) modify search space (convolution filter size) parameters 
```python
 python3 ezgeno.py --conv_filter_size_list [3,7,11,15,19] --task TFBind --cuda 0 --train_pos_data_path ../SUZ12/SUZ12_positive_training.fa --train_neg_data_path ../SUZ12/SUZ12_negative_training.fa  --test_pos_data_path ../SUZ12/SUZ12_positive_test.fa --test_neg_data_path ../SUZ12/SUZ12_negative_test.fa  
 ```
#### (optional) modify the number of output channels parameters 
```python
 python3 ezgeno.py --feature_dim 128 --task TFBind --cuda 0 --train_pos_data_path ../SUZ12/SUZ12_positive_training.fa --train_neg_data_path ../SUZ12/SUZ12_negative_training.fa  --test_pos_data_path ../SUZ12/SUZ12_positive_test.fa --test_neg_data_path ../SUZ12/SUZ12_negative_test.fa 
 ```
#### (optional) load model and predict
```python
 python3 ezgeno.py --load model.t7 --cuda 0 --eval --test_pos_data_path ../SUZ12/SUZ12_positive_test.fa --test_neg_data_path ../SUZ12/SUZ12_negative_test.fa
```
### Performance evaluaion:

![AUC curve](https://github.com/ailabstw/ezGeno/blob/master/model_comparision.png)

![time cost](https://github.com/ailabstw/ezGeno/blob/master/compare_with_methods_time.png)

### 3. visualize and get sub sequence based on prediction model 
```python
 python3 visualize.py --load model.t7 --data_path ../SUZ12/SUZ12_positive_test.fa --dataName SUZ12 --target_layer_names "[2]"
``` 
#### (optional) you can choose sequence range which you want to show based on "show_seq" parameter. e.g.all,top-100,50-200
```python
 python3 visualize.py --show_seq top-200 --load model.t7 --data_path ../SUZ12/SUZ12_positive_test.fa --dataName SUZ12 --target_layer_names "[2]"
``` 



We highlight the important region in each sequence based on the predictive model. As shown in the image below, our model is able to identify regions that are important to determining possible binding sites.

![seq-heatmap](https://github.com/ailabstw/ezGeno/blob/master/seq-heapmap.jpg)

We also collect the sub-sequences whose scores surpass the threshold and save them in fasta format. This file can be treated as the input to a motif discovery tool (e.g. meme) to generate motif in sub sequences. As shown in the image below, the left sequence logo is based on motif discovery from these sub sequences, and the right sequence logo is from hocomoco database. We can find a reliable and consistent result using our tool.

![seq-heatmap](https://github.com/ailabstw/ezGeno/blob/master/motif_discovery_compare_example.jpg)

 
## Example2 - Enhancer Activity:
users can run a sample dataset with the following: "./example/enhancer/run.sh".
### preprocesing
Please refer to the ReadMe file in the preprocessing folder
### train
``` python
python3 ezgeno.py --task AcEnhancer --cuda 0 --train_dNase_path ../dNase/h1hesc_dnase.training.score --train_seq_path ../dNase/h1hesc_dnase.training_input_seq 
--train_label_path ../dNase/h1hesc_dnase.training_label --test_dNase_path ../dNase/h1hesc_dnase.validation.score --test_seq_path ../dNase/h1hesc_dnase.validation_input_seq --test_label_path ../dNase/h1hesc_dnase.validation_label
``` 
### (optional) load model and predict 
``` python
 python3 ezgeno.py --task AcEnhancer --cuda 0 
``` 

### Performance Evaluation
![AUC curve](https://github.com/ailabstw/ezGeno/blob/master/active_enhancer_AUC.png)

![time cost](https://github.com/ailabstw/ezGeno/blob/master/active_enhancer_time.png)

