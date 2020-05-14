# ezGeno
Implementation of an efficient neuronal architecture search algorithms for predicting transcription factor binding sites from ChIP-seq data.

The repository contains a flexible pytorch implementation of a eNAS algorithm, where parameteres can be altered and adjusted by users. The basic architecture and idea was built based on DeepBind (https://github.com/kundajelab/deepbind).


### Contents

* **ezGeno.py** : Model file, implementation of the model. Flexible construction according to hyperparameters provided.

### Requirements
#### Packages
* Python 3.5 +
* Pytorch 1.4 +
* Bedtools (Tested on 2.28.0)
#### Input data
* Pre-processed peak files 

## Usage
```bash
usage: ezgeno.py  [-h help] 
                  [--positive_training_data POSITIVE_TR_DATA] 
                  [--negative_training_data NEGATIVE_TR_DATA]
                  [--search NAS_SEARCH] [--epochs EPOCHS] 
                  [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] 
                  [--supernet_learning_rate SUPERNET_LEARNING_RATE] 
                  [--controller_learning_rate CONTROLLER_LEARNING_RATE] 
                  [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] 
                  [--optimizer OPTIMIZER] [--weight_init WEIGHT_INITILIZATION]
                  [--negative_data_method NEGATIVE_DATA_METHOD] 
                  [--search_cost SEARCH_COST]
                  [--supernet_epochs SUPERNET_EPOCHS]
                  [--out_path OUT_PATH]
                  
                  
Required arguments:
  --positive_training_data    
                        positive training data in fasta (*.fa, *.fasta) format. 
                        [Type: String]  
Optional arguments:
  -h, --help            
                        Show this help message and exit
  --negative_data_method NEGATIVE_DATA_METHOD  
                        If not given the negative training data, ezGeno will generate 
                        negative data based on the selected methods.
                        "random": random sampling from the human genome.
                        "dinucl": generate negative sequence based on same dinucleotide
                                  composition with the positive training data.
                        [Type: String, Default:"dinucl", options: "random, dinucl"]
  --negative_training_data    
                        negative training data in fasta (*.fa, *.fasta) format.
                        [Type: String]
  --search NAS_SEARCH
                        Performing Neural architectural search. 
                        [Type: Flag]
  --epochs EPOCHS
                        Number of epochs for training models. 
                        [Type: Int, default: 100]
  --batch_size BATCH_SIZE
                        Batch size for each training iterations. 
                        [Type: Int, default: 50]
  --learning_rate LEARNING_RATE         
                        Learning rate for training models. 
                        [Type: Float, default: 0.001]
  --supernet_learning_rate SUPERNET_LEARNING_RATE         
                        Learning rate for supernet search models. 
                        [Type: Float, default: 0.01]
  --controller_learning_rate CONTROLLER_LEARNING_RATE         
                        Learning rate during controller phase. 
                        [Type: Float, default: 0.1]
  --momentum MOMENTUM
                        Learning rate for training models. 
                        [Type: Float, default: 0.9]
  --weight_decay WEIGHT_DECAY
                        Weight decay. 
                        [Type: Float, default: 0.0005]  
  --optimizer OPTIMIZER
                        Optimizer used for model training. 
                        [Type: String, default: "sgd", options: "sgd, adam, adagrad"]
  --weight_init WEIGHT_INITILIZATION
                        Method used for weight initilization. 
                        [Type: String, default: "Normal"]
  --search_cost SEARCH_COST
                        Two kinds of pre-defined search cost parameter sets. 
                        [Type: String, default: "Fast", options: "fast, best"]
  --supernet_epochs SUPERNET_EPOCHS
                        Number of epochs for supernet search. 
                        [Type: Int, default: 50]
  --out_path OUT_PATH   
                        The output directory for the trained model. 
                        [Type: String, default: "output_dir"]
```


## Installation
1) Download/Clone ezGeno
```bash
git clone https://github.com/ailabstw/ezGeno.git

cd ezGeno
```

2) Install required packages
```bash
pip3 install torch
apt-get install bedtools
```

### Dataset


### Models
**./models** contains links to already trained models.
