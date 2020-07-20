# ezGeno
Implementation of an efficient neuronal architecture search algorithms for predicting transcription factor binding sites from ChIP-seq data.

The repository contains a flexible pytorch implementation of a eNAS algorithm, where parameteres can be altered and adjusted by users. The basic architecture and idea was built based on DeepBind (https://github.com/kundajelab/deepbind).


### Contents

* **ezGeno.py** : Model file, implementation of the model. Flexible construction according to hyperparameters provided.

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
apt-get install bedtools
apt-get install python3
apt-get install python3-distutils
apt-get install libglib2.0-0
apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install -r requirements.txt
```

### Dataset


### Models
**./models** contains links to already trained models.
