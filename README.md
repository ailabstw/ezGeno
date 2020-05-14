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
                  [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] 
                  [--optimizer OPTIMIZER] [--weight_init WEIGHT_INITILIZATION]
                  [--out_path OUT_PATH]
                  
Required arguments:
  --positive_training_data    
                        positive training data in fasta format (Format: *.fa, *.fasta)
  --negative_training_data    
                        negative training data in fasta format (Format: *.fa, *.fasta)
  
Optional arguments:
  -h, --help            
                        Show this help message and exit
  --search NAS_SEARCH
                        Performing Neural architectural search. 
                        [Type: Flag]
  --epochs EPOCHS
                        Number of epochs for training models. 
                        [Type: Int, default: 50]
  --batch_size BATCH_SIZE
                        Batch size for each training iterations. 
                        [Type: Int, default: 50]
  --learning_rate LEARNING_RATE         
                        Learning rate for training models. 
                        [Type: Float, default: 0.001]
  --momentum MOMENTUM
                        Learning rate for training models. 
                        [Type: Float, default: 0.99]
  --optimizer OPTIMIZER
                        Optimizer used for model training. 
                        [Type: String, default: "Adam"]
  --weight_init WEIGHT_INITILIZATION
                        Method used for weight initilization. 
                        [Type: String, default: "Normal"]
  --out_path OUT_PATH   
                        The output directory for the trained model. 
                        [Type: String, default: "output_dir"]
```

### Data description


### Models
**./models** contains links to already trained models.
