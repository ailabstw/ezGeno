# ezGeno
Implementation of a deep convolutional neuronal network for predicting chromatin features from DNA sequence.

The repository contains a flexible tensorflow implementation of a convolutional neuronal network with max pooling. The basic architecture is built on the principles described in DeepSEA (http://deepsea.princeton.edu/help/) and Basset (https://github.com/davek44/Basset). It is comprised of multiple layers of convolutional filters followed by ReLU and max pooling and a fully connected layer in the end. An additional pre-final fully connected layer can be switch on as well. The number of convolutional layers is flexible and selected as hyperparameter in the beginning of the training procedure. Batch normalization is optional.


### Contents

* **ezGeno.py** : Model file, implementation of the model. Flexible construction according to hyperparameters provided.

### Requirements

* Python 3.5 +
* Tensorflow v1.8.0 +
* Pre-processed chromatin feature data
* Utitlity scripts require bedtools to be installed

### Data

The basic data format is DNA sequences each assoicated with a set of chromatin features. DeepHaem require training, test and validation data stored as tensors in numpy arrays. Each set consists of the one-hot encoded sequences a 3D tensor of dimensitons (num_examples, seq_length, 4) and the labels a 2D tensor of 1's and 0's representing the chromatin features a given sequence is associated with dimensions (num_examples, num_of_chrom_features). We recommend storing data in hdf5 format. The training script provided reads the data from a hdf5 file and expects training and test set to be stored in the same file, while the validation data is provided in a separate file. Sequences and labels are expected as "training_seqs", "training_labels", "test_seqs", "test_labels", "validation_seqs" and "validation_labels"; entries in the hdf5 file. The training script is straight forward to adjust for reading the data in a different format. For saving space sequences and labels can be stored as unsigned integers or boolean values. Parse the respective data type used to the training script for translation.


### Models
**./models** contains links to already trained models.
