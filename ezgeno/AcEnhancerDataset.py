import numpy as np
from Bio import SeqIO
import string
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import csv
from utils import *

class trainset(Dataset):
    def __init__(self, seqFileName,dNaseFileName,labelFileName):
        seq_data=[]
        dNase_data=[]
        with open(seqFileName) as outputfile:
            for sequence in outputfile:
                seq_data.append(str(sequence.split()[0]))
        print(len(seq_data))
        
        with open(dNaseFileName, 'r') as f:
            reader = csv.reader(f, delimiter=" ")
            print(reader)
            dNase_data = np.array(list(reader))
        
        dNase_data=np.delete(dNase_data, -1, axis=1).astype(float)
        labels =np.loadtxt(labelFileName).astype(int)
        shuffle_data = list(zip(seq_data,dNase_data, labels))
        random.shuffle(shuffle_data)
        seq_datas,dNase_datas, labels = zip(*shuffle_data)

        self.encoded_seq = np.transpose(np.array(onehot_encode_sequences(np.array(seq_datas)), dtype='float32'), (0, 2, 1))
        #self.encoded_seq = np.stack(np.array(onehot_encode_sequences(seq_datas),dtype='float32'))
        self.train_labels = np.array(labels)
        self.dNasedata =np.expand_dims(dNase_datas, axis=1).astype(float) 

        print('self.encoded_seq:',self.encoded_seq.shape)
        print('self.dNasedata:',self.dNasedata.shape)

        
    def __getitem__(self, index):
        seq_data = self.encoded_seq[index]
        dNase_data = self.dNasedata[index]
        label = self.train_labels[index]
        seq_data = np.expand_dims(seq_data, axis=0)

        return seq_data,dNase_data, label

    def __len__(self):
        return len(self.encoded_seq)

class testset(Dataset):
    def __init__(self,seqFileName,dNaseFileName,labelFileName):
        seq_data=[]
        dNase_data=[]
        with open(seqFileName) as outputfile:
            for sequence in outputfile:
                seq_data.append(str(sequence.split()[0]))
        
        with open(dNaseFileName, 'r') as f:
            reader = csv.reader(f, delimiter=" ")
            print(reader)
            dNase_data = np.array(list(reader))

        dNase_data=np.delete(dNase_data, -1, axis=1).astype(float) 
        test_labels =np.loadtxt(labelFileName).astype(int)
        #test_shuffle_data = list(zip(seq_data,dNase_data, test_labels))
        #seq_datas,dNase_datas, test_labels = zip(*test_shuffle_data)

        self.encoded_seq = np.transpose(np.array(onehot_encode_sequences(np.array(seq_data)), dtype='float32'), (0, 2, 1))
        #self.encoded_seq = np.stack(np.array(onehot_encode_sequences(seq_datas),dtype='float32'))
        self.test_labels = np.array(test_labels)
        self.dNasedata =np.expand_dims(dNase_data, axis=1)
        print('self.encoded_testing:',self.encoded_seq.shape)
        print('self.dNase_datas:',self.dNasedata.shape)
    def __getitem__(self, index):
        seq_data = self.encoded_seq[index]
        label = self.test_labels[index]
        seq_data = np.expand_dims(seq_data, axis=0)
        dNase_data = self.dNasedata[index]
        return seq_data,dNase_data, label

    def __len__(self):
        return len(self.encoded_seq)



def prepare_all_AcEnhancer_data( training_input_seq,training_input_dNase,trainging_input_label,validation_input_seq,validation_input_dNase,validation_label, batch_size,num_workers, train_supernet=True):
    train_data = trainset(training_input_seq,training_input_dNase,trainging_input_label)
    test_data = testset(validation_input_seq,validation_input_dNase,validation_label)

    
    if train_supernet:
        dataset_size = len(train_data)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        #np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

        return train_loader, valid_loader, test_loader

    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        return train_loader, test_loader
