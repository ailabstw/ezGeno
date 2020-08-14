import numpy as np
from Bio import SeqIO
import string
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *

class trainset(Dataset):
    def __init__(self, train_pos_data_path,train_neg_data_path, transform=None):

        positive_list = []
        try:
            with open(train_pos_data_path) as outputfile:
                for sequence in outputfile:
                    positive_list.append(str(sequence.split()[0]))
        except:
            print('can not find the training positive data')
        print("the number of training positive sequence: {}".format(len(positive_list)))

        negative_list=[]
        try:
            with open(train_neg_data_path) as outputfile:
                for sequence in outputfile:
                    negative_list.append(str(sequence.split()[0]))
        except:
            print('can not find the training positive data')
        print("the number of training negative sequence: {}".format(len(negative_list)))

        data = negative_list + positive_list
        labels = np.ones(len(data), dtype=int)
        labels[0 : len(negative_list) -1 ] = 0
        shuffle_data = list(zip(data, labels))
        random.shuffle(shuffle_data)
        data, labels = zip(*shuffle_data)

        self.encoded_training = np.transpose(np.array(onehot_encode_sequences(np.array(data)), dtype='float32'), (0, 2, 1))

        self.train_labels = np.array(labels)

        self.transform = transform

    def __getitem__(self, index):
        data = self.encoded_training[index]
        label = self.train_labels[index]
        data = np.expand_dims(data, axis=0)
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.encoded_training)

class testset(Dataset):
    def __init__(self, test_pos_data_path,test_neg_data_path):

        test_pos_list = []
        try:
            with open(test_pos_data_path) as outputfile:
                for sequence in outputfile:
                    test_pos_list.append(str(sequence.split()[0]))
        except:
            print('can not find the testing positive data')
        print("the number of testing positive sequence: {}".format(len(test_pos_list)))

        test_negative_list = []
        try:
            with open(test_neg_data_path) as outputfile:
                for sequence in outputfile:
                    test_negative_list.append(str(sequence.split()[0]))
        except:
            print('can not find the testing negative data')
        print("the number of testing negative sequence: {}".format(len(test_negative_list)))


        test_data = test_pos_list + test_negative_list
        test_labels = np.zeros(len(test_data), dtype = int)
        test_labels[0 : len(test_pos_list) -1 ] = 1
        test_shuffle_data = list(zip(test_data, test_labels))
        test_data, test_labels = zip(*test_shuffle_data)

        self.encoded_test = np.transpose(np.array(onehot_encode_sequences(np.array(test_data)), dtype='float32'), (0, 2, 1))
        self.test_labels = np.array(test_labels)

    def __getitem__(self, index):
        data = self.encoded_test[index]
        label = self.test_labels[index]
        data = np.expand_dims(data, axis=0)
        return data, label

    def __len__(self):
        return len(self.encoded_test)

def prepare_all_data(train_pos_data_path,train_neg_data_path,test_pos_data_path,test_neg_data_path, batch_size, train_supernet=True):
    train_data = trainset(train_pos_data_path,train_neg_data_path)
    test_data = testset(test_pos_data_path,test_neg_data_path)

    if train_supernet:

        dataset_size = len(train_data)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=2)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=100, sampler=valid_sampler, pin_memory=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, num_workers=2)

        return train_loader, valid_loader, test_loader
    
    else:
        train_data = trainset(data_path, transform=transforms.Compose([RandomShift(30)]))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, num_workers=2)
        return train_loader, test_loader