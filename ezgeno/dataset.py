import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import onehot_encode_sequences
import csv

class trainset(Dataset):
    def __init__(self, data_source, label_file_name, feaure_file_list):

        i=0
        data=[]

        for file_name in feaure_file_list:
            print("file_name", file_name)
            data.append([])
            if data_source[i]==1:
                #read seq file 
                with open(file_name, 'r') as f:
                    for sequence in f:
                        data[i].append(str(sequence.split()[0]))
                print(len(data[i]))
            else:
                #read other file 
                with open(file_name, 'r') as f:
                    reader = csv.reader(f, delimiter=" ")
                    tmp = np.array(list(reader))
                    tmp = np.delete(tmp, -1, axis=1).astype(float)
                    data[i]=tmp
            i+=1

        self.labels = np.loadtxt(label_file_name).astype(int)
        indices = np.arange(len(self.labels)).astype(int)
        np.random.shuffle(indices)
        self.labels = self.labels[indices]
        
        self.np_data=[]

        i=0
        for i in range(len(data_source)):
            if data_source[i]==1:
                self.np_data.append(np.transpose(np.array(onehot_encode_sequences(np.array(data[i])), dtype='float32')[indices], (0, 2, 1)))
            else:
                data[i] = data[i][indices]
                self.np_data.append(np.expand_dims(data[i], axis=1))
            print(self.np_data[i].shape)
        
    def __getitem__(self, index):
        return [self.np_data[i][index] for i in range(len(self.np_data))], self.labels[index]

    def __len__(self):
        return len(self.labels)

class testset(Dataset):
    def __init__(self, data_source, label_file_name, feaure_file_list):

        i=0
        data=[]

        for file_name in feaure_file_list:
            print("file_name", file_name)
            data.append([])
            if data_source[i]==1:
                #read seq file 
                with open(file_name,'r') as f:
                    for sequence in f:
                        data[i].append(str(sequence.split()[0]))
                print(len(data[i]))
            else:
                #read other file 
                with open(file_name, 'r') as f:
                    reader = csv.reader(f, delimiter=" ")
                    tmp = np.array(list(reader))
                    tmp = np.delete(tmp, -1, axis=1).astype(float)
                    data[i]=tmp
            i+=1

        self.labels = np.loadtxt(label_file_name).astype(int)
        indices = np.arange(len(self.labels)).astype(int)
        np.random.shuffle(indices)
        self.labels=self.labels[indices]
        
        self.np_data=[]

        i=0
        for i in range(len(data_source)):
            if data_source[i]==1:
                self.np_data.append(np.transpose(np.array(onehot_encode_sequences(np.array(data[i])), dtype='float32')[indices], (0, 2, 1)))
            else:
                data[i] = data[i][indices]
                self.np_data.append(np.expand_dims(data[i], axis=1))
            print(self.np_data[i].shape)
        
    def __getitem__(self, index):
        return [self.np_data[i][index] for i in range(len(self.np_data))], self.labels[index]

    def __len__(self):
        return len(self.labels)


def prepare_all_data(train_file_list_str, train_label_path, test_file_list_str, test_label_path, batch_size, num_workers, evaluate, train_supernet=True):

    if evaluate:
        test_file_list = test_file_list_str.split(',')
        data_source = [1 if test_file_list[i].endswith('.sequence') else 0 for i in range(len(test_file_list))]
        print("data_source", data_source)
        test_data = testset(data_source, test_label_path, test_file_list)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        return test_loader, data_source
    else:
        train_file_list = train_file_list_str.split(',')
        test_file_list = test_file_list_str.split(',')
        # seq=>1, 1Dvector=>0
        data_source = [1 if train_file_list[i].endswith('.sequence') else 0 for i in range(len(train_file_list))]
        print("data_source", data_source)
        print("type(train_file_list)", len(train_file_list))
        print("type(test_file_list)", len(test_file_list))
    
        train_data = trainset(data_source, train_label_path, train_file_list)
        test_data = testset(data_source, test_label_path, test_file_list)
    
        if train_supernet:
    
            dataset_size = len(train_data)
            indices = list(range(dataset_size))
            split = int(np.floor(0.2 * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
    
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=num_workers)
            valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
            return train_loader, valid_loader, test_loader, data_source
        
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
            return train_loader, test_loader, data_source
    