import numpy as np
from Bio import SeqIO
import string
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import csv

class trainset(Dataset):
    def __init__(self,dataSource, labelFileName,FeaureFileList):

        i=0
        data=[]

        for fileName in FeaureFileList:
            print("fileName",fileName)
            data.append([])
            if dataSource[i]==1:
                #read seq file 
                with open(fileName,'r') as f:
                    for sequence in f:
                        data[i].append(str(sequence.split()[0]))
                print(len(data[i]))
            else:
                #read other file 
                with open(fileName, 'r') as f:
                    reader = csv.reader(f, delimiter=" ")
                    tmp = np.array(list(reader))
                    tmp = np.delete(tmp, -1, axis=1).astype(float)
                    data[i]=tmp
            i+=1

        self.labels =np.loadtxt(labelFileName).astype(int)
        indices = np.arange(len(self.labels)).astype(int)
        np.random.shuffle(indices)
        self.labels=self.labels[indices]
        
        self.np_data=[]

        i=0
        for i in range(len(dataSource)):
            if dataSource[i]==1:
                self.np_data.append(np.transpose(np.array(onehot_encode_sequences(np.array(data[i])), dtype='float32')[indices], (0, 2, 1)))
            else:
                data[i]=data[i][indices]
                self.np_data.append(np.expand_dims(data[i], axis=1))
            print(self.np_data[i].shape)
        
    def __getitem__(self, index):
        return [self.np_data[i][index] for i in range(len(self.np_data))],self.labels[index]

    def __len__(self):
        return len(self.labels)

class testset(Dataset):
    def __init__(self,dataSource, labelFileName,FeaureFileList):

        i=0
        data=[]

        for fileName in FeaureFileList:
            print("fileName",fileName)
            data.append([])
            if dataSource[i]==1:
                #read seq file 
                with open(fileName,'r') as f:
                    for sequence in f:
                        data[i].append(str(sequence.split()[0]))
                print(len(data[i]))
            else:
                #read other file 
                with open(fileName, 'r') as f:
                    reader = csv.reader(f, delimiter=" ")
                    tmp = np.array(list(reader))
                    tmp = np.delete(tmp, -1, axis=1).astype(float)
                    data[i]=tmp
            i+=1

        self.labels =np.loadtxt(labelFileName).astype(int)
        indices = np.arange(len(self.labels)).astype(int)
        np.random.shuffle(indices)
        self.labels=self.labels[indices]
        
        self.np_data=[]

        i=0
        for i in range(len(dataSource)):
            if dataSource[i]==1:
                self.np_data.append(np.transpose(np.array(onehot_encode_sequences(np.array(data[i])), dtype='float32')[indices], (0, 2, 1)))
            else:
                data[i]=data[i][indices]
                self.np_data.append(np.expand_dims(data[i], axis=1))
            print(self.np_data[i].shape)
        
    def __getitem__(self, index):
        return [self.np_data[i][index] for i in range(len(self.np_data))],self.labels[index]

    def __len__(self):
        return len(self.labels)


def prepareAllData(trainFileListStr,trainLabelPath,testFileListStr,testLabelPath,batch_size,num_workers,evaluate, train_supernet=True):

    if evaluate:
        testFileList=testFileListStr.split(',')
        dataSource= [1 if testFileList[i].endswith('.sequence') else 0 for i in range(len(testFileList))]
        print("dataSource",dataSource)
        test_data = testset(dataSource,testLabelPath,testFileList)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
        return test_loader,dataSource
    else:
        trainFileList=trainFileListStr.split(',')
        testFileList=testFileListStr.split(',')
        # seq => 1 ,1Dvector=>0
        dataSource= [1 if trainFileList[i].endswith('.sequence') else 0 for i in range(len(trainFileList))]
        print("dataSource",dataSource)
        print("type(trainFileList)",len(trainFileList))
        print("type(testFileList)",len(testFileList))
    
        train_data = trainset(dataSource,trainLabelPath,trainFileList)
        test_data = testset(dataSource,testLabelPath,testFileList)
    
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
    
            return train_loader, valid_loader, test_loader ,dataSource
        
        else:
    
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
            return train_loader, test_loader,dataSource
    