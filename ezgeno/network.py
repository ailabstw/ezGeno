import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ezGenoModel(nn.Module):
    def __init__(self,dataSource, layers, feature_dim,conv_filter_size_list, arch=None,device='cpu'):
        super(ezGenoModel, self).__init__()

        arch_count=0
        self.arch=arch
        self.archList=[]

        self.dataSource = dataSource
        self.layers = layers
        self.feature_dim = feature_dim
        self.conv_filter_size_list = conv_filter_size_list

        self.bn= [nn.BatchNorm2d(1).to(device) if self.dataSource[i]==1 else nn.BatchNorm1d(1).to(device) for i in range(len(self.dataSource))]
        self.globalpooling= [ nn.AdaptiveMaxPool1d(1).to(device) for i in range(len(self.dataSource))]

        self.channels=[]
        #self.features=[]
        self.features= torch.nn.ModuleList()
        #arch


        for i in range(len(self.layers)):
            if arch is not None:
                self.archList.append(arch[2*arch_count:2*(arch_count+layers[i])])
            else:
                self.archList.append([])
            arch_count+=self.layers[i]

        #search space
        for i in range(len(self.layers)):
            self.channels.append([])
            self.features.append(torch.nn.ModuleList())
            #self.features[i].append(torch.nn.ModuleList())
            if self.dataSource[i]==1:
                self.channels[i]=[4]+ [self.feature_dim[i] for j in range(self.layers[i])]
            else:
                self.channels[i]=[1]+ [self.feature_dim[i] for j in range(self.layers[i])]
            for j in range(self.layers[i]):
                self.features[i].append(torch.nn.ModuleList())
                #print("i j",i,j)
                input_channel = self.channels[i][j]
                output_channel = self.channels[i][j+1]
                #print(input_channel,output_channel)
                for filter_size in self.conv_filter_size_list[i]:
                    self.features[i][-1].append(nn.Sequential(
                        nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1)//2),
                        nn.ReLU(),
                    ))
                    self.features[i][-1].append(nn.Sequential(
                        nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1), dilation=2),
                        nn.ReLU(),
                    ))
                #print("len(self.features[i][-1])",len(self.features[i][-1]))
                if self.archList[i] != []:
                    for k in range(len(self.features[i][-1])):
                        #print("k archList[i][2*j]",k,archList[i][2*j])
                        if k != self.archList[i][2*j]:
                            #print(self.features[i])
                            self.features[i][-1][k]=None
        
        self.dropout = nn.Dropout(p=0.5)

        self.totalFeatureDim=sum(i for i in self.feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.totalFeatureDim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, arch=None):
        arch_count=0
        feature_maps = []
        #print("arch",arch)

        for i in range(len(x)):
            feature_maps.append([])
            #feature_maps[i].append([])
            #print(x[i].shape)
            #print("x[i].is_cuda",x[i].is_cuda)
            if self.dataSource[i]==1:
                x[i] = x[i].unsqueeze(1)
                x[i]=self.bn[i](x[i])
                x[i]=x[i].squeeze()
            else:
                x[i]=self.bn[i](x[i])
            #print("self.layers[i]",self.layers[i])
            blocks = [arch[2*(j+arch_count)] for j in range(self.layers[i])]
            connections = [arch[2*(j+arch_count)+1] for j in range(self.layers[i])]
            arch_count=arch_count+self.layers[i]
            #print("i",i)
            #print("blocks",blocks)
            #print("connection",connections)
            #print("self.features[i]",self.features[i])
            for idx, (archs, block_id, connect_id) in enumerate(zip(self.features[i], blocks, connections)):
                x[i] = archs[block_id](x[i])
                if connect_id!=0:
                    #print("connect_id",connect_id)
                    y = feature_maps[i][connect_id-1]
                    #print("i connect_id y ",i,connect_id,y)
                    x[i] = x[i] + y
                feature_maps[i].append(x[i])
            x[i] = self.dropout(x[i])
            x[i] = self.globalpooling[i](x[i])
            x[i] = x[i].reshape(-1, self.feature_dim[i])
            if i==0:
                x_total=x[i]
            else:
                x_total = torch.cat((x_total,x[i]),1)
        
        output = self.classifier(x_total)
        return output
