import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ezGenoModel(nn.Module):
    def __init__(self, arch=None, layers=6, feature_dim=128,conv_filter_size_list=[3, 7, 11, 15, 19]):
        super(ezGenoModel, self).__init__()

        self.conv_filter_size_list=conv_filter_size_list
        self.feature_dim = feature_dim
        self.layers = layers
        self.channels = [4] + [self.feature_dim for i in range(self.layers)]
        self.features = torch.nn.ModuleList()
        self.bn = nn.BatchNorm2d(1)
        self.num_conv_choice=len(self.conv_filter_size_list)*2
        if arch is not None:
            self.arch = arch

        for i in range(self.layers):
            self.features.append(torch.nn.ModuleList())
            input_channel = self.channels[i]
            output_channel = self.channels[i+1]

            for filter_size in self.conv_filter_size_list:
                self.features[-1].append(nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1)//2),
                    nn.ReLU(),
                    ))
                self.features[-1].append(nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1), dilation=2),
                    nn.ReLU(),
                    ))

            if arch is not None:
                for j in range(len(self.features[-1])):
                    if j != self.arch[2*i]:
                        self.features[-1][j]=None
        
        self.dropout = nn.Dropout(p=0.5)
        self.globalpooling = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, arch=None):

        x = self.bn(x)
        x = x.squeeze()
        feature_maps = []
        blocks = [arch[2*i] for i in range(self.layers)]
        connections = [arch[2*i+1] for i in range(self.layers)]
        for idx, (archs, block_id, connect_id) in enumerate(zip(self.features, blocks, connections)):
            x = archs[block_id](x)
            if connect_id!=0:
                y = feature_maps[connect_id-1]
                x = x + y
            feature_maps.append(x)


        x = self.dropout(x)
        x = self.globalpooling(x)
        x = x.reshape(-1, self.feature_dim)
        output = self.classifier(x)
        return output




class AcEnhancerModel(ezGenoModel):
    def __init__(self, arch=None, layers=6, feature_dim=64,conv_filter_size_list=[3, 7, 11, 15, 19],dNase_layers=6,dNase_feature_dim=64,dNase_conv_filter_size_list=[3, 7, 11]):

        self.feature_dim=feature_dim
        self.layers=layers
        if arch is not None:
            self.seq_arch=arch[0:2*self.layers]
            self.dNase_arch=arch[2*self.layers:]
        else:
            self.seq_arch=None
            self.dNase_arch=None
        super(AcEnhancerModel, self).__init__(self.seq_arch,self.layers,self.feature_dim,conv_filter_size_list)
        

        self.dNase_conv_filter_size_list=dNase_conv_filter_size_list
        self.dNase_feature_dim = dNase_feature_dim
        self.dNase_layers = dNase_layers
        self.dNase_channels = [1] + [self.dNase_feature_dim for i in range(self.dNase_layers)]
        self.dNase_features = torch.nn.ModuleList()
        self.dNase_bn = nn.BatchNorm1d(1)
        #print("self.dNase_conv_filter_size_list",dNase_conv_filter_size_list)
        self.dNase_num_conv_choice=len(self.dNase_conv_filter_size_list)*2

        for i in range(self.dNase_layers):
            self.dNase_features.append(torch.nn.ModuleList())
            input_channel = self.dNase_channels[i]
            output_channel = self.dNase_channels[i+1]

            for filter_size in self.dNase_conv_filter_size_list:
                self.dNase_features[-1].append(nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1)//2),
                    nn.ReLU(),
                    ))
                self.dNase_features[-1].append(nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1), dilation=2),
                    nn.ReLU(),
                    ))

            if self.dNase_arch is not None:
                for j in range(len(self.dNase_features[-1])):
                    if j != self.dNase_arch[2*i]:
                        self.dNase_features[-1][j]=None

        self.dropout = nn.Dropout(p=0.5)
        self.dNase_globalpooling = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim+self.dNase_feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1,x2, arch=None):
        
        x1 = self.bn(x1)
        x1 = x1.squeeze()
        feature_maps = []
        blocks = [arch[2*i] for i in range(self.layers)]
        connections = [arch[2*i+1] for i in range(self.layers)]

        for idx, (archs, block_id, connect_id) in enumerate(zip(self.features, blocks, connections)):
            x1 = archs[block_id](x1)
            if connect_id!=0:
                y = feature_maps[connect_id-1]
                x1 = x1 + y
            feature_maps.append(x1)
        x1 = self.dropout(x1)
        x1 = self.globalpooling(x1)
        x1 = x1.reshape(-1, self.feature_dim)

        x2 = x2.float()
        x2 = self.dNase_bn(x2)

        dNase_feature_maps = []
        dNase_arch=arch[self.layers*2::]
        blocks = [dNase_arch[2*i] for i in range(self.dNase_layers)]
        connections = [dNase_arch[2*i+1] for i in range(self.dNase_layers)]
        for idx, (archs, block_id, connect_id) in enumerate(zip(self.dNase_features, blocks, connections)):
            x2 = archs[block_id](x2)
            if connect_id!=0:
                y = dNase_feature_maps[connect_id-1]
                x2 = x2 + y
            dNase_feature_maps.append(x2)
        x2 = self.dropout(x2)
        x2 = self.globalpooling(x2)
        x2 = x2.reshape(-1, self.dNase_feature_dim)

        x = torch.cat((x1,x2),1)

        output = self.classifier(x)
        return output

