import torch
import torch.nn as nn

class ezGenoModel(nn.Module):
    def __init__(self, data_source, layers, feature_dim, conv_filter_size_list, arch=None, device='cpu'):
        super(ezGenoModel, self).__init__()

        arch_count=0
        self.arch=arch
        self.arch_list=[]

        self.data_source = data_source
        self.layers = layers
        self.feature_dim = feature_dim
        self.conv_filter_size_list = conv_filter_size_list

        self.bn= [nn.BatchNorm2d(1).to(device) if self.data_source[i]==1 else nn.BatchNorm1d(1).to(device) for i in range(len(self.data_source))]
        self.globalpooling= [ nn.AdaptiveMaxPool1d(1).to(device) for i in range(len(self.data_source))]

        self.channels=[]
        self.features= torch.nn.ModuleList()

        for i in range(len(self.layers)):
            if arch is not None:
                self.arch_list.append(arch[2*arch_count:2*(arch_count+layers[i])])
            else:
                self.arch_list.append([])
            arch_count+=self.layers[i]

        #search space
        for i in range(len(self.layers)):
            self.channels.append([])
            self.features.append(torch.nn.ModuleList())
            if self.data_source[i]==1:
                self.channels[i]=[4]+ [self.feature_dim[i] for j in range(self.layers[i])]
            else:
                self.channels[i]=[1]+ [self.feature_dim[i] for j in range(self.layers[i])]
            for j in range(self.layers[i]):
                self.features[i].append(torch.nn.ModuleList())
                input_channel = self.channels[i][j]
                output_channel = self.channels[i][j+1]
                for filter_size in self.conv_filter_size_list[i]:
                    self.features[i][-1].append(nn.Sequential(
                        nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1)//2),
                        nn.ReLU(),
                    ))
                    self.features[i][-1].append(nn.Sequential(
                        nn.Conv1d(input_channel, output_channel, filter_size, stride=1, padding=(filter_size-1), dilation=2),
                        nn.ReLU(),
                    ))
                if self.arch_list[i] != []:
                    for k in range(len(self.features[i][-1])):
                        if k != self.arch_list[i][2*j]:
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

        for i in range(len(x)):
            feature_maps.append([])
            if self.data_source[i]==1:
                x[i] = x[i].unsqueeze(1)
                x[i]=self.bn[i](x[i])
                x[i]=x[i].squeeze()
            else:
                x[i]=self.bn[i](x[i])
            blocks = [arch[2*(j+arch_count)] for j in range(self.layers[i])]
            connections = [arch[2*(j+arch_count)+1] for j in range(self.layers[i])]
            arch_count=arch_count+self.layers[i]

            for _, (archs, block_id, connect_id) in enumerate(zip(self.features[i], blocks, connections)):
                x[i] = archs[block_id](x[i])
                if connect_id!=0:
                    y = feature_maps[i][connect_id-1]
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
