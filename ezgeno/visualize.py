import os
import sys
import time
import math
import numpy as np
import cv2
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from network import ezGenoModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils import *
from dataset import testset

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.cm as cm

from matplotlib.backends.backend_pdf import PdfPages
import heapq
import ast

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        x =self.model.bn[0](x)
        x = x.squeeze(0)
        feature_maps = []
        i=0

        for name, module in self.model.features[0]._modules.items():
            block_id=self.model.arch[2*i]
            connect_id=self.model.arch[2*i+1]
            x = module[block_id](x)
            if connect_id!=0:
                y = feature_maps[connect_id-1]
                x = x + y
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            feature_maps.append(x)
            i=i+1

        x=self.model.globalpooling[0](x)
        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier[0](output)
        return target_activations, output

class GradCam:
    def __init__(self, model, target_layer_names,seq_length, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.seq_length =seq_length
        if self.cuda:
            self.model = model.cuda()
            #only for one input file
            self.model.bn[0] = model.bn[0].cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        cam = np.expand_dims(cam, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (self.seq_length, 1))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam)+0.001)
        return output,cam

class gradCamTestset(Dataset):
    def __init__(self, data_path):

        test_list = []
        with open(data_path) as outputfile: 
            for sequence in outputfile:
                test_list.append(str(sequence.split()[0]))

        test_data = test_list
        test_labels = np.ones(len(test_data), dtype = int)
        
        self.seq_length = len(test_list[0])
        self.test_seq = test_data
        self.encoded_test = np.transpose(np.array(onehot_encode_sequences(np.array(test_data)), dtype='float32'), (0, 2, 1))
        self.test_labels = np.array(test_labels)

    def __getitem__(self, index):
        data = self.encoded_test[index]
        data = np.expand_dims(data, axis=0)
        label = self.test_labels[index]
        seq = self.test_seq[index]
        return data, label ,seq

    def __len__(self):
        return len(self.encoded_test)

def pattern_avg_pooling(x,window):
    pooling = nn.AvgPool1d(window, stride=1,padding=window//2)
    x=x.unsqueeze(0)
    x=pooling(x)
    return x

def get_sub_seq(idx,pos_list,seq_length,window=9):
    j=0
    seq_pos_list_pair=[]
    while j <= len(pos_list)-1:
        left  = max(pos_list[j]-window//2,0)
        right = min(pos_list[j]+window//2,seq_length-1)
        flag=True
        if j==len(pos_list)-1:
            flag=False
        while flag:
            if right > pos_list[j+1]-window//2:
                right= pos_list[j+1]+window//2
                j=j+1
            else:
                flag=False
            if j==len(pos_list)-1:
                flag=False
        j=j+1
        seq_pos_list_pair.append([idx,left,right])

    return seq_pos_list_pair



def write_sub_seq_file(filename,seq_pos_list_pair,pred_list,all_seq_text,chosen_index_list):
    with open(filename,'w') as fout:
        if chosen_index_list is None:   
            for (seq_pos,start,end) in seq_pos_list_pair:         
                seq="".join(all_seq_text[seq_pos,start:end+1])
                fout.write('>seq_{}_{}_{}_{}\n'.format(seq_pos,start,end,pred_list[seq_pos]))
                fout.write(seq+'\n')
        else:
            for (seq_pos,start,end) in seq_pos_list_pair:
                if seq_pos in chosen_index_list:    
                    seq="".join(all_seq_text[seq_pos,start:end+1])
                    fout.write('>seq_{}_{}_{}_{}\n'.format(seq_pos,start,end,pred_list[seq_pos]))
                    fout.write(seq+'\n')

def write_heatmap(filename,mask,seq):
    with PdfPages(filename) as pdf:
        num_of_seqs=mask.shape[0]
        seq_length=mask.shape[1]
        print("num_of_seqs",num_of_seqs)
        print("seq_length",seq_length)
        for i in range(math.ceil(num_of_seqs/100)):
            plt.figure(figsize=(seq_length, 100))
            end=min(i*100+100,num_of_seqs)
            sns.heatmap(mask[i*100:end],fmt='',cmap=cm.Blues, annot=seq[i*100:end,],annot_kws={'size':35},linewidths = 0.02,cbar=False)
            plt.title('Page {}'.format(i+1))
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

def show_grad_cam(args,model_path,dataName,model,use_cuda,window=9):
    print("show_grad_cam")
    fo = open("{}_grad_cam.csv".format(dataName), "w")
    test_data = gradCamTestset(args.data_path)
    #test_data = testset(args.dataSource,testLabelPath,testFileList)
    seq_length=test_data.seq_length
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=8)
    grad_cam = GradCam(model=model, target_layer_names=args.target_layer_names,seq_length=seq_length, use_cuda=True)
    target_index = None
    
    pred_list=[]
    seq_pos_list=[]
    seq_pos_list_pair=[]
    grad_cam_score_list=[]

    all_seq_text=np.empty([len(test_loader), seq_length], dtype = str)
    all_mask=np.zeros((len(test_loader),seq_length))
    for idx, (data, target,seq) in enumerate(test_loader):
        seq_text = np.asarray([[s for s in seq[i]] for i in range(len(seq))])
        output,mask = grad_cam(data, target_index)

        all_seq_text[idx,:]=seq_text
        all_mask[idx,:]=mask

        tensor_mask = torch.Tensor(mask)
        pooling_res=pattern_avg_pooling(tensor_mask,window)
        pooling_res=pooling_res.squeeze()
        pred_list.append(output)
        
        res=np.where(np.array(pooling_res) >= 0.25)
        pos_list=res[0]
        
        seq_pos_list_pair.extend(get_sub_seq(idx,pos_list,seq_length,window))
        seq_pos_list.append(pos_list)

        mask = mask[0].tolist()
        mask = ','.join(['%.3f'%v for v in mask])
        fo.write(mask+'\n')

    fo.close()

    chosen_index_list=None
    if args.show_seq=="all":
        out_mask=all_mask
        out_seq_text=all_seq_text
    
    elif args.show_seq.startswith('top'):  
        num = re.findall(r'top-(\d+)',args.show_seq)[0]
        top_pred_res = heapq.nlargest(num, enumerate(pred_list), key=lambda x:x[1])
        max_pred_index_list,_ = zip(*top_pred_res)
        chosen_index_list=max_pred_index_list
        pred_top_seq=np.empty([num, seq_length], dtype = str)
        pred_top_100_gradcam=np.empty([num, seq_length], dtype = float)
        for i in range(num):
            out_seq_text[i,:]=all_seq_text[max_pred_index_list[i],:]
            out_mask[i,:]=all_mask[max_pred_index_list[i],:]
    else:
        range_list = re.findall(r'(\d+)-(\d+)',args.show_seq)
        range_end =range_list[1]
        range_start = range_list[0]
        out_mask=all_mask[range_start:range_end,:]
        out_seq_text=all_seq_text[range_start:range_end,:]
        chosen_index_list=[i for i in range(start,end)]

    write_heatmap('{}_seq_pos_heatmap.pdf'.format(dataName),out_mask,out_seq_text)
    write_sub_seq_file('{}_sequence_logo.fa'.format(dataName),seq_pos_list_pair,pred_list,all_seq_text,chosen_index_list)

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser("ezGeno")
    parser.add_argument('--show_seq', type=str, default="all", help='all ,top-100 ,30-100')
    parser.add_argument('--load', type=str, default="./model.t7", help='from model to predict seq')
    parser.add_argument('--data_path', type=str, default="../SUZ12/SUZ12_positive_test.fa", help='input data')
    parser.add_argument('--dataName', type=str, default="SUZ12", help='from model to predict seq')
    parser.add_argument('--target_layer_names', type=str, default="[2]", help='want to extract features from target layers')
    parser.add_argument('--use_cuda',help='True or False flag, input should be either "True" or "False".',type=ast.literal_eval, default=True,dest='use_cuda')

    args, unparsed = parser.parse_known_args()
    print(args)

    checkpoint = torch.load(args.load)
    info=checkpoint["info"]
    model=ezGenoModel(dataSource=info["dataSource"],arch=checkpoint["best_arch"],layers=info["layers"], feature_dim=info["feature_dim"],conv_filter_size_list=info["conv_filter_size_list"])
    #self.subnet = ezGenoModel(self.dataSource,self.layers, self.feature_dim,self.conv_filter_size_list,arch=self.best_arch,device=self.device)

    show_grad_cam(args,args.load,args.dataName,model,args.use_cuda)
    end_time = time.time()
    duration = end_time - start_time
    print("total time: %.3fs"%(duration))
