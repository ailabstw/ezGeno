import numpy as np
from Bio import SeqIO
import string
import random
import time
import argparse
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from utils import *
from controller import Controller
from network import ezGenoModel
import json
#from network import ezGenoModel,AcEnhancerModel

class ezGenoTrainer():
    def __init__(self, args,dataSource):

        self.supernet_epochs = args.supernet_epochs
        self.cstep = args.cstep
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.save = args.save
        self.best_arch = None
        self.subnet = None
        
        self.weight_decay=args.weight_decay
        self.momentum=args.momentum
        self.optimizer=args.optimizer
        self.controller_optimizer=args.controller_optimizer

        self.device = 'cpu' if args.cuda==-1 else 'cuda:%d'%args.cuda
        self.dataSource = dataSource
        if not args.feature_dim:
            self.feature_dim=[64 for i in range(len(self.dataSource)) ]
        else:
            self.feature_dim=args.feature_dim

        if not args.layers:
            self.layers=[6 for i in range(len(self.dataSource)) ]
        else:
            self.layers=args.layers

        if not args.conv_filter_size_list:
            self.conv_filter_size_list=[[3, 7, 11, 15, 19] if self.dataSource[i]==1 else [3,7,11]for i in range(len(self.dataSource)) ]
        else:
            self.conv_filter_size_list=np.array(json.loads(args.conv_filter_size_list))

        self.num_conv_choice_list=[2*len(self.conv_filter_size_list[i]) for i in range(len(self.dataSource)) ]


        print("self.layers",self.layers)
        print("self.conv_filter_size_list",self.conv_filter_size_list)
        print("self.num_conv_choice_list",self.num_conv_choice_list)
        print("self.feature_dim",self.feature_dim)

        if args.eval and os.path.isfile(args.load):
            print("loading model {}".format(args.load))
            self.load_model(args)
            
        else:
            print("no checkpoint found.")
            self.supernet = ezGenoModel(self.dataSource,self.layers, self.feature_dim,self.conv_filter_size_list,device=self.device)
            self.controller = Controller(args, self.layers,self.num_conv_choice_list)
        
        self.criterion = nn.BCELoss()
        self.supernet_optimizer = choose_optimizer(self.optimizer,self.supernet,args.supernet_learning_rate,[self.weight_decay,self.momentum])
        self.num_choices = []

        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                self.num_choices.append(self.num_conv_choice_list[i])
                self.num_choices.append(j+1)
        print("self.num_choices",self.num_choices)
        self.get_random_cand = lambda:tuple(np.random.randint(i) for i in self.num_choices)
        self.controller_optimizer = choose_optimizer(self.controller_optimizer,self.controller,args.controller_learning_rate,[self.weight_decay,self.momentum])
        
        if self.subnet is not None:
            self.subnet_optimizer = choose_optimizer(self.optimizer,self.subnet,self.learning_rate,[self.weight_decay,self.momentum])
        else:
            self.subnet_optimizer = None
        self.info={'layers':self.layers,'feature_dim':self.feature_dim,'conv_filter_size_list':self.conv_filter_size_list}


    def train_supernet(self,model, train_loader, optimizer, criterion, epoch, get_random_cand=None, arch=None):
        print("Epoch {:d}".format(epoch))
        model.train()
        correct = 0
        all_label = []
        all_pred = []
        for (data, target) in train_loader:
            target = target.float().to(self.device)
            data_Device= [ data[i].float().to(self.device) for i in range(len(data)) ]
            optimizer.zero_grad()
            if get_random_cand is not None:
                output = model(data_Device, arch=get_random_cand())
            elif arch is not None:
                output = model(data_Device, arch=arch)
            else:
                output = model(data_Device)
            pred = output>0.5
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_label.extend(target.reshape(-1).tolist())
            all_pred.extend((output[:]).reshape(-1).tolist())

            output = output.reshape(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print("Training acc: %.2f"%(100. * correct/len(all_pred))+"%")
        print("Train AUC score: {:.4f}".format(roc_auc_score(np.array(all_label), np.array(all_pred))))
    
    def train_controller(self,cstep, model, controller, valid_loader, controller_optimizer):
        controller.train()
        model.eval()
        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        total_loss = 0
        controller_step = 0

        while controller_step < cstep:
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                # sample models
                if controller_step>=cstep:
                    break
                arch, log_probs, entropies = controller.sample(with_details=True)

                # calculate reward
                np_entropies = entropies.data.cpu().numpy()
                with torch.no_grad():
                    targets = targets.float().to(self.device)
                    inputs_Device= [ inputs[i].float().to(self.device) for i in range(len(inputs)) ]
                    outputs = model(inputs_Device, arch=arch)
                    predicted = outputs>0.5
                    try:
                        rewards = roc_auc_score(np.array(targets.reshape(-1).tolist()), np.array(outputs.reshape(-1).tolist()))
                    except:
                        print("AUC error.")
                reward_history.append(rewards)
                entropy_history.append(np_entropies)

                # moving average baseline
                if baseline is None:
                    baseline = rewards
                else:
                    decay = 0.95
                    baseline = decay * baseline + (1 - decay) * rewards

                adv = rewards - baseline
                adv_history.append(adv)

                # policy loss
                loss = -log_probs*torch.tensor(adv).to(self.device)
                loss = loss.sum()

                # update
                controller_optimizer.zero_grad()
                loss.backward()
                controller_optimizer.step()
                total_loss += loss.item()
                controller_step += 1

            print("controller_step :{} total loss: {}".format(controller_step, total_loss))

    def train_subnet(self,model, train_loader, optimizer, criterion, epoch, arch):
        print("Epoch {:d}".format(epoch))
        model.train()
        correct = 0
        all_label = []
        all_pred = []
        for (data, target) in train_loader:
            target = target.float().to(self.device)
            data_Device= [ data[i].float().to(self.device) for i in range(len(data)) ]
            
            optimizer.zero_grad()
            output = model(data_Device, arch=arch)
            pred = output>0.5
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_label.extend(target.reshape(-1).tolist())
            all_pred.extend((output[:]).reshape(-1).tolist())

            output = output.reshape(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print("Training acc: %.2f"%(100. * correct/len(all_pred))+"%")
        print("Train AUC score: {:.4f}".format(roc_auc_score(np.array(all_label), np.array(all_pred))))

    
    def test(self,model, test_loader):
        model.eval()
        model.to(self.device)
        correct = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_label = []
        all_pred = []
        for batch_idx, (data, target) in enumerate(test_loader):
            target = target.float().to(self.device)
            data_Device= [ data[i].float().to(self.device) for i in range(len(data)) ]         
            
            output = model(data_Device, arch=self.best_arch)
            pred = output>0.5
            correct += pred.eq(target.view_as(pred)).sum().item()
            for p, t in zip(pred, target.view_as(pred)):
                if p.eq(t) and p.item()==1:
                    tp += 1
                elif  p.eq(t) and p.item()==0:
                    tn += 1
                elif p.item()==1:
                    fp += 1
                else:
                    fn += 1
            all_label.extend(target.reshape(-1).tolist())
            all_pred.extend((output[:]).reshape(-1).tolist())

        print("Test AUC score: {:.4f}\n".format(roc_auc_score(np.array(all_label), np.array(all_pred))))
        return roc_auc_score(np.array(all_label), np.array(all_pred))

    def train(self, train_loader, valid_loader, test_loader,enable_stage_1=True, enable_stage_2=True, enable_stage_3=True):
        ''' stage 1: train supernet '''
        if enable_stage_1:
            self.supernet = self.supernet.to(self.device)
            for epoch in range(1, self.supernet_epochs):
                self.train_supernet( self.supernet , train_loader, self.supernet_optimizer, self.criterion, epoch, self.get_random_cand)

        ''' stage 2: train controller '''
        if enable_stage_2:
            self.supernet = self.supernet.to(self.device)
            self.controller = self.controller.to(self.device)

            self.train_controller(self.cstep, self.supernet, self.controller, valid_loader, self.controller_optimizer)
            
            self.best_arch = self.controller.sample(is_train=False)
            print("self.best_arch",self.best_arch)

            self.subnet = ezGenoModel(self.dataSource,self.layers, self.feature_dim,self.conv_filter_size_list,device=self.device,arch=self.best_arch)
            self.subnet_optimizer = choose_optimizer(self.optimizer,self.subnet,self.learning_rate,[self.weight_decay,self.momentum])

        ''' stage 3: train from scratch '''
        if enable_stage_3:
            if self.subnet is None:
                raise Exception("please specify a network architecture or run stage 2 to search for best arch.")
            else:
                self.subnet = self.subnet.to(self.device)
                model_check_epoch = copy.deepcopy(self.subnet)
                optimizer_check_epoch = choose_optimizer(self.optimizer,model_check_epoch,self.learning_rate,[self.weight_decay,self.momentum])
                valid_auc_list=[]
                best_valid_auc=0
                for epoch in range(1, self.epochs):
                    self.train_subnet(model_check_epoch, train_loader, optimizer_check_epoch, self.criterion, epoch, arch=self.best_arch)
                    valid_auc = self.test(model_check_epoch, valid_loader)
                    valid_auc_list.append(valid_auc)
                    print("valid_auc:",valid_auc)
                    if valid_auc>best_valid_auc:
                        best_epoch=epoch
                        best_valid_auc = valid_auc
                    try:
                        if (valid_auc_list[-3] - valid_auc_list[-1]) > 0.003:
                            break
                    except:
                        pass

                print("check_valid_epoch: {}".format(epoch))
                print("best_epoch: {}".format(best_epoch))

                for epoch in range(1, best_epoch):
                    self.train_subnet(self.subnet, train_loader, self.subnet_optimizer, self.criterion, epoch, arch=self.best_arch)
                    self.train_subnet(self.subnet, valid_loader, self.subnet_optimizer, self.criterion, epoch, arch=self.best_arch)
                    print("test auc:", self.test(self.subnet, test_loader))
        self.save_model(self.info)

    def load_model(self, args):
        checkpoint = torch.load(args.load)
        self.best_arch= checkpoint["best_arch"]
        self.info= checkpoint["info"]
        self.feature_dim = self.info["feature_dim"]
        self.conv_filter_size_list= self.info["conv_filter_size_list"]
        self.layers= self.info["layers"]

        self.supernet = ezGenoModel(self.dataSource,self.layers, self.feature_dim,self.conv_filter_size_list,device=self.device)
        self.controller = Controller(args,self.layers, self.num_conv_choice_list)

        self.load_supernet(checkpoint)
        self.load_controller(checkpoint)
        
        if self.best_arch is not None:
            self.subnet = ezGenoModel(self.dataSource,self.layers, self.feature_dim,self.conv_filter_size_list,arch=self.best_arch,device=self.device)
        self.load_subnet(checkpoint)

    def save_model(self,info):
        torch.save({
            'best_arch': self.best_arch, 
            'info':info,
            'supernet_state_dict': self.supernet.state_dict(), 
            'controller_state_dict': self.controller.state_dict(),
            'subnet_state_dict': self.subnet.state_dict() if self.subnet is not None else None
            }, self.save)

    def load_supernet(self,checkpoint):
        try:
            self.supernet.load_state_dict(checkpoint["supernet_state_dict"])
        except:
            print("fail to load supernet. please check whether the setting is the same as the checkpoint.")

    def load_controller(self,checkpoint):
        try:
            self.controller.load_state_dict(checkpoint["controller_state_dict"])
        except:
            print("fail to load controller. please check whether the setting is the same as the checkpoint.")
   
    def load_subnet(self,checkpoint):
        if checkpoint["subnet_state_dict"] is not None:
            try:
                self.subnet.load_state_dict(checkpoint["subnet_state_dict"])
            except:
                print("fail to load subnet. please check whether the setting is the same as the checkpoint.")
