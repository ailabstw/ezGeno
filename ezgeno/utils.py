import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from collections import defaultdict

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
        
    if cuda==-1:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs.to('cuda:%d'%cuda), **kwargs)
    return out

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def onehot_encode_sequences(sequences):
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            if letter=='N':
                arr[i,:]=0.25
            else:
                arr[i, mapping[letter]] = 1.0
        onehot.append(arr)
    return onehot

def choose_optimizer(optimizerName,model,learning_rate,parameters_list):
    weight_decay=parameters_list[0]
    momentum=parameters_list[1]
    if optimizerName=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizerName=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def output_arch(arch, conv_filter_size_list, layers):
    print("the Architecture of the network we choosed is:")
    print("==============================================")
    arch_count=0
    for i in range(len(layers)):
        for index in range(2*arch_count,2*(arch_count+layers[i])):
            # conv layer
            if (index%2)==0:
                if arch[index]%2==0:
                    print("conv:{}".format(conv_filter_size_list[i][arch[index]//2]))
                else:
                    print("conv:{} + dilation".format(conv_filter_size_list[i][arch[index]//2]))
            # connect layer
            else:
                if arch[index]==0:
                    print("no connection layer added")
                else:
                    print("connect layer {}".format(arch[index]))
        arch_count+=layers[i]
    print("==============================================")