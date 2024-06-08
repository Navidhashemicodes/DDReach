
from trainer import *
from functions import export2matlab
# from convex_adversarial import *
# from convex_adversarial.dual_network import *

# import sys
# sys.path.append("convex_adversarial/")
# from convex_adversarial import robust_loss


import torch

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time
import gc
import warnings
warnings.filterwarnings('ignore')

import scipy.io
import numpy as np
from scipy.io import savemat
from scipy.linalg import block_diag
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
import hdf5storage


def main():

    print(2020)
    train_batch_size = 600

    transform = transforms.ToTensor()
    
    
    data = loadmat('Data_Main_traj_quad')
    # data = hdf5storage.loadmat('DPL_Data_gym.mat')

    Xtrain = data['X']
    Xtrain = torch.Tensor(np.transpose(Xtrain[:, :99600]))
    # Xtrain = Xtrain.cuda()
    Ytrain = data['Y']
    Ytrain = torch.Tensor(np.transpose(Ytrain[:, :99600]))
    # Ytrain = Ytrain.cuda()
    print(Xtrain.shape)
    print(Ytrain.shape)
    
    
    trainset=TensorDataset(Xtrain, Ytrain)
    # trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=6)
    
    print('Train data loaded')

    ##############################################
    initweights_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Conf_TNN_overall_ss_trajectory_TCPS_robust_TV/Test_cases/Submission/noisy_Quadrotor-problem_submission2/init_weights.mat'
    initbiases_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Conf_TNN_overall_ss_trajectory_TCPS_robust_TV/Test_cases/Submission/noisy_Quadrotor-problem_submission2/init_biases.mat'

    weights_mat = loadmat(initweights_file_path)['W']
    biases_mat = loadmat(initbiases_file_path)['b']
            
    weights = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_mat[0]]
    biases = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_mat[0]]

    initial_params = {}
    for i in range(len(weights)):
        initial_params[f'{2 * i}.weight'] = weights[i]
        BB = biases[i]
        initial_params[f'{2 * i}.bias'] = BB.flatten()

    net = nn.Sequential(
        nn.Linear(12,200),
        nn.ReLU(),
        nn.Linear(200,400),
        nn.ReLU(),
        nn.Linear(400,600)
    )

    net.load_state_dict(initial_params)
    
    
    ################################################
    
    
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    epsilon = 0.02
    verbose = False
    epoch = 285
    print('Training started')


    train_lip_bound(trainloader, net, 9*50, optimizer, epoch, verbose)


    export2matlab('Main_network_quad',net)


if __name__ == '__main__':
    main()
