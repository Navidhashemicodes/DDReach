
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
    
    
    data = loadmat('Quad_Data_traj_stochasticsystem')
    
    
    
    
    Xtrain = data['X']
    Xtrain = torch.Tensor(np.transpose(Xtrain[:, :40200]))
    # Xtrain = Xtrain.cuda()
    Ytrain = data['Y']
    Ytrain = torch.Tensor(np.transpose(Ytrain[:, :40200]))
    # Ytrain = Ytrain.cuda()
    print(Xtrain.shape)
    print(Ytrain.shape)
    
    
    trainset=TensorDataset(Xtrain, Ytrain)
    # trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=6)
    
    print('Train data loaded')
    
    ##############################################
    initweights_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Conf_TNN_overall_ss_trajectory_TCPS_robust_TV/Test_cases/Submission/noisy_Quadrotor-problem_submission_noshift/init_weights.mat'
    initbiases_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Conf_TNN_overall_ss_trajectory_TCPS_robust_TV/Test_cases/Submission/noisy_Quadrotor-problem_submission_noshift/init_biases.mat'

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
    
    # export2matlab('Main_network_Quad_0',net)
    ################################################
    
        
    initweights_file_path = 'C:/Users/navid/Documents/MATLAB/MATLAB_prev/others/Files/CDC2023/Conf_TNN_overall_ss_trajectory_TCPS_robust_TV/Test_cases/Submission/noisy_Quadrotor-problem_submission_noshift/init_alphas.mat'
    weights_mat = loadmat(initweights_file_path)['alphas']
    param2 = nn.Linear(600,1, bias = False)
    weights_torch = torch.from_numpy(weights_mat.astype(np.float32))
    param2.weight.data = weights_torch
    # weight_matrix0 = param2.weight.data.numpy().astype(float)
    # scipy.io.savemat('init_alphas.mat', {'alphas': weight_matrix0})
    ################################################
    
    

    delta = torch.tensor([0.9999])

    all_parameters = list(net.parameters())+list(param2.parameters())
    optimizer = optim.Adam(all_parameters, lr=1e-3)
    epsilon = 0.02
    verbose = False
    epoch = 200
    print('Training started')
    penalty = 1000000
    q = 0.001
    train_lip_bound(trainloader, net,  param2, 100, optimizer, epoch, verbose, delta, penalty, q)


    
    export2matlab('Main_network_Quad_stochasticsystem',net)


        
    # Load the trained weights (replace 'path_to_trained_weights.pth' with the actual path)
        
    # Extract the weight matrix as a NumPy array
    weight_matrix = np.exp(param2.weight.data.numpy().astype(float))

    # Save the weight matrix to a MATLAB .mat file
    scipy.io.savemat('Alpha_params_Quad_stochasticsystem.mat', {'alpha_values': weight_matrix})

    


if __name__ == '__main__':
    main()
