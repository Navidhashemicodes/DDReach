#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:13:26 2020
"""



import torch
import torch.nn as nn
import torchvision as tcn
import numpy as np
import time
from convex_adversarial.dual_network import DualNetwork
import scipy.io
from scipy import sparse


#import matlab.engine
#import sys
#sys.append('../')



def generate_random_net(dims):
    '''
    generate a random fully-connected relu network
    '''

    num_layers = len(dims)-2
    dim_in = dims[0]

    modules = []

    for i in range(0,num_layers):
        param = nn.Linear(dims[i],dims[i+1])
        #param.weight.data = torch.from_numpy(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[i+1],dims[i])))
        param.weight.data = torch.Tensor(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[i+1],dims[i])))
        #param.bias.data = torch.from_numpy(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[i+1],1)))
        #param.bias.data = torch.Tensor(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[i+1],1)))
        modules.append(param)
        modules.append(nn.ReLU())

    param = nn.Linear(dims[-2],dims[-1])
    #param.weight.data = torch.from_numpy(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[-1],dims[-2])))
    #param.bias.data = torch.from_numpy(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[-1],1)))

    param.weight.data = torch.Tensor(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[-1],dims[-2])))
    #param.weight.bias = torch.Tensor(np.random.normal(0,1.0/np.sqrt(dim_in),(dims[-1],1)))

    modules.append(param)
    net = nn.Sequential(*modules)

    return net




def get_weights(net):

    num_layers = int((len(net)-1)/2)

    # network dimensions
    #dim_in = int(net[0].weight.shape[1])
    #dim_out = int(net[-1].weight.shape[0])
    #hidden_dims = [int(net[2*i].weight.shape[0]) for i in range(0,num_layers)]
    #dims = [dim_in] + hidden_dims + [dim_out]

    # get weights
    weights = np.zeros((num_layers+1,), dtype=object)
    weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]


    # get biases
    biases = np.zeros((num_layers+1,), dtype=object)
    biases[:] = [net[2*i].bias.detach().numpy().astype(np.float64).reshape(-1,1) for i in range(0,num_layers+1)]

    return weights,biases


def export_mnist_matlab(normalize=True):
    '''
    export mnist test data to matlab as a struct with fields labels (10000,1) and images (10000,784)
    '''
    image_dim =  784
    #images = np.zeros((num_images,), dtype=object)
    #images[:] = [dataset.test_data[i].detach().numpy().astype(np.float64) for i in range(0,num_images)]

    transform = tcn.transforms.ToTensor()

    dataset = tcn.datasets.MNIST('/tmp', train = False, download=True, transform=transform)
    num_images = dataset.__len__()
    images = dataset.test_data.detach().numpy().astype(np.float64).reshape((num_images,image_dim)).T
    if normalize:
        images = images/255.0
    images = sparse.csr_matrix(images)
    labels = np.zeros((num_images,1))
    labels = dataset.test_labels.detach().numpy().astype(np.float64)

    Xtest = images
    Ytest = labels


    dataset = tcn.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    num_images = dataset.__len__()
    images = dataset.train_data.detach().numpy().astype(np.float64).reshape((num_images,image_dim)).T
    if normalize:
        images = images/255.0
    images = sparse.csr_matrix(images)
    labels = np.zeros((num_images,1))
    labels = dataset.train_labels.detach().numpy().astype(np.float64)
    Xtrain = images
    Ytrain = labels


    data = {}
    data['mnist'] = {'Xtrain': Xtrain, 'Ytrain': Ytrain,'Xtest': Xtest, 'Ytest': Ytest}

    #testset = {}
    #testset['mnist'] = {'Xtest': images, 'Ytest':labels}

    scipy.io.savemat('mnist_data' + '.mat', data)


def export2matlab(file_name,net,save_model=False):
    '''
    Export pytorch fully connected network to matlab
    '''

    num_layers = int((len(net)-1)/2)
    dim_in = float(net[0].weight.shape[1])
    dim_out = float(net[-1].weight.shape[0])
    hidden_dims = [float(net[2*i].weight.shape[0]) for i in range(0,num_layers)]

    # network dimensions
    dims = [dim_in] + hidden_dims + [dim_out]

    # get weights
    weights = np.zeros((num_layers+1,), dtype=object)
    weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]


    # get biases
    biases = np.zeros((num_layers+1,), dtype=object)
    biases[:] = [net[2*i].bias.detach().numpy().astype(np.float64).reshape(-1,1) for i in range(0,num_layers+1)]

    activation = str(net[1])[0:-2].lower()

    # export network data to matlab
    data = {}
    data['net'] = {'weights': weights,'biases':biases, 'dims': dims, 'activation': activation, 'name': file_name}

    scipy.io.savemat(file_name + '.mat', data)

    if save_model:
        torch.save(net, file_name + '.pt')


def pre_activation_bounds(model, X, epsilon):
    '''
    This function compute the bounds on the activation values of the neural
    network model assuming that the input to the neural net is L_inf ball with
    center X and radius epsilon
    Parameters
    ----------
    model : pytorch nn sequential
        a fully connected neural network.
    X : tensor
        center of the input region.
    epsilon : tensor
        radius of the input region (in l_inf norm).
    Returns
    -------
    l : tensor
        lower bound on the activation values.
    u : tensor
        upper bounds on the activation values.
    '''

    dual = DualNetwork(model, X, epsilon,
                       proj=None, # exact bound
                       norm_type='l1', # no random projections
                       bounded_input=True) #[0,1] pixel range
    l,u = [], []
    for dual_layer in dual.dual_net:
        if 'DualReLU' in str(dual_layer.__class__):
            l.append(dual_layer.zl)
            u.append(dual_layer.zu)
    return l,u


def tanh_diff(x):
    y_prim= np.divide( np.power(np.cosh(x),2)-np.power(np.sinh(x),2) ,  np.power(np.cosh(x),2)  )
    return y_prim

def next_layer_prebound(W,b,l_pre,u_pre):
    '''  Write the function '''

    return  l_next,u_next