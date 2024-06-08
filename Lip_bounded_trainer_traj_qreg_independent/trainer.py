#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:48:15 2020

@author: mahyarfazlyab
"""

#from convex_adversarial.dual_network import DualNetwork


import torch

from torch.autograd import Variable

import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time
#import gc
# from convex_adversarial import robust_loss, robust_loss_parallel
from convex_adversarial.dual_network import RobustBounds
from convex_adversarial import DualNetwork
import warnings
warnings.filterwarnings('ignore')

import scipy.io
#import numpy as np
#from scipy.io import savemat
#from scipy.linalg import block_diag
from scipy import sparse



def navid_residual(out, y, model, delta, q):
    

    print('-----------------------------------')

    ttt = y.shape
    tt = ttt[0]
    NN = ttt[1]
    residual = torch.abs(out-y)
    weights = model.weight
    weights_abs = torch.exp(weights)
    
    r_scale = residual*weights_abs 
    R_max = torch.max( r_scale , dim=1 ).values.reshape([1,tt])
    
    
    u_neg = 0.5*(1+torch.sign(R_max-q))
    print('******')
    print([torch.max(u_neg), torch.min(u_neg), torch.mean(u_neg)])      
    print('******')    
    # check = torch.mean(u_neg)
    
    L3 =  torch.sum(q/weights_abs.unsqueeze(1)  , dim=2)
    
    
    Loss2 = torch.mean(delta*nn.ReLU()(R_max - q)+(1-delta)*nn.ReLU()(q -R_max), dim =1)
    
    
    return L3, Loss2







def train_lip_bound(loader, model, param2, lam, opt, epoch, verbose, delta, penalty, q):

    '''
    Train a neural net by constraining the lipschitz constant of each layer
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()
    param2.train()
    end = time.time()
    for t in range(epoch):
        for i, (X,y) in enumerate(loader):
            #X,y = X.cuda(), y.cuda()
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)
            data_time.update(time.time() - end)

            out = model(Variable(X))
            
            ce1_1, ce1_2 = navid_residual(out, Variable(y), param2, delta, q)
            ce2 = nn.MSELoss()(out, Variable(y))
            print([ce1_1, ce1_2, ce2])
            
            ce = ce1_1 + penalty*ce1_2

            
            

            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

            opt.zero_grad()
            ce.backward()
            
            
            
            opt.step()

            num_layers = int((len(model)-1)/2)
            for c in range(num_layers+1):
                scale = max(1,np.linalg.norm(model[2*c].weight.data,2)/lam)
                model[2*c].weight.data = model[2*c].weight.data/scale


            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(ce.item(), X.size(0))
            errors.update(err, X.size(0))
        
        print('epoch: ',t,'CrossEntropyLoss1: ',ce.item())

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
               epoch, i, len(loader), batch_time=batch_time,
               data_time=data_time, loss=losses, errors=errors))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def robust_loss(net, epsilon, X, y,
                size_average=True, device_ids=None, parallel=False, **kwargs):
    if parallel:
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else:
        f = RobustBounds(net, epsilon, **kwargs)(X,y)
    err = (f.max(1)[1] != y)
    if size_average:
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err

def evaluate_baseline(loader, model):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        #X,y = X.cuda(), y.cuda()
        #out = model(Variable(X))
        TEST_SIZE = X.shape[0]
        #out = model(Variable(X.view(TEST_SIZE, -1)))
        out = model(X.view(TEST_SIZE, -1))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
    return errors.avg