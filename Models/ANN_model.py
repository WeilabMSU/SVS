#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:17:41 2023

@author: shenli
"""

from __future__ import print_function
import argparse
import sys
import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


#======================================Classes==================================
class Net(nn.ModuleList):
    def __init__(self,in_nn,out_nn,neuron_size,layer_size):
        super(Net, self).__init__()
        
        
        
        self.network_nn=[in_nn]
        for layer in range(layer_size):
            self.network_nn.append(neuron_size)
        self.network_nn.append(out_nn)
        
        
        self.FCs=nn.ModuleList()
        self.RELUs=nn.ModuleList()
        for fc_layer in range(len(self.network_nn)-1):
            linear_temp = nn.Linear(self.network_nn[fc_layer],self.network_nn[fc_layer+1],bias=True)
            nn.init.xavier_uniform_(linear_temp.weight)
            relu_temp = nn.ReLU()
            self.RELUs.append(relu_temp)
            self.FCs.append(linear_temp)
        
    def forward(self, X):
        for i in range(len(self.FCs)-1):
            X = self.FCs[i](X)
            X = self.RELUs[i](X)
            
        X = self.FCs[-1](X)
        y_hat = F.log_softmax(X,dim=1)
        return y_hat

#=================================Training & Testing============================

class ANN:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        
    def feed_para(self,no_cuda,seed,log_interval,test_batch_size,l2_penalty,lr,batch_size,neuron_size,layer_size,epochs):
        self.in_nn = self.X_train.shape[1]
        self.out_nn = 2
        
        self.no_cuda = no_cuda

        self.seed = seed
        self.log_interval = log_interval
        self.alpha = l2_penalty
        self.lr = lr        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.neuron_size = neuron_size
        self.layer_size = layer_size
        self.epochs = epochs
        
    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            
            l2_penalty = self.alpha * sum([torch.norm(param) for param in model.parameters()])
            loss_with_penalty = loss + l2_penalty
            
            optimizer.zero_grad()
            loss_with_penalty.backward()
            optimizer.step()
            '''
            if epoch % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
            '''
    
    def test(self, model, device, epoch, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader.dataset)
        '''
        if epoch % self.log_interval == 0:
            print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
        '''
        return correct/len(test_loader.dataset)
        
    
    def process_training(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        device = torch.device("cuda:0" if use_cuda else "cpu")
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        
        train_data = torch.from_numpy(self.X_train).float()
        test_data = torch.from_numpy(self.X_test).float()

        trainset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(self.y_train.ravel()))
        testset = torch.utils.data.TensorDataset(test_data, torch.from_numpy(self.y_test.ravel()))

        # Define data loader
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.test_batch_size, shuffle=False, **kwargs)
        
        model = Net(self.in_nn, self.out_nn,self.neuron_size,self.layer_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-08, amsgrad=False)
        lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5, last_epoch = -1)
        
        
        for epoch in range(1, self.epochs + 1):
            lr_adjust.step()
            self.train(model, device, train_loader, optimizer, epoch)
            self.test(model, device, epoch, test_loader)
    
            
        return self.test(model,device,epoch,test_loader)

