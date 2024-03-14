# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:12:28 2024

@author: 99614
"""
import torch
import torch.nn as nn

class kernelloss(nn.modules):
    def __init__(self,neibor,train_num,device,temperature,):
        super(kernelloss, self).__init__()
        self.train_num = train_num
        self.neibor = neibor
        self.device = device
        self.temperature = temperature
    def forward(self,K):
        K1 = torch.exp(self.neibor*K/self.temperature)
        K2 = torch.exp(K/self.temperature)
        loss = 0
        for i in range(self.train_num):
            for j in range(self.train_num):
                loss = loss + torch.log(K1[i,j]/(torch.sum(torch.K2[i,:])-torch.K2[i,i]))
        return -loss/self.train_num
        
            
        
        
