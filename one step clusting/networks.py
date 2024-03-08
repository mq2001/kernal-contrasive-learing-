# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:45:29 2024

@author: 99614
"""
import torch.nn as nn
import torch
import contrastive_loss

d_view1 = 1024


class Networks(nn.Module):
    def __init__(self, train_num, class_num, views, K0):
        super(Networks, self).__init__()
        self.K = nn.Parameter(torch.Tensor(train_num, train_num,views))
        self.K = K0
        self.E = nn.Parameter(torch.Tensor(class_num,train_num))
        self.E = nn.init.xavier_uniform_(self.E)
        self.E = torch.clamp(self.E,min=0)
        self.G = nn.Parameter(torch.Tensor(class_num,train_num))
        self.G = nn.init.xavier_uniform_(self.G)
        self.G = torch.clamp(self.G,min=0)

    def forward(self):
        K1 = torch.clamp(self.K,min=0)
        u, s, vh = torch.linalg.svd(K1)
        s = torch.clamp(s,min=10**-14)
        K = u@torch.diag(s)@(u.t)
        
        E = torch.clamp(self.E,min=0)
        G = torch.clamp(self.G,min=0)
        
        return K ,E ,G
