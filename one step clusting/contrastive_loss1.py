# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:46:18 2024
计算类类间的对比损失
@author: 99614
"""

import torch
import torch.nn as nn

class InstanceLoss2(nn.modules):
    def __init__(self, class_num, temperature, device):
        super(InstanceLoss2, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def mask_correlated_samples(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        mask[class_num:,:class_num].fill_diagonal_(0)
        mask[:class_num,class_num:].fill_diagonal_(0)
        mask = mask.bool()
        return mask
#z_i的形状为class_num*sample_num
    def forward(self, z_i, z_j):
        N = 2 * self.class_num
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss