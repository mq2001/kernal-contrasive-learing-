# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:46:58 2024
计算样本间的对比损失。
@author: 99614
"""
import torch
import torch.nn as nn

class InstanceLoss1(nn.modules):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss1, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        mask[batch_size:,:batch_size].fill_diagonal_(0)
        mask[:batch_size,batch_size:].fill_diagonal_(0)
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
