# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:45:36 2024

@author: 99614
"""
import numpy as np
import torch
import scipy.io
import scipy.io as sio
import warnings
import constrastive_loss
import networks

#数据导入
filepath = ''   #文件路径
dataset = sio.loadmat(filepath) 
sample = torch.tensor(dataset['X'],dtype=torch.float) #导入样本,形状为样本数*样本特征数
label  = np.squeeze(dataset['Y']) #导入样本的实际标签以验证准确率
warnings.filterwarnings("ignore")

#训练准备
loss_device = torch.device('cuda')

