# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:29:36 2024

@author: 99614
"""

import numpy as np

def KNN(dataset,K):
    (m,n) = np.shape(dataset)
    D1 = np.zeros((n,n))
    weights = np.eye(n)
    w_01 = np.zeros((n,n))
    for i in range(0,n-1):
        for j in range(i+1,n):
            D1[i,j] = np.linalg.norm(dataset[:,i]-dataset[:,j],ord=2)
            D1[j,i] = D1[i,j]
    for i in range(0,n):
        
        weights[:,i] = np.exp(-D1[:,i]/(np.linalg.norm(D1,ord=1)/n))
    for i in range(n):
        D_sort=np.sort(D1[:,i])[::-1]
        Label = np.argsort(D1[:,i])[::-1]
        label_1 = np.where(D1[:,i]>=D_sort[K-1]+10**(-4))
        w_01[label_1[0],i] = 1
    return D1,w_01
