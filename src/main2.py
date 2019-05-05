#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:22:16 2019

@author: rajs
"""

from TRN import *
from criterion import *
from ModelParam import *
from data_loader import *
from annotations import *

import random
import torch

import numpy as np

SEED = 4028
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

D_inpseq = 64
D_future = 8
D_in = 4096
D_hid = 4096
D_out = 21
BatchSize = 5
alpha = 1.
   
#x = torch.randn(D_inpseq, BatchSize, D_in , device = device, requires_grad = True )
train_ann, test_ann = annotations()
train_keys_ann = list(train_ann.keys())
test_keys_ann = list(test_ann.keys())

steps = 10
for i in range(0, len(train_ann), BatchSize):
    x = data_loader(i, BatchSize)
    
    traindata = x[0]
    testdata  = x[1]
    
    zerrow = torch.zeros(1, D_in)
    
    m = 0
    for j in range(BatchSize):
        if traindata[j].size(0) > m:
            m = traindata[j].size(0)
        
    for j in range(BatchSize):
        r = traindata[j].size(0)
        result = np.zeros((m, D_in))
        result[:r,:] = traindata[j]
        traindata[j] = result

    traindata = np.transpose(traindata, (1,0,2))        
    traindata = torch.from_numpy(traindata)

    m = 0
    for j in range(BatchSize):
        if testdata[j].size(0) > m:
            m = testdata[j].size(0)
        
    for j in range(BatchSize):
        r = testdata[j].size(0)
        result = np.zeros((m, D_in))
        result[:r,:] = testdata[j]
        testdata[j] = result

    testdata = np.transpose(testdata, (1,0,2))        
    testdata = torch.from_numpy(testdata)

    for j in range(BatchSize):
        r = train_ann[train_keys_ann[i+j]]
        seq_tar = 
        result = np.zeros((m, D_in))
        result[:r,:] = testdata[j]
        testdata[j] = result
    
                    
    if i == steps * BatchSize:
        break



t = TRN(D_in, D_hid, D_out, D_future, D_inpseq, BatchSize, device)
hx,cx = t.initialise_hidden_parameters()

t = t.to(device)    
Act, Act_bar = t(x, (hx,cx))

btarget = torch.arange(Act.size(1), device = device, requires_grad = False)
cur_target = btarget.repeat(D_inpseq, 1)

fut_target = btarget.repeat(D_inpseq,D_future,1)

crit = criterion(D_inpseq, alpha)

loss, Avgloss = crit((Act, Act_bar), (cur_target,fut_target))
print(loss, Avgloss)
Avgloss.backward()

Model_Parameters = ModelParam(t)
optimizer = torch.optim.Adam(Model_Parameters,lr=0.0005,
                             betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')