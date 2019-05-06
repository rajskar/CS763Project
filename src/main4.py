#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:00:03 2019

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

import argparse
parser = argparse.ArgumentParser(description='checkmodel')
parser.add_argument('--D_inpseq', type= int, default= 64,
                    help='input sequence')

parser.add_argument('--D_future', type= int, default= 4,
                    help='future')

parser.add_argument('--D_in', type= int, default= 4096,
                    help='input feature')

parser.add_argument('--D_hid', type= int, default= 64,
                    help='hidden')

parser.add_argument('--D_out', type= int, default= 21,
                    help='output dim')

parser.add_argument('--BatchSize', type= int, default= 8,
                    help='BatchSize')

parser.add_argument('--alpha', type= float, default= 1.,
                    help='alpha')

parser.add_argument('--lr', type= float, default= 0.1,
                    help='lr')

parser.add_argument('--path', type= str,default='./../Model/',
                    help='path to the models')

args = parser.parse_args()

D_inpseq = args.D_inpseq
D_future = args.D_future
D_in = args.D_in
D_hid = args.D_hid
D_out = args.D_out
BatchSize = args.BatchSize
alpha = args.alpha
lr = args.lr

print('Creating model')
Model = TRN(D_in, D_hid, D_out, D_future, D_inpseq, BatchSize, device)
print('Passing model to device')
Model = Model.to(device) 
Model = Model.float()

print('Criterion - Optim - scheduler Init')
crit = criterion(D_inpseq, alpha)


Model_Parameters = ModelParam(Model)
optimizer = torch.optim.Adam(Model_Parameters,lr=lr,
                             betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0.05)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')



print('Reading Annotation File')

train_ann, test_ann = annotations(D_out)
train_keys_ann = list(train_ann.keys())
test_keys_ann = list(test_ann.keys())

steps = 10
skip = False

hx,cx = Model.initialise_hidden_parameters() 

for i in range(0, len(train_ann) - BatchSize, BatchSize):
    print('Loading Data of given Batchsize: ', BatchSize)
    
    x, vidtag = data_loader(i, BatchSize)
    
    traindata, traintag = x[0], vidtag[0]
    testdata, testtag  = x[1], vidtag[1]

    print('Preprocessing data')
    m = 0
    for j in range(BatchSize):
        if traindata[j].size(0) > m:
            m = traindata[j].size(0)
        
    ptraindata = traindata   
    for j in range(BatchSize):
        r = ptraindata[j].size(0)
        result = np.zeros((m, D_in))
        result[:r,:] = ptraindata[j]
        ptraindata[j] = result
       
    m = 0
    for j in range(BatchSize):
        if testdata[j].size(0) > m:
            m = testdata[j].size(0)
            
    ptestdata = testdata
    for j in range(BatchSize):
        r = ptestdata[j].size(0)
        result = np.zeros((m, D_in))
        result[:r,:] = ptestdata[j]
        ptestdata[j] = result
        

    train_target = []
    test_target  = []
    for j in range(BatchSize):
        r = train_ann[traintag[j]]                
        rtg = np.zeros((1,len(ptraindata[j])))
        
        for elm in r:
            rtg[elm[1]:elm[2]] = elm[0]
            
        train_target.append(rtg)
        
    for j in range(BatchSize):
        r = test_ann[testtag[j]]                
        rtg = np.zeros((1,len(ptestdata[j])))
        
        for elm in r:
            rtg[elm[1]:elm[2]] = elm[0]
            
        test_target.append(rtg)
 
    ptraindata = np.transpose(ptraindata, (1,0,2))        
    ptraindata = torch.tensor(ptraindata, requires_grad=False, device = device)

    ptrain_target = np.transpose(train_target, (2,0,1))
    ptrain_target = torch.tensor(ptrain_target, requires_grad=False, device = device)
    ptrain_target = ptrain_target.squeeze()
    
    ptestdata = np.transpose(ptestdata, (1,0,2))        
    ptestdata = torch.torch.tensor(ptestdata, requires_grad=False, device = device)

    ptest_target = np.transpose(test_target, (2,0,1))        
    ptest_target = torch.tensor(ptest_target, requires_grad=False, device = device)
    ptest_target = ptest_target.squeeze()
    
    print('Preprocessing Done')
        
    print('Training started iter:', i)
    for j in range(0, len(ptraindata)- D_inpseq, D_inpseq ):
#        print(j)
        bptraindata = ptraindata[j: j+ D_inpseq]
        bptrain_target = ptrain_target[j: j+ D_inpseq]
               
        bptraindata.requires_grad_()
        
        bptraindata = bptraindata.to(device)
        bptrain_target = bptrain_target.to(device)
        
        bptraindata = bptraindata.type(torch.cuda.FloatTensor)
        bptrain_target = bptrain_target.type(torch.cuda.LongTensor)
        
#        print('Clearing Gradients')
        optimizer.zero_grad()
        
        hx,cx = Model.initialise_hidden_parameters() 
        Act, Act_bar = Model(bptraindata, (hx,cx))
        
#        print('Received Outputs from Model')
        
        cur_target = bptrain_target
        
        btarget = []

        for k in range(D_inpseq):            
            if (j+ k+ D_future) < len(ptrain_target):
                btarget.append(ptrain_target[j+k: j+ k+ D_future])            
            else:
                skip = True
                break

        if skip:
            break
        
        fut_target = torch.stack(btarget)
        fut_target = fut_target.type(torch.cuda.LongTensor)
        
#        print('Calcualting Loss')
        loss, Avgloss = crit((Act, Act_bar), (cur_target,fut_target))        
        print('Loss:', loss.item(), Avgloss.item())
        
#        print('backprop')
#        Avgloss.backward()
        loss.backward()
    print('Testing started')
    for j in range(0, len(ptestdata)- D_inpseq, D_inpseq ):
#        print(j)
        bptestdata = ptestdata[j: j+ D_inpseq]
        bptest_target = ptest_target[j: j+ D_inpseq]
        
        bptestdata = bptestdata.to(device)
        bptest_target = bptest_target.to(device)
        
        bptestdata = bptestdata.type(torch.cuda.FloatTensor)
        bptest_target = bptest_target.type(torch.cuda.LongTensor)
                
#        hx,cx = Model.initialise_hidden_parameters() 
        Act, Act_bar = Model(bptestdata, (hx,cx))
         
        _, preds = torch.max(Act, 2)
#        print('Received Outputs from Model')
        
        cur_target = bptest_target
        
        corrects = float(sum(sum((preds == bptest_target))).item())
        
        Accuracy = corrects/(Act.size(0) * Act.size(1))
        
        print("Accuracy:", Accuracy)
        
    if i == steps * BatchSize:
        break

torch.save(Model.state_dict(), 'model_states.pth')

torch.save(Model,'model.pth')




