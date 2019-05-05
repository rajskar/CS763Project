#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:43:24 2019

@author: rajs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import time

SEED = 4028

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


numFutureCells = 8
BatchSize = 32
Featurelen = 4096
HiddenSize = 4096
Classes = 21

x = torch.randn(BatchSize,Featurelen )
hx = torch.randn(BatchSize, HiddenSize)
cx = torch.randn(BatchSize, HiddenSize)
r  = torch.zeros(BatchSize, Classes)

ohx = hx
ocx = cx

fts_bar = []
Act_bar = []

DecRNNCell = nn.LSTMCell(Classes, HiddenSize)
LinLayerR0 = nn.Linear(HiddenSize, HiddenSize)
LinLayerRp = nn.Linear(HiddenSize, Classes)
LinLayerRr = nn.Linear(Classes, Classes)

LinLayerS0 = nn.Linear(HiddenSize, Featurelen)
Avgpool = nn.AvgPool1d(numFutureCells)
StaRNNCell = nn.LSTMCell(2* Featurelen, HiddenSize)

LinLayerSn = nn.Linear(HiddenSize, Classes)


hx = LinLayerR0(hx)
for i in range(numFutureCells):
  hx, cx = DecRNNCell(r, (hx, cx))
  fts_bar.append(hx)
  p = LinLayerRp(hx)
  r = LinLayerRr(p)  
  Act_bar.append(p)

fts_bar = torch.stack(fts_bar)  
Act_bar = torch.stack(Act_bar)

fts_bar = fts_bar.permute(1,2,0)
hbar = Avgpool(fts_bar)
hbar = hbar.view(-1,HiddenSize)

xbar = LinLayerS0(hbar)        
xbar = F.relu(xbar)

xcat = torch.cat((x, xbar), 1)

hx, cx = StaRNNCell(xcat, (ohx, ocx))
Act = LinLayerSn(hx)