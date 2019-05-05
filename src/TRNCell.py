#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:40:33 2019

@author: rajs
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

__all__ = ['TRNCell']

class TRNCell(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_out, D_future, device):
        super(TRNCell, self).__init__()
        
        self.D_future = D_future
        self.D_in  = D_in
        self.D_hid = D_hid
        self.D_out = D_out
        
        self.DecRNNCell = nn.LSTMCell(D_out, D_hid)
        self.LinLayerR0 = nn.Linear(D_hid, D_hid)
        self.LinLayerRp = nn.Linear(D_hid, D_out)
        self.LinLayerRr = nn.Linear(D_out, D_out)

        self.LinLayerS0 = nn.Linear(D_hid, D_in)
        self.Avgpool = nn.AvgPool1d(D_future)
        self.StaRNNCell = nn.LSTMCell(2* D_in, D_hid)

        self.LinLayerSn = nn.Linear(D_hid, D_out)        
        
        self.device = device
        
    def forward(self, x, h):
        (hx,cx) = h
        ohx     = hx
        ocx     = cx
        fts_bar = []
        Act_bar = []        
        BatchSize = x.size(0)
        
        r  = torch.zeros(BatchSize, self.D_out, 
                         device = self.device, requires_grad = True)
        
        hx = self.LinLayerR0(hx)
        
        for i in range(self.D_future):
          hx, cx = self.DecRNNCell(r, (hx, cx))
          fts_bar.append(hx)         
          p = self.LinLayerRp(hx)
          r = self.LinLayerRr(p)  
          Act_bar.append(p)

        fts_bar = torch.stack(fts_bar)  
        Act_bar = torch.stack(Act_bar)

        fts_bar = fts_bar.permute(1,2,0)
        hbar = self.Avgpool(fts_bar)
        hbar = hbar.view(-1,self.D_hid)
        
        xbar = self.LinLayerS0(hbar)        
        xbar = F.relu(xbar)
        
        xcat = torch.cat((x, xbar), 1)

        hx, cx = self.StaRNNCell(xcat, (ohx, ocx))
        Act = self.LinLayerSn(hx)
        
        return Act, Act_bar, (hx, cx)

def main():
    print('Enterted No Entry Zone')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)  
    SEED = 4028
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True    
    
    D_future = 8
    D_in = 4096
    D_hid = 4096
    D_out = 21
    BatchSize = 32    
    x = torch.randn(BatchSize,D_in , device = device, requires_grad = True )
    hx = torch.randn(BatchSize, D_hid,  device = device,requires_grad = True)
    cx = torch.randn(BatchSize, D_hid,device = device,  requires_grad = True)    
    tc = TRNCell(D_in, D_hid, D_out, D_future, device)
    tc = tc.to(device)    
    return tc(x, (hx,cx))
    
if __name__ == '__main__':
    Act, Act_bar, (hx, cx)= main()