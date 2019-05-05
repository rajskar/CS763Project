#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:50:23 2019

@author: rajs
"""

from TRNCell import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

__all__ = ['TRN']

class TRN(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_out, D_future, D_inpseq, batch_size, device):
        super(TRN, self).__init__()
        
        self.D_future = D_future
        self.D_in  = D_in
        self.D_hid = D_hid
        self.D_out = D_out
        self.D_inpseq= D_inpseq

        self.device = device
        self.batch_size = batch_size
        
        self.TRNCell = TRNCell(D_in, D_hid, D_out, D_future, device)
        
    def forward(self, x, h):
        (hx,cx) = h
        
        all_Act = []
        all_Act_bar = []
        for i in range(self.D_inpseq):
            Act, Act_bar, (hx, cx) = self.TRNCell(x[i], (hx,cx))
            all_Act.append(Act)
            all_Act_bar.append(Act_bar)
                
        return torch.stack(all_Act), torch.stack(all_Act_bar)
    
    def initialise_hidden_parameters(self):
        return (torch.zeros(self.batch_size, self.D_hid, 
                            device = self.device,requires_grad = True),
                torch.zeros(self.batch_size, self.D_hid, 
                            device = self.device,requires_grad = True))
        
    
def main():
    print('Enterted No Entry Zone')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)  
    SEED = 4028
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True    
    
    D_inpseq = 64
    D_future = 8
    D_in = 4096
    D_hid = 4096
    D_out = 21
    BatchSize = 32    
    
    x = torch.randn(BatchSize,D_in , device = device, requires_grad = True )
    
    t = TRN(D_in, D_hid, D_out, D_future, D_inpseq, BatchSize, device)
    
    hx,cx = t.initialise_hidden_parameters()
    
    t = t.to(device)    
    return t(x, (hx,cx))
    
if __name__ == '__main__':
    Act, Act_bar = main()