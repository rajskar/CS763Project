#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:32:43 2019

@author: rajs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['criterion']

class criterion(torch.nn.Module):
    def __init__(self, inpseq, alpha = 1.):
        super(criterion, self).__init__()  
        self.inpseq = inpseq
        self.alpha = alpha
        self.cur_crit = nn.CrossEntropyLoss()
        self.fut_crit = nn.CrossEntropyLoss()
                
    def forward(self, x, target):
        Act, Act_bar = x
        cur_target, fut_target = target
        
        loss = 0.
        for i in range(self.inpseq):                
            cur_loss = self.cur_crit(Act[i], cur_target[i])
        
            fut_loss = 0.
            for (act, tar) in zip(Act_bar[i], fut_target[i]):
                fut_loss += self.fut_crit(act, tar)
        
            loss += cur_loss + (self.alpha * fut_loss)
               
        nseq = Act_bar.size(0)
        nft  = Act_bar.size(1)
        nbz  = Act_bar.size(2)
        ntotal_losses = (nseq * nft * nbz) + (nseq * nbz)
        Avgloss = loss/(ntotal_losses)
        return loss, Avgloss
        


