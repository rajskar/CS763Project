#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:40:00 2019

@author: rajs
"""
import numpy as np
import torch

__all__ = ["ModelParam"]

def ModelParam(tc):
    params_to_update = tc.parameters()
    cnt = 0
    for name,param in tc.named_parameters():
        if param.requires_grad == True:
           t = torch.prod(torch.tensor(param.size()))
           print("\t",name, t)
           cnt+=t
    print('Total trainable parameters', cnt)
    

#    model_parameters = filter(lambda p: p.requires_grad, tc.parameters())
#    params = [ np.prod(p.size()) for p in model_parameters]
#    print('trainable parameters', params)
#    print('Total trainable parameters', sum(params))
    
    return params_to_update