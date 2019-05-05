#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:56:21 2019

@author: rajs
"""

from rgb_vgg16 import *
from rgb_resnet import *
from flow_inception import *
import torch
import numpy as np

def main(model_type, num_classes):
    if model_type == 'resnet':
        m,n,c = 224, 224, 3
        model = rgb_resnet200_pretrained(num_classes = num_classes)
    elif model_type == 'VGG':
        m,n,c = 224, 224, 3
        model = rgb_vgg16_pretrained(num_classes = num_classes)
    elif model_type == 'bn_incpetion':
        m,n,c,N = 299,299,10,2
        model = inception_v3( in_channels = c,pretrained = True, aux_logits = False)
        
    else:
        print('Wrong Model Selected')
       
    x = np.random.rand(m,n,c)
    x = np.reshape(x, (1,m,n,c))
    x = np.transpose(x, (0,3,1,2))
    x = torch.tensor(x).float()
    g = model(x) 
    print('Output shape in ', model_type, ': ', g.shape)
            

num_classes = 20
models = ['resnet', 'VGG', 'bn_incpetion']
if __name__ == "__main__":
    main(models[2],num_classes)