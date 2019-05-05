#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:33:02 2019

@author: rajs
"""
from __future__ import print_function

###################### Method 1 ##############################
import torch.nn as nn
import torch.nn.functional as F
import torch

class mVGG16(nn.Module):
    def __init__(self):
        super(mVGG16, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc6 = nn.Linear(7*7*512, 4096)


    def forward(self, x, training=True):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.fc6(x)
        return x

model = mVGG16()
pretrained_dict = torch.load('/home/rajs/.torch/models/vgg16-397923af.pth')
new = list(pretrained_dict.items())

model_dict = model.state_dict()

for index, (key, value) in enumerate(model_dict.items()):
  layer_name, weights = new[index]  
  model_dict[key] = weights


###################### Method 2 ##############################
from torchvision import models
class Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Backbone, self).__init__(*args, **kwargs)
        
        vgg16_pretrained = models.vgg16(pretrained=True)
        
        self.conv1_1 = vgg16_pretrained.features[0]
        self.conv1_2 = vgg16_pretrained.features[2]
        
        self.conv2_1 = vgg16_pretrained.features[5]
        self.conv2_2 = vgg16_pretrained.features[7]

        self.conv3_1 = vgg16_pretrained.features[10]
        self.conv3_2 = vgg16_pretrained.features[12]
        self.conv3_3 = vgg16_pretrained.features[14]

        self.conv4_1 = vgg16_pretrained.features[17]
        self.conv4_2 = vgg16_pretrained.features[19]
        self.conv4_3 = vgg16_pretrained.features[21]

        self.conv5_1 = vgg16_pretrained.features[24]
        self.conv5_2 = vgg16_pretrained.features[26]
        self.conv5_3 = vgg16_pretrained.features[28]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc6 = vgg16_pretrained.classifier[0]
        
    def forward(self, x, training=True):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.fc6(x)
        return x

model = Backbone()
import numpy as np
x = np.random.rand(224,224,3)
x = np.reshape(x, (1,224,224,3))
x = np.transpose(x, (0,3,1,2))
x = torch.tensor(x).float()
g = model(x) 
print(g.shape)

################# Method 3 ###################### 
from torchvision import models
vgg16_pretrained = models.vgg16(pretrained=True)

import torch 
import numpy as np
x = np.random.rand(224,224,3)
x = np.reshape(x, (1,224,224,3))
x = np.transpose(x, (0,3,1,2))
x = torch.tensor(x).float()
print(x.shape)

for index, layer in enumerate(vgg16_pretrained.features):
    print(index, layer)
    print(x.shape)
    x = layer(x)
    print(x.shape)
    
x = x.view(-1, 7 * 7 * 512)
x = vgg16_pretrained.classifier[0](x)
print(x.shape)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    