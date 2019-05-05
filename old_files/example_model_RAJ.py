#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:19:44 2019

@author: rajs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.rand(3,32,32)
x = x.unsqueeze(0)
x.size()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
o1 = net.forward(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.Model = nn.Sequential(nn.Conv2d(3, 6, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(6, 16, 5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   Flatten(),
                                   nn.Linear(16 * 5 * 5, 120),
                                   nn.ReLU(),
                                   nn.Linear(120, 84),
                                   nn.ReLU(),
                                   nn.Linear(84, 10)
                                   )
net1 = Net1()
o2 = net1.Model(x)