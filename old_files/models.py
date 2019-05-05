# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import deque

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResearchModels(nn.Module):
    def __init__(self, nb_classes = 20, model = "lstm", seq_length = 40,
                 saved_model = None, features_length = 2048
                 ):
        super(ResearchModels, self).__init__()
        
        self.seq_length = seq_length
        self.load_model = []
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()        
        self.features_length = features_length        
        self.input_shape = (self.seq_length, self.features_length)
        
        self.lstm = nn.LSTM(self.features_length,2048)   
        self.fc0  = nn.Linear(2048, 512, bias=True)
        self.dpt  = nn.Dropout(0.5)
        self.fc1  = nn.Linear(512, self.nb_classes, bias=True)
        
    def forward(self, x):
        lstm_out, (h,c) = self.lstm(x)
        print(h.shape, c.shape)
        lin_out     = self.fc1(self.dpt(self.fc0(lstm_out)))
        return lin_out


def fmain():
    lmodel = ResearchModels()
    print(lmodel.model)
    
if __name__ == '__main()__':
    fmain()