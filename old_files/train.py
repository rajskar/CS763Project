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

from data import DataSet
from models import ResearchModels

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 1000
steps_per_epoch = 1000
val_steps = 10
seq_length = 40
batch_size = 32

def train_model(model,  dataloaders_dict,
                criterion, optimizer, scheduler,
                num_epochs=num_epochs,
                steps_per_epoch = steps_per_epoch,
                val_steps = val_steps):
    
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            itercnt = 0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.squeeze(2)
#                inputs = inputs.transpose(1,0,2)
                
                inputs = torch.tensor(inputs).to(device)
                labels = torch.tensor(labels).to(device)
                
                print(inputs.size())
                print(labels.size())
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                                                
                    loss = criterion(outputs, labels.type(torch.cuda.LongTensor))

                    _, preds = torch.max(outputs, 2)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()                    
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                itercnt+=1
                if(phase == 'train'):                    
                    if(itercnt > steps_per_epoch):                                            
                        break
                else:
                    if(itercnt > val_steps):                                            
                        break

            if(phase == 'train'):  
                epoch_loss = running_loss / (itercnt * batch_size)
                epoch_acc = running_corrects.double() / (itercnt * batch_size)
            else:
                epoch_loss = running_loss / (itercnt * batch_size)
                epoch_acc = running_corrects.double() / (itercnt * batch_size)   

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

        
model = ResearchModels()
model = model.to(device)
# Now compile the network.
optimizer = optim.Adam(model.parameters(),
                       lr=1e-5, weight_decay=1e-6
                       )
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.CrossEntropyLoss().to(device)  

print("Initializing Datasets and Dataloaders...")

data = DataSet(seq_length=seq_length)

#steps_per_epoch = (len(data.data) * 0.7) // batch_size

load_to_memory = False
data_type = 'features'
if load_to_memory:
    # Get data.    
    X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type)                    
    X_test, y_test = data.get_all_sequences_in_memory(batch_size,'test', data_type)
else:
    # Get generators.
    generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

dataloaders_dict = {'train':generator, 
               'val' : val_generator} 

model_ft, hist = train_model(model, dataloaders_dict, criterion, 
                             optimizer, scheduler, 
                             num_epochs=num_epochs,
                             steps_per_epoch = steps_per_epoch,
                             val_steps = val_steps,
                             )