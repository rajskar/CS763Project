#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:12:26 2019

@author: rajs
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_shape = 299
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 360
input_shape = 299 
use_parallel = True
use_gpu = True

data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

        'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}

data_dir = './data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
#                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            loopcount = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if(phase == 'train'):
                        outputs, aux = model(inputs)
                    else:
                        outputs = model(inputs)
                        
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                loopcount +=1
                if(loopcount == 100):
                    print('.')
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                PATH = '/home/rajs/ML_DL/CS763Project/Using_CNNs/UCF-101_video_classification-master/Models/dict' + str(epoch)
                
                torch.save(model.state_dict(), PATH)
                PATH = '/home/rajs/ML_DL/CS763Project/Using_CNNs/UCF-101_video_classification-master/Models/model' + str(epoch)
                torch.save(model, PATH)
                
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


 
base_model = models.inception_v3(pretrained= True)   
for param in base_model.parameters():
    param.requires_grad = False

#num_ftrs = base_model.AuxLogits.fc.in_features
#base_model.AuxLogits.fc = nn.Linear(num_ftrs, len(class_names))  
   
num_ftrs = base_model.fc.in_features
#base_model.fc = nn.Linear(num_ftrs,len(class_names))

base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(1024, len(class_names), bias=True)
                                 )

epochs = 10
model = base_model.to(device)    
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

best_model = train_model(model, criterion, optimizer,scheduler,
                     num_epochs=epochs)


" fine-tune "
ct = 0
for child in model.children():
    ct += 1
    if ct < 172:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True

criterion = nn.CrossEntropyLoss().to(device)
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)
        
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
epochs = 1000
best_model = train_model(model, criterion, optimizer,scheduler,
                     num_epochs=epochs)











