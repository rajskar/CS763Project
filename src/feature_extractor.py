# -*- coding: utf-8 -*-

from rgb_vgg16 import *
from rgb_resnet import *
from flow_inception import *
import torch
import numpy as np

root_dir = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames'
folders = ['train', 'test'] 

num_classes = 20
models = ['resnet', 'VGG', 'incpetion']
chunk_size = 6
batch_size = 64
c = 10

#def features_ext(x, model_type, num_classes):
#    if model_type == 'resnet':
#        m,n,c = 224, 224, 3
#        model = rgb_resnet200_pretrained(num_classes = num_classes)
#    elif model_type == 'VGG':
#        m,n,c = 224, 224, 3
#        model = rgb_vgg16_pretrained(num_classes = num_classes)
#    elif model_type == 'incpetion':
#        pass
    
import os, glob, cv2
model = inception_v3(in_channels = c,pretrained = True, aux_logits = False)
print('Loaded Optical Model')
model = model.cuda()

b = []
for folder in folders:
    dp = os.path.join(root_dir, folder)
    ofiles = sorted(glob.glob(os.path.join(dp, '*opt*')))            
    
    for vid in ofiles:        
        dest = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/' + folder + '_features/flow_inc'

        imgs = sorted(glob.glob(os.path.join(vid, '*.jpg')))
        f = []
        batch = []
        check = 0  
        for index, img in enumerate(imgs):   
            index = index + 2
            x = cv2.imread(img)
            
            m,n,c = 299, 299, 3
            x.resize(m, n, c)
            x = x/255.
            x = np.array([x[...,0], x[...,2]])
        
            check = len(f) + 2            
            if((check % chunk_size) == 0):
                f.append(x)
                f = np.array(f)          
                f = torch.tensor(f).float()                
                f = np.resize(f, (1, f.shape[0]* f.shape[1], m, n))
                f = torch.tensor(f).float()                
                ft = model(f.cuda()).cpu().detach().numpy()
                                
                tar_path = os.path.join(dest,os.path.split(vid)[1])
                if not os.path.exists(tar_path):
                    os.mkdir(tar_path)
                    
                p = os.path.join(tar_path, img[-9:-4]+ '.npy')
                  
                np.save(p,ft)                
                f = []
            else:
                f.append(x)

#model = rgb_vgg16_pretrained(num_classes = num_classes)
#print('Loaded RGB Model')
#m,n,c = 224, 224, 3
#
#for folder in folders:
#    print(folder)
#    dp = os.path.join(root_dir, folder)
#    rfiles = sorted(glob.glob(os.path.join(dp, '*rgb*')))
#    
#    start = 0
#    for ind, vid in enumerate(rfiles):
#        if vid.split('/')[-1] == 'thumos15_video_validation_0000426_rgb_24fps':
#            start = 1
#        
#        if(start == 0):
#            continue
#        
#        dest = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/' + folder + '_features/rgb_vgg'
#        
#        imgs = sorted(glob.glob(os.path.join(vid, '*.jpg')))
#        
#        print(ind, end ="")
#        for i in range(3, len(imgs),6):          
#            img = imgs[i]
#            
#            x = cv2.imread(img)            
#            x.resize(m, n,c)
#            x = x/255.           
#            f = np.array(x)              
#            f = np.expand_dims(f,0)            
#            f = np.transpose(f, (0, 3, 1,2))
#            f = torch.tensor(f).float()             
#            ft = model(f).detach().numpy()                       
#            tar_path = os.path.join(dest,os.path.split(vid)[1])
#            if not os.path.exists(tar_path):
#                os.mkdir(tar_path)
#            p = os.path.join(tar_path, img[-9:-4]+ '.npy')
#            np.save(p,ft)

model = model.cuda()
batch_size = 32
for folder in folders:
    print(folder)
    dp = os.path.join(root_dir, folder)
    rfiles = sorted(glob.glob(os.path.join(dp, '*rgb*')))
    
    if folder == 'train':
        start = 0
    else:
        start = 1
        
    for ind, vid in enumerate(rfiles):
        if vid.split('/')[-1] == 'thumos15_video_validation_0000455_rgb_24fps':
            start = 1
        
        if(start == 0):
            continue
        
        dest = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/' + folder + '_features/rgb_vgg'
        imgs = sorted(glob.glob(os.path.join(vid, '*.jpg')))
        
        print(ind, end ="")
        b = []
        for i in range(3, len(imgs),6):
            img = imgs[i]
            
            x = cv2.imread(img)
            x.resize(m, n,c)
            x = x/255.
            b.append(x)
            
            if len(b) == batch_size:
                f = np.array(b)
                f = np.transpose(f, (0, 3, 1,2))
                f = torch.tensor(f).float()                
                print(f.shape)
                f = f.cuda()
                ft = model(f)
                tar_path = os.path.join(dest,os.path.split(vid)[1])
                
                if not os.path.exists(tar_path):
                    os.mkdir(tar_path)
                p = os.path.join(tar_path, img[-9:-4]+ '.npy')
                np.save(p,ft.cpu().detach().numpy())
                b = []

for i in range(5):
    print(i, end="")