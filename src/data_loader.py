#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:01:29 2019

@author: rajs
"""

import numpy as np
import glob
import os
import torch

__all = ['data_loader']

folders = ['train', 'test']

def data_loader(st, bz):
    cur_dir = os.getcwd()   
    fw = []
    for folder in folders:
        print('Loading', folder, 'data')
        vdirpath = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/' + folder +'_features/rgb_vgg'
        os.chdir(vdirpath)
        videos = sorted(glob.glob('*'))
            
        fv = []
        for i, video in enumerate(videos):
            print(i+st, end =" ")
            video = videos[i+st]
            lpath = os.path.join(vdirpath, video)
            images = sorted(glob.glob(lpath + '/*npy'))
            
            fi = []
            for image in images:
                fi.append(np.load(image))
            fi = np.array(fi)
            ft = torch.from_numpy(fi)
            ft = ft.view(-1, 4096)           
            fv.append(ft)            
            
            if i == bz-1:
                break
            
        fw.append(fv)
        
    os.chdir(cur_dir)

    return fw

#x = data_loader()
