#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:57:42 2019

@author: rajs
"""

import scipy.io
import os
import glob
import shutil

cur_dir = os.getcwd()

vdirpath = '/mnt/DataDrive/Thumos/thumos15_validation_meta'
os.chdir(vdirpath)
vmat = scipy.io.loadmat('thumos15_validation_meta.mat')
os.chdir(cur_dir)

vkey = list(vmat.keys())[3]

bdirpath = '/mnt/DataDrive/Thumos/thumos15_background_meta'
os.chdir(bdirpath)
bmat = scipy.io.loadmat('thumos15_background_meta.mat')
os.chdir(cur_dir)

bkey = list(bmat.keys())[3]

fdirpath = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_9/TH15_Temporal_annotations_validation/annotations'
vsrcdir   = '/mnt/DataDrive/Thumos/thumos15_validation'
bsrcdir   = '/mnt/DataDrive/Thumos/thumos15_background'

os.chdir(fdirpath)
paths = sorted(glob.glob('*.txt'))
os.chdir(cur_dir)

for path in paths:
    file = open(os.path.join(fdirpath, path), "r")     
    content = file.readlines()
    print(content)
    
    for line in content:
        tmp = os.path.join('/mnt/DataDrive/Thumos/UCF101_Project/UCF101_9/val_videos/',
                                 line[:33] + '.mp4')
        if not os.path.exists(tmp):
            shutil.copy(os.path.join(vsrcdir, line[:33] + '.mp4'), 
                    tmp)
            
classes = []
for path in paths:    
    classes.append(path[28:-4])
    
    
            
##
vdata = vmat[vkey].squeeze()
bdata = bmat[bkey].squeeze()

vdtype = vdata.dtype
bdtype = bdata.dtype

for vid in bdata:
    action = vid['primary_action'][0]
    if action in classes:
        print(action)
        fn = vid['fnTHUMOS15'][0]
        
        tmp = os.path.join('/mnt/DataDrive/Thumos/UCF101_Project/UCF101_9/backg_videos/',
                                 fn + '.mp4')
        if not os.path.exists(tmp):
            shutil.copy(os.path.join(bsrcdir, fn + '.mp4'), 
                    tmp)

