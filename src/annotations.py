#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:45:24 2019

@author: rajs
"""

import numpy as np
import glob
import os

__all = ['annotations']


def annotations():
    
    folders = ['train', 'test']
    cur_dir = os.getcwd()   
    fw = {}
    for folder in folders:    
        vdirpath = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/' + folder +'_features/rgb_vgg'
        os.chdir(vdirpath)
        videos = sorted(glob.glob('*'))
        fw.update({folder: videos})
        
    os.chdir(cur_dir)
    
    cur_dir = os.getcwd()   
    vdirpath = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames/annotations'
    os.chdir(vdirpath)
    annfile = sorted(glob.glob('*txt'))
    
    antd = []
    for ann in annfile:        
        file = open(ann,"r+") 
        antd.append(file.readlines())
        file.close()
        
    os.chdir(cur_dir)
    
    trainVideos = fw['train']
    testVideos = fw['test']
    
    classes = 21
    dictrn = {}
    dictst = {}
    for i in range(classes):
        vid_class = antd[i]
        for vid in vid_class:
            vid = vid.strip('\n')
            vidname = vid.split(' ')[0]
            vidstrt = vid.split(' ')[1]
            videndt = vid.split(' ')[2]
            vidname = vidname + '_rgb_24fps'
            
            if vidname in trainVideos:
                if vidname in dictrn:
                    l = dictrn[vidname]
                else:
                    l= []
                l.append((i, vidstrt, videndt))
                dictrn.update({vidname: l})
                
            if  vidname in testVideos:
                if vidname in dictst:
                    l = dictst[vidname]
                else:
                    l= []
                l.append((i, vidstrt, videndt))
                dictst.update({vidname: l})
    
    return dictrn, dictst
