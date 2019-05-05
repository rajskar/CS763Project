#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:12:24 2019

@author: rajs
"""
import os
import glob

cur_dir = os.getcwd()

Vid_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_videos'
Ann_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/annotations'
frames_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames'
optflow_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/opt_frames'

os.chdir(Vid_path )
Vidfiles = sorted(glob.glob('*.mp4'))
print(Vidfiles, '\n Length: ', len(Vidfiles))

os.chdir(Ann_path)
Annfiles = sorted(glob.glob('*.txt'))
print(Annfiles, '\n Length: ', len(Annfiles))
os.chdir(cur_dir)

import random
random.seed(4028)
random.shuffle(Vidfiles)

train_data_size = 200
train, test = Vidfiles[:train_data_size], Vidfiles[train_data_size:]
print(len(train), len(test))

import glob
import os
import os.path
from subprocess import call

import cv2
import numpy as np


data_dict = {'train': train, 'test':test}
if not os.path.exists(os.path.join(frames_path, 'train')):
    os.mkdir(os.path.join(frames_path, 'train'))

if not os.path.exists(os.path.join(frames_path, 'test')):
    os.mkdir(os.path.join(frames_path, 'test'))
    
    
for index, (key, val) in enumerate(data_dict.items()):
    for elem in val:   
        
        print(index)             
        
        src = os.path.join(Vid_path, elem)

        parts = src.split(os.path.sep)
        filename = parts[-1]
        filename_no_ext = filename.split('.')[0]
        
        tdp = os.path.join(frames_path, key, filename_no_ext + "_rgb_24fps")
        
        if not os.path.exists(tdp):
            os.mkdir(tdp)
        
        dest = os.path.join(tdp, '%05d.jpg')
        call(["ffmpeg", "-i", src, "-vf", "fps=24", dest])
        
        number_files = len(os.listdir(tdp))
        print(number_files)
        
        
        img_path = sorted(glob.glob(os.path.join(tdp,'*.jpg')))
        
        frame1 = cv2.imread(img_path[0])
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        
        i = 1
        while i < number_files:
            frame2 = cv2.imread(img_path[i])
            fnext = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        
            flow = cv2.calcOpticalFlowFarneback(prvs,fnext, None, 0.5, 3, 15, 3, 7, 1.5, 0)
        
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                   
            tdp = os.path.join(frames_path, key, filename_no_ext + "_optflow_24fps")
            if not os.path.exists(tdp):
                os.mkdir(tdp)            
                
            cv2.imwrite(tdp + '/'+img_path[i].split('/')[-1][:-4] + '.jpg',rgb)
            prvs = fnext
            i+=1