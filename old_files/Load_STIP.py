#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:49:45 2019

@author: rajs
"""

import pandas as pd
import numpy as np

import os
import glob

cur_dir = os.getcwd()

ftdirpath = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_9/UCF101_9_STIP'
os.chdir(ftdirpath)
files = sorted(glob.glob('*.txt'))

files = ['BasketballDunk.txt', 'CricketBowling.txt'] 
data = {}

for file in files:
    tmp = pd.read_csv(file, header = None)[0]        
#    header = tmp[0].split(' ')[1:]

    index_count = 1
    filenames = []
    fdata = []
    tmpdata = []
    while index_count < len(tmp):
        rd = tmp[index_count]            
        if(rd[0] == '#'):
           file_name = rd.split(' ')[1:][0] + '.avi'
           filenames.append(file_name)
           
           if len(tmpdata) > 0:
               fdata.append(tmpdata)
               tmpdata = []
               
        else:
            payload   = tmp[index_count].split(' ')[:-1]
            tmpdata.append(payload)
            
        index_count+=1
        
    fdata.append(tmpdata)        
    data.update({file : (filenames, fdata)})
    del tmpdata,  tmp, index_count, rd, payload, file, fdata, filenames, file_name
        
        
            
#                    
#            data = {header[i]:payload[i] for i in range(len(header) - 2)}
#            offset = len(header) - 2
#            data.update( {header[offset] : payload[offset:offset + 72 ]})
#            data.update( {header[offset + 1] : payload[offset+ 72: ]})
        
os.chdir(cur_dir)
