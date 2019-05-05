#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:31:13 2019

@author: rajs
"""
import os
import glob

cur_dir = os.getcwd()

Vid_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_videos'
Ann_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/annotations'
frames_path = '/mnt/DataDrive/Thumos/UCF101_Project/UCF101_20/val_frames'

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

data_dict = {'train': train, 'test':test}
if not os.path.exists(os.path.join(frames_path, 'train')):
    os.mkdir(os.path.join(frames_path, 'train'))

if not os.path.exists(os.path.join(frames_path, 'test')):
    os.mkdir(os.path.join(frames_path, 'test'))
    
    
for key, val in data_dict.items():
    for elem in val:        
        src = os.path.join(Vid_path, elem)
        
        parts = src.split(os.path.sep)
        filename = parts[-1]
        filename_no_ext = filename.split('.')[0]
        
        tdp = os.path.join(frames_path, key, filename_no_ext)
        
        if os.path.exists(tdp):
            dest = os.path.join(tdp, '%04d.jpg')
            call(["ffmpeg", "-i", src, "-vf", "fps=1/24", dest])
        else:
            os.mkdir(tdp)
            dest = os.path.join(tdp, '%04d.jpg')
            call(["ffmpeg", "-i", src, "-vf", "fps=1/24", dest])  
            
         

            # Now get how many frames it is.
#            nb_frames = get_nb_frames_for_video(dest)

#            data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

#            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

#    with open('data_file.csv', 'w') as fout:
#        writer = csv.writer(fout)
#        writer.writerows(data_file)

#    print("Extracted and wrote %d video files." % (len(data_file)))


#def get_nb_frames_for_video(video_parts):
#    """Given video parts of an (assumed) already extracted video, return
#    the number of frames that were extracted."""
#    train_or_test, classname, filename_no_ext, _ = video_parts
#    generated_files = glob.glob(dest)
#    return len(generated_files)

