#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:31:10 2019

@author: rajs
"""

import cv2

path = '/media/rajs/Elements/ng-lab-entr-vid.avi'
vc = cv2.VideoCapture(path)

fc = 0
while cv2.waitKey(1):
    ret, frame = vc.read()           
    if ret:                       
        cv2.imshow('Input',frame)  
        fc+=1
    else:
        break
        
print(fc)
cv2.destroyAllWindows()