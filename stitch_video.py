#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 05:03:07 2019

@author: karthik
"""

import cv2

frame_path = "./object/"
cur_frame_number = 52
frame_array = []
while cur_frame_number <= 92:
    cur_frame_path = frame_path + str(cur_frame_number)+".jpg"
    cur_img = cv2.imread(cur_frame_path, cv2.IMREAD_COLOR)
    #print(cur_img.shape)
    height, width, depth = cur_img.shape
    size = ( width , height)
    frame_array.append(cur_img)
    cur_frame_number += 1
out = cv2.VideoWriter('./object_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()