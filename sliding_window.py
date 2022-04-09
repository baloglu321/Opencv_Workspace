# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:22:33 2022

@author: Mehmet
"""
import cv2
import matplotlib.pyplot as mp

def sliding_window(image,step,ws):
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            yield(x,y,image[y:y+ws[1],x:x+ws[0]])
            

"""
img=cv2.imread("damla.png")
im=sliding_window(img,17,(500,500))

for i,image in enumerate(im):
    print(i)
    if i ==500:
        print(image[0],image[1])
        mp.imshow(image[2])

"""


