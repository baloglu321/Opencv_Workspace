# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:48:14 2022

@author: Mehmet
"""
import cv2
import matplotlib.pyplot as mp
import numpy as np

img=cv2.imread("kontur.png",0)
mp.figure(),mp.imshow(img,cmap="gray"),mp.axis("off")

contours,hierarch=cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


external_contours=np.zeros(img.shape)

internal_contours=np.zeros(img.shape)                                        

for i in range (len(contours)):
    #external (yani dış kotörse )
    
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255 , -1)
    
    #internal ise (yani içindeki kontörlerse) 
    
    else:
        cv2.drawContours(internal_contours, contours, i, 255 , -1) 


mp.figure(),mp.imshow(external_contours,cmap="gray"),mp.axis("off")
mp.figure(),mp.imshow(internal_contours,cmap="gray"),mp.axis("off")

