# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:16:31 2022

@author: mehmet
"""

import cv2
import matplotlib.pyplot as mp

img1=cv2.imread("Lenna_(test_image).png")
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_shape=img1.shape

img2=cv2.imread("damla.png")
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2=cv2.resize(img2, (512,512))
img2_shape=img2.shape

mp.figure()
mp.imshow(img1)

mp.figure()
mp.imshow(img2)


print(f" 1. Resim Boyutu : {img1_shape} \n 2. Resim Boyutu :{img2_shape}")

#karıştırılan resim =alpha*1. resim + beta*2. resim

blended=cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.6, gamma=0)

mp.figure()
mp.imshow(blended)