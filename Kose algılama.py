# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:46:11 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp
import numpy as np

img=cv2.imread("sudoku.png",0)
img=np.float32(img)
mp.figure()
mp.imshow(img, cmap="gray")
mp.axis("off")

#harris corner detection
dst=cv2.cornerHarris(img, blockSize=2 , ksize=3 , k=0.04)
mp.figure()
mp.imshow(dst, cmap="gray")
mp.axis("off")

#köşeler çok belirsiz olduğu için köşeleri genişletiyoruz
dst=cv2.dilate(dst, kernel=None)
image=img.copy()
img[dst>0.2*dst.max()]=1
#bu genişlettiğim köşeleri tekrar köşe belirlemeye atarsam
dst=cv2.cornerHarris(img, blockSize=2 , ksize=3 , k=0.04)

mp.figure()
mp.imshow(dst, cmap="gray")
mp.axis("off")


#shi thomsai corner detection
corners=cv2.goodFeaturesToTrack(image, 120, 0.01, 10)#(işlem yapılacak resim ismi,resimde bulunacak köşe sayısı, köşe keskinliği,iki köşe arasındaki minimum mesafe)
corners=np.int64(corners)#köşe konumlarını integer a çevirdik

for i in corners :
    x,y=i.ravel()#köşe konumlarını x ve y ye atadık
    cv2.circle(image, (x,y),  5,  (125,125,125),  cv2.FILLED)#bu köşeleri içi dolu dairelerle işaretledik

mp.figure()
mp.imshow(image)
mp.axis("off")

