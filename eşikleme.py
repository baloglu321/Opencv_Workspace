# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:37:34 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp

#resimi siyah beyaz yapmak için

img=cv2.imread("Lenna_(test_image).png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#yada !
"""
img2=cv2.imread("Lenna_(test_image).png",0)

mp.figure()
mp.imshow(img2,cmap="gray")"""

#ikiside aynı şey
mp.figure()
mp.imshow(img,cmap="gray")

#eşikleme

_,filtred_img=cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
_,filtred_img2=cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)



mp.figure()
mp.imshow(filtred_img2,cmap="gray")


#uyarlamalı eşikleme(yandaki pixellere göre resimdeki gölgelemelere dikkat edip eşik değerlerini adapte eder)

adaptive_img=cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11,C=8)
adaptive_img2=cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=11,C=8)


fig=mp.figure()
ax=fig.add_subplot(1,2,1)
mp.imshow(adaptive_img,cmap="gray")
ax.set_title("adaptive")
ax=fig.add_subplot(1,2,2)
mp.imshow(filtred_img,cmap="gray")
ax.set_title("uyarlamalı")

fig=mp.figure()
ax=fig.add_subplot(1,2,1)
mp.imshow(adaptive_img2,cmap="gray")
ax.set_title("adaptive")
ax=fig.add_subplot(1,2,2)
mp.imshow(filtred_img2,cmap="gray")
ax.set_title("uyarlamalı")
