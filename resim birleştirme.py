# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:45:22 2022

@author: mehmet
"""

import cv2
import numpy as np



img=cv2.imread("Lenna_(test_image).png")
cv2.imshow("lenna",img)

#yatay birleştirme
hor=np.hstack((img,img))
cv2.imshow("Horizontal",hor)

#dikey birleştirme

ver=np.vstack((img,img))
cv2.imshow("vertical",ver)