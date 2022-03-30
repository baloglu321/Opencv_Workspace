# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:34:39 2022

@author: Mehmet
"""

# Havza Algoritması : Watershed Algorithm

#nesne tespiti için topgrafi kullanılır.

import cv2
import matplotlib.pyplot as mp
import numpy as np

img=cv2.imread("bozuk_para.png")

mp.figure()
mp.imshow(img)
mp.axis("off")

b_img=cv2.medianBlur(img, 11)

mp.figure()
mp.imshow(b_img)
mp.axis("off")

gray_img=cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)

mp.figure()
mp.imshow(gray_img,cmap="gray")
mp.axis("off")

#binary treshold :
    
_,filtred_img=cv2.threshold(gray_img, thresh=75, maxval=255, type=cv2.THRESH_BINARY)   

mp.figure()
mp.imshow(filtred_img,cmap="gray")
mp.axis("off")


contur,hierarch=cv2.findContours(filtred_img.copy(),cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)


for i in range (len (contur)):
    if hierarch[0][i][3]==-1:
        cv2.drawContours(img, contur, i,(0,255,0),10)
        

mp.figure()
mp.imshow(img)
mp.axis("off")

#watershed
#konturu alınmış filtred img e  morfolojik operasyon uygulanırsa aralarındaki bağlantı azalacaktır.

kernel = np.ones((3,3),dtype=np.uint8)
opening=cv2.morphologyEx(filtred_img.astype(np.float32), cv2.MORPH_OPEN, kernel,iterations=2)

mp.figure()
mp.imshow(opening,cmap="gray")
mp.axis("off")

distance=cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 5)

mp.figure()
mp.imshow(distance,cmap="gray")
mp.axis("off")

#öndeki resmi bulabilmek için nesneleri küçültelim

ret, sure_foreground=cv2.threshold(distance, 0.32*np.max(distance), 255, 0)

mp.figure()
mp.imshow(sure_foreground,cmap="gray")
mp.axis("off")


#arka planı belilemek için nesneleri büyütelim:
    
sure_background=cv2.dilate(opening, kernel, iterations=1)
sure_background=np.uint8(sure_background)
sure_foreground=np.uint8(sure_foreground)
unkown=cv2.subtract(sure_background, sure_foreground) #ikisinin farkını aldık

mp.figure()
mp.imshow(unkown,cmap="gray")
mp.axis("off")

#bağlantı
ret,marker=cv2.connectedComponents(sure_foreground)
marker=marker+1
marker[unkown==255]=0

mp.figure()
mp.imshow(marker,cmap="gray")
mp.axis("off")

#havza allgoritması ile segmentasyon

havza_marker=cv2.watershed(img, marker)
mp.figure()
mp.imshow(havza_marker,cmap="gray")
mp.axis("off")


contur,hierarch=cv2.findContours(havza_marker.copy(),cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)


for i in range (len (contur)):
    if hierarch[0][i][3]==-1:
        cv2.drawContours(img, contur, i,(255,0,0),2)
        

mp.figure()
mp.imshow(img)
mp.axis("off")