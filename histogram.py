# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:24:31 2022

@author: Mehmet

"""
import cv2
import matplotlib.pyplot as mp
import numpy as np


img=cv2.imread("kirmizi_mavi.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


mp.figure()
mp.imshow(img)
mp.axis("off")
mp.title("orjinal resim")

print(img.shape)

img_hist=cv2.calcHist(images=[img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
print(img_hist.shape)

mp.figure()
mp.plot(img_hist)

color=("b","g","r")
mp.figure()
for i,c in enumerate(color):
    hist=cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0,256])
    mp.plot(hist,color=c)


resim=cv2.imread("arcane-fanart-25.jpg")  
resim=cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
print(resim.shape) 
mp.figure()
mp.imshow(resim)
mp.axis("off")
mp.title("orjinal resim")


#maskeleme


mask=np.zeros(resim.shape[:2],np.uint8)
mask[250:850,2000:3000]=255
mp.figure()
mp.imshow(mask,cmap="gray")
mp.axis("off")
mp.title("maske")


mask_resim=cv2.bitwise_and(resim, resim,mask=mask)
mp.figure()
mp.imshow(mask_resim)
mp.axis("off")
mp.title("maskelenmiş resim")

resim_hist=cv2.calcHist(images=[mask_resim], channels=[0], mask=mask, histSize=[256], ranges=[0,256])

mp.figure()
mp.plot(resim_hist)


mp.figure()
for i,c in enumerate(color):
    hist=cv2.calcHist(images=[mask_resim], channels=[i], mask=mask, histSize=[256], ranges=[0,256])
    mp.plot(hist,color=c)


#histogram  eşitleme
#karşıtlık artırma

img2=cv2.imread("dusuk_karsitlik.png",0)
mp.figure()
mp.imshow(img2,cmap="gray")
mp.axis("off")
mp.title("düşük karşıtlıklı resim")

img2_hist=cv2.calcHist(images=[img2], channels=[0], mask=None, histSize=[256], ranges=[0,256])
mp.figure()
mp.plot(img2_hist)

eq_hist=cv2.equalizeHist(img2)
mp.figure()
mp.imshow(eq_hist,cmap="gray")
mp.axis("off")
mp.title("kontrastı yükseltilmiş(eşitlenmiş)\nresim")

equ_img2_hist=cv2.calcHist(images=[eq_hist], channels=[0], mask=None, histSize=[256], ranges=[0,256])
mp.figure()
mp.plot(equ_img2_hist)


