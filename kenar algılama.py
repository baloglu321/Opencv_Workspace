# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:53:36 2022

@author: Mehmet
"""

import cv2
import  matplotlib.pyplot as mp
import numpy as np

img=cv2.imread("london.png",0)
mp.figure()
mp.imshow(img,cmap="gray")
mp.axis("off")

#kenar algılama
def treshHold(image): #treshhold değerlerini hesaplama için formül
    
    med_val=np.median(image)
    
    low=int(max(0,(1 - 0.33)*med_val)) #treshold aralığı oluşturmak için formül
    high=int(max(0,(1 + 0.33)*med_val))
    return low,high

low,high=treshHold(img)

edges=cv2.Canny(img, threshold1=low, threshold2=high) #treshold değerleri formül ile belirlendiğinde detay seviyesi düşsede yeterli olmayabilir bu durumda blurlama uygulanabilir.

mp.figure()
mp.imshow(edges,cmap="gray")
mp.axis("off")

#blur uygulayalım
blured_img=cv2.blur(img,ksize=(4,4))
mp.figure()
mp.imshow(blured_img,cmap="gray")
mp.axis("off")


#yeniden treshold hesaplama
b_low,b_high=treshHold(blured_img)

blured_edges = cv2.Canny(blured_img, threshold1=b_low, threshold2=b_high)
mp.figure()
mp.imshow(blured_edges,cmap="gray")
mp.axis("off")

#detay seviyesi yinede fazla gelirse blur seviyesi artırılarak ayarlanabilir.







