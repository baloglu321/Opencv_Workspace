# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:52:20 2022

@author: mehmet
"""
import cv2

veri=cv2.imread("Lenna_(test_image).png")
boyut=veri.shape
print(f"Resim Boyutu : {boyut}")
cv2.imshow("Orjinal Resim" ,veri)

#yendien boyutlandırma
imgResized=cv2.resize(veri,(800,800))
boyut2=imgResized.shape
print(f"Yeniden Boyutlandırılmış Resim Boyutu : {boyut2}")

cv2.imshow("Yeniden Boyutlandirilmis Resim" ,imgResized)

#kırpma

imgCorped=veri[:200,:300]
cv2.imshow("Kirpilmis Resim ",imgCorped)