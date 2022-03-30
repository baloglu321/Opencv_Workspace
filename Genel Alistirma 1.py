# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:02:20 2022

@author: Mehmet
"""
import cv2
import matplotlib.pyplot as mp


img=cv2.imread("kopek_kedi_at.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img=cv2.imread("kopek_kedi_at.png",0)

mp.figure()
mp.imshow(img)
mp.axis("off")


mp.figure()
mp.imshow(gray_img,cmap="gray")
mp.axis("off")

b_img=(img.shape),(gray_img.shape)



print(f"Renkli Resim Boyutu: {b_img[0]}\nGri Resim Boyutu : {b_img[1]}")

def resimOlcekle(image,oran):
    size=image.shape[:2]
    a=size[0]*oran
    b=size[1]*oran
    re_img=cv2.resize(image,(int(b),int(a)))
    return re_img

re_img=resimOlcekle(img, 0.8)
gray_img=resimOlcekle(gray_img, 0.8)

re_size=re_img.shape

print(f"Yeniden Boyutlandırılan Resim Boyutu: {re_size}")



cv2.circle(re_img, (74,167), 125, (255,0,0),3)
cv2.circle(re_img, (269,143), 75, (255,0,0),3)
cv2.circle(re_img, (381,171), 60, (255,0,0),3)
cv2.putText(re_img, "At", (33,31), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
cv2.putText(re_img, "Kopek", (226,60), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
cv2.putText(re_img, "Kedi", (337,76), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

mp.figure()
mp.imshow(re_img)
mp.axis("off")



_,filtred_img=cv2.threshold(gray_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

mp.figure()
mp.imshow(filtred_img,cmap="gray")
mp.axis("off")

adaptive_img=cv2.adaptiveThreshold(gray_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11,C=8)
mp.figure()
mp.imshow(adaptive_img,cmap="gray")
mp.axis("off")

gb=cv2.GaussianBlur(gray_img, ksize=(3,3), sigmaX=7)
mp.figure()
mp.imshow(gb,cmap="gray")
mp.title("Gaus Yöntemi İle Bulanıklaştırılmış Resim")
mp.axis("off")


laplacian=cv2.Laplacian(gray_img, ddepth=cv2.CV_64F)
cv2.imshow("Laplacian", laplacian)



color=("b","g","r")
mp.figure()
for i,c in enumerate(color):
    hist=cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0,256])
    mp.plot(hist,color=c)

img_hist=cv2.calcHist(images=[gray_img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    
mp.figure()
mp.plot(img_hist)    