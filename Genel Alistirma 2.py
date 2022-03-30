# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:53:31 2022

@author: Mehmet
"""


import cv2
import numpy as np


img=cv2.imread("yaya_img.png",0)

cv2.imshow("Resim", img)


#kenar tespiti


def treshHold(image): #treshhold değerlerini hesaplama için formül
    
    med_val=np.median(image)
    
    low=int(max(0,(1 - 0.33)*med_val)) #treshold aralığı oluşturmak için formül
    high=int(max(0,(1 + 0.33)*med_val))
    return low,high

low,high=treshHold(img)



kenar=cv2.Canny(img, low, high)
cv2.imshow("Kenar Algilama Sonuc", kenar)

#%%yüz tespiti

image=img.copy()

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rect=face_cascade.detectMultiScale(image,scaleFactor=1.070,minNeighbors=5)

for (x,y,w,h) in rect:
    
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
    cv2.putText(image, "Fakir", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.55, (255,0,0))

cv2.imshow("Face Detect", image)

if cv2.waitKey(0) &0xFF==ord("q"):cv2.destroyAllWindows()
    

#%% Hog İlklendirme ile yaya tespiti

hog=cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects,weight)=hog.detectMultiScale(img,padding=(5,5),scale=1.05)

for (x,y,w,h) in rects:
    cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)


cv2.imshow("Yaya Tespiti", img)

if cv2.waitKey(0) &0xFF==ord("q"): cv2.destroyAllWindows()

    

    






