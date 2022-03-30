# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:54:17 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp


img=cv2.imread("Albert_Einstein_Head.jpg",0)
mp.figure(),mp.imshow(img,cmap="gray"), mp.axis("off")



face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect=face_cascade.detectMultiScale(img)


for (x,y,w,h) in face_rect:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255),10)
    

mp.figure(),mp.imshow(img,cmap="gray"), mp.axis("off")


#grup resim tespiti
#%%
new_img=cv2.imread("galatasaray.jpg")
new_img=cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)

mp.figure(),mp.imshow(new_img), mp.axis("off")



face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect=face_cascade.detectMultiScale(new_img,minNeighbors=3)


for (x,y,w,h) in face_rect:
    cv2.rectangle(new_img, (x,y), (x+w,y+h), (255,255,255),2)
    

mp.figure(),mp.imshow(new_img), mp.axis("off")


#%%
#real time yüz algılama

cap=cv2.VideoCapture(0)

while True:
    
    success,img=cap.read()
    
    if success:
        
        face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        face_rect=face_cascade.detectMultiScale(img,minNeighbors=2)

        for (x,y,w,h) in face_rect:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255),2)

        cv2.imshow("Görüntü", img)

    if cv2.waitKey(1) &0xFF ==ord("q"):
      

        break


cap.release()
cv2.destroyAllWindows()


