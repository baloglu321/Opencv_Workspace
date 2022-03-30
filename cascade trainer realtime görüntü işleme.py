# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:51:46 2022

@author: Mehmet
"""

import cv2

nesne="Kalem"
frameWidth=280
frameHeight=360


cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

def empty (a):pass


#track bar 

cv2.namedWindow("Sonuc")
cv2.resizeWindow("Sonuc", frameWidth, frameHeight+100)
cv2.createTrackbar("Scale", "Sonuc", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Sonuc", 8, 50, empty)


cascade=cv2.CascadeClassifier("cascade.xml")

while True:
    
    
    success,img=cap.read()
    
    
    if success:
        
       gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       
       
       scale=1+(cv2.getTrackbarPos("Scale", "Sonuc")/1000)
       neighbor=cv2.getTrackbarPos("Neighbor", "Sonuc")
       
       
       rect=cascade.detectMultiScale(gray,scale,neighbor)
       
       for (x,y,w,h) in rect:
           
           cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
           
           
           cv2.putText(img, nesne, (x,y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0),2)
           
       cv2.imshow("Sonuc", img)
       
    if cv2.waitKey(1) &0xFF==ord("q"):
        break
    
    
           
cap.release()
cv2.destroyAllWindows()

















