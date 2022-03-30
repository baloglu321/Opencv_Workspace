# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:29:49 2022

@author: Mehmet
"""

import cv2
import os

folder="yaya tespiti"

current=os.getcwd()
os.chdir(current+"\\"+folder)

file=os.listdir()

#%%

print(file)

img_list=[]

for i in file:
    if i.endswith(".jpg"):
        
        img_list.append(i)

#hog tanımlayıcısı

hog=cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for img in img_list:
    
    img=cv2.imread(img)
    
    (rects,weight)=hog.detectMultiScale(img,padding=(8,8),scale=1.05)
    
    for (x,y,w,h) in rects:
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),3)
        
        
        
        
    cv2.imshow("Yaya tespiti", img)
        
        
    if cv2.waitKey(0) &0xFF==ord("d"):continue
    
    if cv2.waitKey(1) &0xFF==ord("q"):break


cv2.destroyAllWindows()







