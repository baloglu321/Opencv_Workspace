# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:03:02 2022

@author: Mehmet
"""

import cv2
import os
import matplotlib.pyplot as mp


current_dir=os.getcwd()
new_folder="kediler"

os.chdir(current_dir+"\\"+new_folder)

#%%

files=os.listdir()
print(files)

img_list=[]

for i in files:
    if i.endswith(".png"):
        img_list.append(i)
    else:
        cat_cascade=cv2.CascadeClassifier(i)



def yuzAlgıla(img):
    
    face_rect=cat_cascade.detectMultiScale(img,scaleFactor=1.026,minNeighbors=3)
    
    for (i,(x,y,w,h)) in enumerate(face_rect):
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255),2)
        center=(int(x+(w/2)),int(y+(h/2)))
        cv2.circle(img, center, 5, (255,0,255),-1)
        kedi=i+1
        cv2.putText(img,f" Kedi {kedi}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0),2)

   
    return img


    
    
  
for j in img_list:
        
               
    img=cv2.imread(j)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   
    new_img=yuzAlgıla(img)
        
    cv2.imshow(j, new_img)
    
    if cv2.waitKey(0) &0xFF==ord("q"): break
        
    if cv2.waitKey(1) &0xFF==ord("d"): continue
    
    
cv2.destroyAllWindows()





