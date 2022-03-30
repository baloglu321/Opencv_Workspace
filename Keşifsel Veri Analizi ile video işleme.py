# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:27:53 2022

@author: Mehmet

- Veri Setini al

-Veri setini indir

-veri setini videoya çevirme

-E.D.A :Exploratory data analysis (Keşifsel Veri Analizi)

- Objeye Ait Grand truth (GT) çıkar




"""

import cv2
import os
import time
import pandas as pd
import numpy as np


 

current=os.getcwd()

folder="Kesifsel Analiz"
os.chdir(current+"\\"+folder+"\\"+"img1")

file=os.listdir()
#%%

images=[]
for i in file:
    if i.endswith(".jpg"):
        images.append(i)
        

img=cv2.imread(images[44])
cv2.imshow("winname", img)
size=(img.shape[1],img.shape[0])
fps=25

#%%
img=cv2.imread(images[44])
#cv2.imshow("winname", img)



out=cv2.VideoWriter("deneme.mp4",cv2.VideoWriter_fourcc(*"MP4V"),fps,size,True)


for i in images:
    print(i)

    
    img=cv2.imread(i)
    out.write(img)
    
    
out.release()    
    

#%% Grand truth bilgisi
os.chdir(current+"\\"+folder+"\\"+"gt")
print(os.listdir())
#%%
columns=["frame number","id","left","top","width","height","score","class","visibilty"]


gt=pd.read_csv("gt.txt",names=columns)

car=gt[gt["class"]==3]  


#%% os.chdir(current+"\\"+folder+"\\"+"img1")
    
video=cv2.VideoCapture("deneme.mp4")



if video.isOpened()==False:  #videonun açılıp açılmadığını kontrol ediyor
    print("Hata")

grand_truth=[]






video=cv2.VideoCapture("deneme.mp4")



if video.isOpened()==False:  #videonun açılıp açılmadığını kontrol ediyor
    print("Hata")

for i in range (np.max(gt["frame number"])-1):
    
    
    ret,frame=video.read()
   
    
    if ret==True:
        
        
        
       filter_id=np.logical_and(car["frame number"]==i+1,car["id"]==29)
       if len(car[filter_id])!=0:
           
            x=int(car[filter_id].left.values[0])
            y=int(car[filter_id].top.values[0])
            w=int(car[filter_id].width.values[0])
            h=int(car[filter_id].height.values[0])
            
            
            grand_truth.append([i,x,y,w,h,int(x+w/2),int(y+h/2)])
            cv2.rectangle(frame, (x,y),(x+w,y+h) , (255,0,0),3)
       cv2.putText(frame, f"Frame Number :{i}", (10,30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,0,0),2)
       time.sleep(0.01) #videoyu yavaşlatmak için        
       cv2.imshow("Takip", frame)
            
       if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    else:break       
            
            
            
cv2.destroyAllWindows()

df=pd.DataFrame(grand_truth,columns=["i","x","y","w","h","center_x","center_y"])



""" YADA
for i in range (np.max(gt["frame number"])-1):
    
    ret,frame=video.read()
   
    
    if ret==True:
        
        
        if i<450:
            x=int(car[car.id==29].iloc[i,2])
            y=int(car[car.id==29].iloc[i,3])
            w=int(car[car.id==29].iloc[i,4])
            h=int(car[car.id==29].iloc[i,5])
            cv2.rectangle(frame, (x,y),(x+w,y+h) , (255,0,0),3)
            
            cv2.putText(frame, f"Frame Number :{i}", (10,30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,0,0),2)
            
            #i=frame,x=left,y=top,w,h,center_x,center_y
            grand_truth.append([i,x,y,w,h,int(x+w/2),int(y+h/2)]) 
            time.sleep(0.01) #videoyu yavaşlatmak için
            cv2.imshow("Takip", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
               break
        else:
            cv2.putText(frame, f"Frame Number :{i}", (10,30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,0,0),2)
            time.sleep(0.01) #videoyu yavaşlatmak için    
            cv2.imshow("Takip", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
               break
           
         
            
    else:break        
            
            
            
cv2.destroyAllWindows()



df=pd.DataFrame(grand_truth,columns=["i","x","y","w","h","center_x","center_y"])


df.to_csv("grand_truth_new.txt",index=False)

"""

