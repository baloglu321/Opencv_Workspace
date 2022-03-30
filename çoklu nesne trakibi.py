# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:22:41 2022

@author: Mehmet
"""

import cv2
import os
import pandas as pd


folder="coklu nesne algilama"
current=os.getcwd()

os.chdir(current+"\\"+folder+"\\"+"img1")
#%%
columns=["frame number","id","left","top","width","height","score","class","visibilty"]

files=os.listdir()

images=[]
for i in files:
    if i.endswith(".jpg"):
        images.append(i)
        
    else:
        gt_df=pd.read_csv(i,names=columns)
        
        
opencv_algo={"csrt"       :cv2.TrackerCSRT_create,
             "kcf"        :cv2.TrackerKCF_create,
             "boosting"   :cv2.legacy.TrackerBoosting_create,
             "mil"        :cv2.TrackerMIL_create,
             "tld"        :cv2.legacy.TrackerTLD_create,
             "medianflow" :cv2.legacy.TrackerMedianFlow_create,
             "mosse"      :cv2.legacy.TrackerMOSSE_create  } 

img=cv2.imread(images[42])
fps=30
f=0
size=(img.shape[1],img.shape[0])

out=cv2.VideoWriter("video.mp4",cv2.VideoWriter_fourcc(*"MP4V"),fps,size,True)


for i in images:
    print(i)

    
    img=cv2.imread(i)
    out.write(img)
    
    
out.release() 
        
#%% 

tracker_name="mosse"
trackers=cv2.legacy.MultiTracker_create() #tanımlama farklı
fps=30
f=0
cap=cv2.VideoCapture("video.mp4")

while True:
    
    ret,frame=cap.read()
    
    
    if ret:
        frame=cv2.resize(frame, dsize=(960,540))
        (H,W)=frame.shape[:2]
        
        success,box=trackers.update(frame)
        
        info=[("Tracker :" , tracker_name),
              ("Success :","Yes" if success else "No" )]
        for (i,(j,k)) in enumerate(info):
            
            cv2.putText(frame, f"{j}{k}", (10,H-((i*20)+10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)        
        
        
        for box in box:
            (x,y,w,h)=[int(i) for i in box]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
            
        cv2.imshow(f"{tracker_name} takip algoritmasi",frame)
        f+=1
        
        key=cv2.waitKey(1) &0xFF
        
        if key==ord("t"): #bu kısım farklı 
            box=cv2.selectROI(f"{tracker_name} takip algoritmasi", frame,fromCenter=False)
            tracker=opencv_algo[tracker_name]()
            trackers.add(tracker,frame,box)
        
        elif key ==ord("q"):break

    else:break
    
cap.release()
cv2.destroyAllWindows()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            