# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:22:08 2022

@author: Mehmet

Takip Algoritmaları

-BOOSTİNG takip algoritması
-Tracking learning and detection (TLD) algoritması
-Medianflow algoritması
-Mosse (Minimum Output sum of squared error) algoritması
-CSRT Takip algoritması
-KCF(kernelized correlation filters)algoritması
MIL(multiple instance learning) algoritması

"""

import cv2
import os
import numpy as np
import  pandas as pd
import matplotlib.pyplot as mp
import time
import cv2.legacy

folder="Kesifsel Analiz"
current=os.getcwd()

os.chdir(current+"\\"+folder+"\\"+"img1")
#%%

opencv_algo={"csrt"       :cv2.TrackerCSRT_create,
             "kcf"        :cv2.TrackerKCF_create,
             "boosting"   :cv2.legacy.TrackerBoosting_create,
             "mil"        :cv2.TrackerMIL_create,
             "tld"        :cv2.legacy.TrackerTLD_create,
             "medianflow" :cv2.legacy.TrackerMedianFlow_create,
             "mosse"      :cv2.legacy.TrackerMOSSE_create  } 

tracker_name="mosse"

print(tracker_name)
tracker=opencv_algo[tracker_name]()

gt=pd.read_csv("grand_truth_new.txt")

video_path="deneme.mp4"







cap=cv2.VideoCapture(video_path)
tracker=opencv_algo[tracker_name]()
#parametreler
initBB=None
fps=25
frame_number=[]
f=0
success_track_frame=0
track_list=[]
start_time=time.time()

while True:
    
    ret,frame=cap.read()
    
    
    if ret:
        frame=cv2.resize(frame, dsize=(960,540))
        (H,W)=frame.shape[:2]
        
        car_gt=gt[gt.i==f]
        
        if len (car_gt)!=0:
            x=int(car_gt.x.values[0]/2)
            y=int(car_gt.y.values[0]/2)
            w=int(car_gt.w.values[0]/2)
            h=int(car_gt.h.values[0]/2)
            
            center_x=int(car_gt.center_x.values[0]/2)
            center_y=int(car_gt.center_y.values[0]/2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
            cv2.circle(frame, (center_x,center_y), 4, (255,0,0),-1)
            
        
        if initBB is not None:
            (success,box)=tracker.update(frame)
            
            if f<=np.max(gt.i):
                
                (x,y,w,h)=[int(i) for i in box]
                
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
                success_track_frame+=1
                track_center_x=int(x+w/2)
                track_center_y=int(y+h/2)
                track_list.append([f,track_center_x,track_center_y])
                
            info=[("Tracker :" , tracker_name),
                  ("Success :","Yes" if success else "No" )]
            
            for (i,(j,k)) in enumerate(info):
                
                cv2.putText(frame, f"{j}{k}", (10,H-((i*20)+10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
                
        cv2.putText(frame, f"{f}", (10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255),2)    
        time.sleep(0.01)
        cv2.imshow(f"{tracker_name} takip algoritmasi", frame)
        
        frame_number.append(f)
        f+=1
        
        key=cv2.waitKey(1) &0xFF
        
        if key==ord("t"):
            initBB=cv2.selectROI(f"{tracker_name} takip algoritmasi", frame,fromCenter=False)
            tracker.init(frame,initBB)
        
        elif key ==ord("q"):break

    else:break

cap.release()
cv2.destroyAllWindows()

stop_time=time.time()
time_dif=stop_time-start_time

#değerlendirme
track_df=pd.DataFrame(track_list,columns=["frame_no","x_center","y_center"])



if len(track_df) !=0:
    print (f"Track Metod : {tracker_name} ")
    print(f"Time: {time_dif}")
    print (f"Number of frame to track (gt) :{len(gt)}")
    print(f"Number of frame to track (track success): {success_track_frame}")
    
    track_df_frame=track_df.frame_no
    
    gt_center_x=gt.center_x[track_df_frame].values
    gt_center_y=gt.center_y[track_df_frame].values


    track_df_center_x=track_df.x_center.values
    track_df_center_y=track_df.y_center.values
    
    
    mp.plot(np.sqrt((gt_center_x-track_df_center_x)**2+(gt_center_y-track_df_center_y)**2))
    mp.xlabel("frame")
    mp.ylabel("Öklid mesafesi btw gt ve track")
    error=np.sum((gt_center_x-track_df_center_x)**2+(gt_center_y-track_df_center_y)**2)
    
    print(f"Toplam Hata : {error}")
    
    
    