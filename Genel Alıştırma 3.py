# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:23:05 2022

@author: Mehmet
"""
import cv2
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as mp

folder="coklu nesne algilama"
current=os.getcwd()

os.chdir(current+"\\"+folder+"\\"+"img1")
#%%


tracker_name="csrt"

video_path="video.mp4"
gt_path="gt.txt"



def track_compare(track1,video_path,gt_path):
    
    opencv_algo={"csrt"       :cv2.TrackerCSRT_create,
                 "kcf"        :cv2.TrackerKCF_create,
                 "boosting"   :cv2.legacy.TrackerBoosting_create,
                 "mil"        :cv2.TrackerMIL_create,
                 "tld"        :cv2.legacy.TrackerTLD_create,
                 "medianflow" :cv2.legacy.TrackerMedianFlow_create,
                 "mosse"      :cv2.legacy.TrackerMOSSE_create  } 
    
    columns=["frame_number","id","x","y","w","h","score","class","visibilty"]
    
    tracker1=opencv_algo[track1]()
    gt=pd.read_csv(gt_path,names=columns)
    gt=gt[gt.id==1]
    #parametreler
    initBB=None
    frame_no=[]
    f=0
    success_track_frame=0
    track_list=[]
    start_time=time.time()
    gt_list=[]
    
    
    
    cap=cv2.VideoCapture(video_path)
    while True:
        
        ret,frame=cap.read()
        
        
        if ret:
            frame=cv2.resize(frame, dsize=(960,540))
            (H,W)=frame.shape[:2]
            
            car_gt=gt[gt.frame_number==f]
            
            if len (car_gt)!=0:
                x_gt=int(car_gt.x.values[0]/2)
                y_gt=int(car_gt.y.values[0]/2)
                w_gt=int(car_gt.w.values[0]/2)
                h_gt=int(car_gt.h.values[0]/2)
                
                center_x=int(x_gt+w_gt/2)
                center_y=int(y_gt+h_gt/2)
                
                cv2.rectangle(frame, (x_gt,y_gt), (x_gt+w_gt,y_gt+h_gt), (0,255,255),2)
                cv2.circle(frame, (center_x,center_y), 4, (255,0,0),-1)
                
            
            if initBB is not None:
                (success,box)=tracker1.update(frame)
                
                if f<=np.max(gt.frame_number):
                    
                    (x,y,w,h)=[int(i) for i in box]
                    
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
                    success_track_frame+=1
                    track_center_x=int(x+w/2)
                    track_center_y=int(y+h/2)
                    track_list.append([f,track_center_x,track_center_y])
                    gt_list.append([f,center_x,center_y])
                    
                info=[("Tracker :" , track1),
                      ("Success :","Yes" if success else "No" )]
                
                for (i,(j,k)) in enumerate(info):
                    
                    cv2.putText(frame, f"{j}{k}", (10,H-((i*20)+10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
                    
            cv2.putText(frame, f"{f}", (10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255),2)    
            time.sleep(0.01)
            cv2.imshow(f"{track1} takip algoritmasi", frame)
            
            frame_no.append(f)
            f+=1
            
            key=cv2.waitKey(1) &0xFF
            
            if key==ord("t"):
                initBB=cv2.selectROI(f"{track1} takip algoritmasi", frame,fromCenter=False)
                tracker1.init(frame,initBB)
            
            elif key ==ord("q"):break

        else:break

    cap.release()
    cv2.destroyAllWindows()
   
        
    stop_time=time.time()
    time_dif=stop_time-start_time

    #değerlendirme
    track_df=pd.DataFrame(track_list,columns=["frame_no","x_center","y_center"])
    gt_list_df=pd.DataFrame(gt_list,columns=["frame_no","x_center","y_center"])


    
    print (f"Track Metod : {track1} ")
    print(f"Time: {time_dif}")
    print (f"Number of frame to track (gt) :{len(gt)}")
    print(f"Number of frame to track (track success): {success_track_frame}")
            
        
            
    gt_center_x=gt_list_df.x_center.values
    gt_center_y=gt_list_df.y_center.values
    
    
    track_df_center_x=track_df.x_center.values
    track_df_center_y=track_df.y_center.values
            
           
    mp.plot(np.sqrt((gt_center_x-track_df_center_x)**2+(gt_center_y-track_df_center_y)**2))
    mp.xlabel("frame")
    mp.ylabel("Öklid mesafesi btw gt ve track")
    
    error=np.sum((gt_center_x-track_df_center_x)**2+(gt_center_y-track_df_center_y)**2)
            
    print(f"Toplam Hata : {error}")
    
    return

track_compare(tracker_name, video_path, gt_path)
        