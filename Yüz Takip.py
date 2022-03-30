# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:02:07 2022

@author: Mehmet
"""

import cv2
import numpy as np

cap=cv2.VideoCapture(0)
ret,frame=cap.read()

if ret==False:
    print("Uyarı")
    

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rects=face_cascade.detectMultiScale(frame)

(x,y,w,h)=tuple(rects[0])
track_window=(x,y,w,h)


#region of intrest
roi=frame[y:y+h,x:x+w]

hsv_roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


roi_hist=cv2.calcHist([hsv_roi],[0] , None, [180], [0,180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit=( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)


while True:
    
    
    ret,frame=cap.read()
    
    
    if ret:
        
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        dst=cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        
        ret,track_window=cv2.meanShift(dst, track_window, term_crit)
        
        x,y,w,h = track_window
        
        img2=cv2.rectangle(frame, (x,y), (x+w,y+h) ,(0,0,255),3)
        
        cv2.imshow("Takip", img2)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        




cap.release()
cv2.destroyAllWindows()










