# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:39:24 2022

@author: mehmet
"""

import cv2
import time


video_name="CULZ8gmW4AAj5EQ.mp4"

video=cv2.VideoCapture(video_name)

wide=video.get(3)
hide=video.get(4)

print(f"Video Genişliği : {wide}")
print(f"Video Yüksekliği : {hide}")

if video.isOpened()==False:  #◘videonun açılıp açılmadığını kontrol ediyor
    print("Hata")

while True:    #sonsuz döngü
    
    ret,frame=video.read()   #videodaki tüm frameler tek tek frame e gider frame bitince ret =False olur.

    if ret==True:
        time.sleep(0.08) #videoyu yavaşlatmak için
        cv2.imshow("video",frame)
        
    else:
        break
    
    
    if cv2.waitKey(1) &0xFF == ord("q"):
        break
    
    
video.release()    
cv2.destroyAllWindows()