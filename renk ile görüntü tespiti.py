# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:36:55 2022

@author: Mehmet
"""
import cv2
import matplotlib.pyplot as mp
import numpy as np
from collections import deque


#kontur aynı renk veya yoğunluğa sahip noktlaarı birleştirilmesine denir

#nesne merkezi depolama için
pts=deque(maxlen=16)

#mavi renk aralığı belileyelim (bu aralık kamerada hangi rengi göreceğine göre değişir)
#sırayla hsv yani ton, doygunluk,parlaklık aralığı buu renk aralığı en kolay hsv renk kodu ile oluşturuşturulur
blue_lower=(85,98,50)
blue_upper=(160,255,230)

#capture
cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success,img=cap.read()
    
    
    if success:
        
        #görüntü blurlama
        blur_img=cv2.GaussianBlur(img,(15,15), 0)
        #görüntüyü BGR dan HSV ye dönüştürme
        hsv=cv2.cvtColor(blur_img,cv2.COLOR_BGR2HSV)
        
        #hsv görüntü al
        
        #cv2.imshow("HSV Görüntü", hsv)
        
        #mavi için maske oluştur
        mask=cv2.inRange(hsv, blue_lower,blue_upper)
        #cv2.imshow("Maskelenmiş Görüntü", mask)
       
        
       #gürültüleri engellemek için erezyon + genişleme
        
        e_img=cv2.erode(mask, None)
        d_img=cv2.dilate(e_img, None)
        #cv2.imshow("Açılmış Görüntü", d_img)
        
        #kontur algılama
        (contours,_)=cv2.findContours(d_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center=None
        if len(contours)>0:
            #max veya min ile en yakındaki yada uzaktaki nesne alınabilir
            c=max(contours,key=cv2.contourArea)
          
            #nesnenin etrafını kare ile çiz
            rect=cv2.minAreaRect(c)
            ((x,y),(width,height),rotation)=rect
            x=np.round(x)
            y=np.round(y)
            width=np.round(width)
            height=np.round(height)
            rotation=np.round(rotation)
            print(f"Nesnenin Konumu:{(x,y)}\nNesnenin Yüksekliği ve Genişliği:{(width,height)}\nNesnenin Açısı:{rotation}")
            
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            
            #moment : görüntünün merkezini bulmamıza yarayan yapı
            
            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            
            #konturu çizdir
            cv2.drawContours(img,[box], 0, (0,255,255),2)
            
            #merkez noktası ekleme(-1 doldur demek)
            
            cv2.circle(img,center,5,(255,0,255),-1)
            
            #bilgileri ekrana yazsın
            
            cv2.putText(img, f"x,y:{(x,y)} Y,G :{(width,height)} rotation:{rotation}", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)
            
            #cv2.imshow("Son Görüntü", img)
            
            #takip algoritması
        
       
        pts.append(center)
        for i in range (1,len(pts)):
            
            if pts[i-1] is None or pts[i] is None:
                continue
            cv2.line(img, pts[i-1], pts[i], (0,255,0),3)
                        
        cv2.imshow("Son Görüntü", img)
        
        
    if cv2.waitKey(1) &0xFF ==ord("q"):
      

        break


cap.release()
cv2.destroyAllWindows()











