# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:16:51 2022

@author: Mehmet

-veri seti oluştur
 p,n
-cascade programı indir :https://amin-ahmadi.com/cascade-trainer-gui/
-cascade programına p ve n dosyalarını göster ve negatif örnek sayısını belirt
-trainerden çıkan cascade classification dosyasına göre algoritmanı yaz

"""

import cv2
import os

path="images"

imgwidht=180
imgheight=120

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,120)

global counFolder

def saveDataFunc():
    global countFolder
    countFolder=0
    while os.path.exists(path+str(countFolder)):
        countFolder+=1
    
    os.makedirs(path+str(countFolder)) 

saveDataFunc()


count=0
countSave=0

while True:
    
    
    success,img=cap.read()
    
    if success:
        
        
        img=cv2.resize(img,(imgwidht,imgheight))
        
        
        if count % 5==0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png", img)
            countSave+=1
            print(countSave)
        
        count+=1
        
        cv2.imshow("image", img)
    
        
    
    if cv2.waitKey(1) &0xFF==ord("q"):
        break
    
    
cap.release()
cv2.destroyAllWindows()
    








    