# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:51:08 2022

@author: mehmet
"""

import cv2
import numpy as np

veri=np.zeros((512,512,3),np.uint8)



#çizgi
#(çizilecek resim ismi,(başlangıç noktası),(bitiş noktası),(rgb şeklinde rengi),kalınlık değeri)
cv2.line(veri,(0,0),(512,512),(0,255,0),3)
#kare
cv2.rectangle(veri,(50,50),(250,250),(0,0,255),3)



#daire
cv2.circle(veri,(350,350),50, (255,0,0),3)
cv2.circle(veri,(450,450),50,(255,0,150),cv2.FILLED)
cv2.imshow("çizgili ve çemberli kare",veri)




#metin ekleme
cv2.putText(veri, "Kare", (50,300), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
cv2.putText(veri, "Daire", (300,425), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
cv2.putText(veri, "Dolu Daire", (415,385), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,150))
cv2.putText(veri, "Cizgi", (275,275), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
cv2.imshow("çizgili ve çemberli kare",veri)


if cv2.waitKey(0) &0xFF==ord("q"):
    cv2.destroyAllWindows()