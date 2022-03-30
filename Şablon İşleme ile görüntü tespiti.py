# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:13:30 2022

@author: Mehmet

"""
#Şablon Eşleme :Template Matching 

import cv2
import matplotlib.pyplot as mp

# şablonu resimden çıkardım ama normalde resim kırpılarak şablon ayrıca okutulur.


img=cv2.imread("Lenna_(test_image).png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_sablon=img[50:385,117:450]

boyut=((img.shape),(img_sablon.shape))

height,widht=img_sablon.shape[:2]

print(f"Orjinal Resim Boyurtu :{boyut[0]}\nŞablon Resim Boyutu : {boyut[1]}")

mp.figure()
mp.imshow(img)
mp.axis("off")


mp.figure()
mp.imshow(img_sablon)
mp.axis("off")


#tüm metodları deneyip buna göre sonuçlara bakmak için for döngüsü ile şablon eşleme metodlarını dolaşıyoruz.

methods=["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED",
         "cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]


for i in methods:
    img=cv2.imread("Lenna_(test_image).png")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    method=eval(i) # string olarak tanımlanmış olduğumuz metodaları metoda çevirmek için eval kullandık
    
    res= cv2.matchTemplate(img, img_sablon, method)
    print(res.shape)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        
        top_left=min_loc
    else:
        top_left=max_loc
        
    bottom_right=(top_left[0]+widht,top_left[1]+height)
    
    cv2.rectangle(img, top_left, bottom_right, (255,255,0),2)
    
    mp.figure()
    mp.subplot(121),mp.imshow(res),mp.title("Eşleşen Sonuç"),mp.axis("off")
    mp.subplot(122),mp.imshow(img),mp.title("Tespit Edilen Sonuç"),mp.axis("off")
    mp.suptitle(i)
    mp.show()

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    