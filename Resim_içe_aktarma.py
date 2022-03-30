# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:27:17 2022

@author: mehmet
"""

import cv2
veri=cv2.imread("messi-sb.jpg",0)

cv2.imshow("resim",veri)

k=cv2.waitKey(0) &0xFF

if k==27:
    cv2.destroyAllWindows()
    
elif k== ord ('s'):    
    cv2.imwrite ("messi_gray.png",veri)
   
    
    
    