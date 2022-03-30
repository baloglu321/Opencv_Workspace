# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:53:23 2022

@author: mehmet
"""

import cv2
import numpy as np


img=cv2.imread("Kart.png")

cv2.imshow("resim", img)

pt1=np.float32([[(45,92),(144,489),(352,30),(449,429)]]) #perspektifi bozuk resmin köşe kordinatları (sol üst:1, sol alt: 2, sağ üst :3, sağ alt :4 )
pt2=np.float32([[(0,0),(0,500),(512,0),(512,500)]])  #bu köşe kordinatlarının gitmesini istediğimiz kordinantlar

matrix=cv2.getPerspectiveTransform(pt1, pt2)#transformasyon matrisini al
print(matrix)

img_r=cv2.warpPerspective(img,matrix,(512,500))
cv2.imshow("rotate image", img_r)