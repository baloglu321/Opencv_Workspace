# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:59:41 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp


def image_pyramid(image,scale=1.5,minSize=(224,224)):
    
    yield image
    
    while True:
        w=int(image.shape[1]/scale)
        image=cv2.resize(image,dsize=(w,w))
        
        
        if image.shape[0]<minSize[1] or image.shape[1]<minSize[0]:
            yield image
            break
        
        yield image
        
"""    
img=cv2.imread("damla.png")
im=image_pyramid(img,1.5,(100,100))

for i,image in enumerate(im):
    print(i)
    mp.imshow(image)
        
"""    