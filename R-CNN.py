# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:04:22 2022

@author: Mehmet
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json



def selective_search (image):
    
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()

    print("start")
    rects=ss.process()

   
    return rects 

    


image=cv2.imread("mnist.png")
cv2.imshow("window",image)




rects=selective_search(image)

proposals=[]
boxes=[]


for (x,y,w,h) in rects[:100]:
    
    roi=image[y:y+h,x:x+w]
    roi=cv2.resize(roi, dsize=(32,32),interpolation=cv2.INTER_LANCZOS4)
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    roi=img_to_array(roi)
    
    proposals.append(roi)
    boxes.append((x,y,x+w,y+h))
    
    
proposals=np.array(proposals,dtype="float64")
boxes=np.array(boxes,dtype="int32")

print("sınıflandırma")


model=model_from_json(open("model.json","r").read())
model.load_weights("mnist_trained.h5")

proba=model.predict(proposals)


number_list=[]
idx=[]

for i in range(len(proba)):
    max_prob=np.max(proba[i,:])
    if max_prob>0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i]))
        
for i in range(len(number_list)):
    j=idx[i]
    cv2.rectangle(image, (boxes[j,0],boxes[j,1]), (boxes[j,2],boxes[j,3]), (0,0,255),2)
    cv2.putText(image, str(np.argmax([proba[j]])), (boxes[j,0]+5,boxes[j,1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0),1)
    cv2.imshow("winname", image)








