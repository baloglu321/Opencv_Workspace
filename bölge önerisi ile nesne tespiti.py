# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:02:46 2022

@author: Mehmet
"""

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

def non_maxi_suppression(boxes,probs=None,overLapThresh=0.3):
    
    if len(boxes)==0:
        return []
    
    if boxes.dtype.kind=="i":
        boxes=boxes.astype("float")
        
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    
    #alanın bulunması
    area=(x2-x1+1)*(y2-y1+1)
    
    idxs=y2
    
    #olasılık değeri
    if probs is not None:
        idxs=probs
        
    idxs=np.argsort(idxs)
    pick=[]
    
    while len(idxs) >0:
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        
        #enbüyük ve en küçük x ve y
        xx1=np.maximum(x1[i],x1[idxs[:last]])
        yy1=np.maximum(y1[i],y1[idxs[:last]]) 
        xx2=np.minimum(x2[i],x2[idxs[:last]]) 
        yy2=np.minimum(y2[i],y2[idxs[:last]]) 
        
        
        # w,h bulma
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        
        #overlap
        overlap=(w*h/area[idxs[:last]])
        
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>overLapThresh)[0])))
        
        
    return boxes[pick].astype("int")      

def selective_search (image):
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()

    print("start")
    rects=ss.process()

    output=image.copy()

    for (x,y,w,h) in rects:
        color=[np.random.randint(0,255) for j in range(0,3)]
        cv2.rectangle(output, (x,y), (x+w,y+h), color,2)
        
    cv2.imshow("output",output)
    return rects[:1000]  
    
model=ResNet50(weights="imagenet")

image=cv2.imread("animals.jpg")
image=cv2.resize(image, (600,600))
cv2.imshow("image",image)
(H,W)=image.shape[:2]


rects=selective_search(image)

propa=[]
boxes=[]

for (x,y,w,h) in rects:
    
    if w/float(W)<0.1 or h/float(H)<0.1:continue
    
    roi=image[y:y+h,x:x+h]
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    roi=cv2.resize(roi,(224,224))
    
    roi=img_to_array(roi)
    
    propa.append(roi)
    boxes.append((x,y,w,h))
    
propa=np.array(propa)

print("predict")
preds=model.predict(propa)
preds=imagenet_utils.decode_predictions(preds,top=1)

      
labels={}

for  (i,p) in enumerate(preds):
    (_,label,prob)=p[0]
    if prob>0.8:
        (x,y,w,h)=boxes[i]
        box=(x,y,x+w,y+h)
        L=labels.get(label,[])
        L.append((box,prob))
        labels[label]=L
        
        
c_img=image.copy()
        
for label in labels.keys():
     
   
    
    # non maxima
    for (box,prop) in labels[label]:
        boxes=np.array([p[0] for p in labels[label]])
        proba=np.array ([p[1] for p in labels[label]]) 
    
        boxes=non_maxi_suppression(boxes,proba)
    
        for (x,y,endx,endy) in boxes:
            cv2.rectangle(c_img, (x,y), (endx,endy), (0,255,0),2)
            name_y=y-10 if y-10>10 else y+10
            cv2.putText(c_img, label, (x,name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0))
    cv2.imshow("son", c_img)
    