# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:22:15 2022

@author: Mehmet
"""
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

def sliding_window(image,step,ws):
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            yield(x,y,image[y:y+ws[1],x:x+ws[0]])
            
            
def image_pyramid(image,scale=1.5,minSize=(224,224)):
    
    yield image
    
    while True:
        w=int(image.shape[1]/scale)
        image=cv2.resize(image,dsize=(w,w))
        
        
        if image.shape[0]<minSize[1] or image.shape[1]<minSize[0]:
            yield image
            break
        
        yield image

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
            
width=600
height=600
pyr_scale=1.5
win_step=16
roi_size=(200,150)
input_size=(224,224)

print("Resnet Yükleniyor")
model=ResNet50(weights="imagenet",include_top=True)

data=cv2.imread("husky.jpg")
data=cv2.resize(data, dsize=(width,height))
cv2.imshow("husky",data)

#%%

(H,W)=data.shape[:2]

#image pyramid

pyramid=image_pyramid(data,pyr_scale,roi_size)

rois=[]
locs=[]

for image in pyramid:
    scale=W/float(image.shape[0])
    
    for (x,y,Roi_data) in sliding_window(image, win_step, roi_size):
        x=int(x*scale)
        y=int(y*scale)
        w=int(roi_size[0]*scale)
        h=int(roi_size[1]*scale)
        
        roi=cv2.resize(Roi_data, input_size)
        roi=img_to_array(roi)
        roi=preprocess_input(roi)
        

        rois.append(roi)
        locs.append((x,y,x+w,y+h))

rois=np.array(rois,dtype="float32")

print("sınıflandırma işlemi")
preds=model.predict(rois)


preds=imagenet_utils.decode_predictions(preds,top=1)  

      
labels={}

for  (i,p) in enumerate(preds):
    (_,label,prob)=p[0]
    if prob>0.89:
        box=locs[i]
        L=labels.get(label,[])
        L.append((box,prob))
        labels[label]=L
        
for label in labels.keys():
     
    c_img=data.copy()
    for (box,prob) in labels[label]:
        (x,y,endx,endy)=box
        cv2.rectangle(c_img, (x,y), (endx,endy), (0,255,0),2)
    
    cv2.imshow("ilk",c_img)        
    
    c_img=data.copy()
    
    # non maxima
    
    boxes=np.array([p[0] for p in labels[label]])
    proba=np.array ([p[1] for p in labels[label]]) 
    
    boxes=non_maxi_suppression(boxes,proba)
    
    for (x,y,endx,endy) in boxes:
        cv2.rectangle(c_img, (x,y), (endx,endy), (0,255,0),2)
        name_y=y-10 if y-10>10 else y+10
        cv2.putText(c_img, label, (x,name_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0))
    cv2.imshow("son", c_img)
    
    





