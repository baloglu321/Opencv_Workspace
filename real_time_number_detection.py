# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:11:40 2022

@author: Mehmet
"""

import cv2
import numpy as np
from keras.models import model_from_json
import warnings

warnings.filterwarnings("ignore")


def pre_process(img) :
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255 #normalize
    return img


cap=cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)


model=model_from_json(open("model.json","r").read())
model.load_weights("trained_model.h5")





while True:
    ret,frame=cap.read()
    img=np.asarray(frame)
    img=cv2.resize(img,(32,32))
    img=pre_process(img)
    img=img.reshape(1,32,32,1)
    
    #predict 
    predict=model.predict(img) 
    
    class_index=str(int(np.argmax(predict,axis=1)))
    
    predictions=model.predict(img)
    prop_val=np.amax(predictions)
    print(class_index,prop_val)
    
   
    if prop_val>0.65:
        prop_val=str(prop_val)
        cv2.putText(frame, f"Tahmin edilen sayi: {class_index}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),1)
        cv2.putText(frame, f"Yuzde: {prop_val}", (10,60), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),1)
            
    cv2.imshow("Number_Detection",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

    
    
cap.release()
cv2.destroyAllWindows()
  




