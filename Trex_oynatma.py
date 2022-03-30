# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:35:15 2022

@author: Mehmet
"""

from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon={"top":350, "left":730, "width":250, "height":130}
sct=mss()

width=125
height=50

#model yükleme
model=model_from_json(open("model.json","r").read())
model.load_weights("trex_weight.h5")

labels=["Down","Right","Up"]

framerate_time=time.time()
counter=0
i=0
delay=0.508
key_down_pressed=False
sleep_time1=0.373
sleep_time2=0.17

while True:
    
    
    img=sct.grab(mon)
    im=Image.frombytes("RGB", img.size, img.rgb)
    im2=np.array(im.convert("L").resize((width,height)))
    im2=im2/255
    X=np.array([im2])
    X=X.reshape(X.shape[0],width,height,1)
    r=model.predict(X)
    result=np.argmax(r)
    
    
    
    if result==0:
        keyboard.press(keyboard.KEY_DOWN)
        time.sleep(0.01)
        key_down_pressed=True
        
        
    elif result==2:
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        if i <1200:
            time.sleep(sleep_time1)
        
        elif 1200<i and i<5000:
            time.sleep(sleep_time2)
            
        else:
            time.sleep(0.12)
            
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
        
        
    counter+=1
    
    if (time.time()-framerate_time)>1:
        counter=0
        framerate_time=time.time()
        if i<=1200:
            delay-=0.004989
            sleep_time1-=0.0062
        else:
            delay-=0.0065
            sleep_time2-=0.0045
        if delay<0:
            delay=0
        if sleep_time1<0:
            sleep_time1=0
            
        if sleep_time2<0:
            sleep_time2=0  
            
        print("-------------------")
        print(f"Down:{r[0,0]}\nRight:{r[0,1]}\nUp:{r[0,2]}\nframe : {i}")
        i+=1
   

