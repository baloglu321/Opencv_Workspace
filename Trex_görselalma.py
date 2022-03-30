# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:24:17 2022

@author: Mehmet

"""

#veri toplama

import keyboard
import uuid
import time
from PIL import Image
from mss import mss

"https://www.trex-game.skipser.com/"


mon={"top":350, "left":730, "width":250, "height":130}

sct=mss()
i=0
def record_screen(record_id,key):
    global i
    i+=1
    print(f"{key},{i}")
    img=sct.grab(mon)
    im=Image.frombytes("RGB", img.size, img.rgb)
    im.save(f"./trex_game/{key}{i}.png" )
    
is_exit=False

def exit():
    global is_exit
    is_exit=True



keyboard.add_hotkey("esc",exit)

record_id=uuid.uuid4()



while True:
    
    
    if is_exit:break
    
    
    
    
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id,"up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id,"down")
            time.sleep(0.1)
            
        elif keyboard.is_pressed("right"):
            record_screen(record_id,"right")
            time.sleep(0.1)

    except RuntimeError: continue

    
    
    
    
    
    
    





