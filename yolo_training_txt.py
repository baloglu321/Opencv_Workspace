# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:39:36 2022

@author: Mehmet
"""

import os
import pandas as pd


img=[".jpg",".JPG",".png",".PNG"]


path=os.getcwd()

path_i=path+"\\"+"obj"

imglist=os.listdir(path_i)



liste=[]



for i in imglist:
    
    for j in img:
        if i.endswith(j):
            a=(path+"\\"+ i)
            liste.append(a)
            
            
list_df=pd.DataFrame(liste)

list_df.to_csv("train.txt")
         
""" 
train=path+"\\"+"train.txt"
valid=path+"\\"+"train.txt"
names=path+"\\"+"obj.names"    
clases=["klaslar buraya"]

obj_data={"classes = 2"
,"train  ="+train
,"valid  = "+valid
,"names = "+names
,"backup = backup/"}

      
clases_df=pd.DataFrame(clases)

clases_df.tocsv("obj.names")            

obj_data_df=pd.DataFrame(obj_data,index=None)

obj_data_df.to_csv("obj.data")
"""