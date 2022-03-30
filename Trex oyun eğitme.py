# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:42:13 2022

@author: Mehmet
"""

import glob
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,Conv2D,MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sbn
import warnings

warnings.filterwarnings("ignore")

images=glob.glob("./trex_game/*.png")

width=125
height=50

X=[]
Y=[]

for img in images:
    
    filename=os.path.basename(img)
    label=filename.split("_")[0]
    im=np.array(Image.open(img).convert("L").resize((width,height)))
    im=im/255   #normalizasyon
    X.append(im)
    Y.append(label)
    
X=np.array(X)
X=X.reshape(X.shape[0],width,height,1)

sbn.countplot(Y)

# Y encode işlemi için one hot encoder
y=Y.copy()
Y=pd.DataFrame(Y)
ohe=OneHotEncoder()
Y=ohe.fit_transform(Y.values).toarray()


#alternatif encode işlemi 
def onehot_labels(values):
    label_encoder=LabelEncoder()
    integer_encoded=label_encoder.fit_transform(values)
    onehot_encoder=OneHotEncoder(sparse=False)
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


#y=onehot_labels(y)

#verileri bölme 
x1,x2,y1,y2=train_test_split(X,Y,test_size=0.25,random_state=42)

#cnn model compile

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation="relu",input_shape=(width,height,1)))
model.add(Conv2D(64,kernel_size=(3,3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x1,y1,epochs=30,batch_size=64)


score_train=model.evaluate(x1,y1)
print("Eğitim Doğruluğu %",score_train[1]*100)

score_test=model.evaluate(x2,y2)
print("Test Verisi Doğruluğu %",score_test[1]*100)


#model kaydı
open("model.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")

tahmin=model.predict(x2)
tahmin=pd.DataFrame(tahmin)

for i in range(0,55):
    for j in range(0,3):
        tahmin.iloc[i,j]=np.round(tahmin.iloc[i,j])
        
tahmin=np.array(tahmin)





