# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:18:17 2022

@author: Mehmet
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mp
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


path="Fruit_360"

mylist=os.listdir(path)

classes=len(mylist)

print("Label (Sınıf) Sayısı",classes)

images=[]
class_name=[]

for i in range(classes):
    class_fruit=os.listdir(path+"\\"+mylist[i])
    
    for j in class_fruit:
        img=cv2.imread(path+"\\"+mylist[i]+"\\"+j)
        img=cv2.resize(img, (32,32))
        images.append(img)
        class_name.append(mylist[i])
        
print(len(images))
print(len(class_name))         
#%%
images=np.array(images)
class_name=np.array(class_name)


#train,test,validation split

x1,x2,y1,y2=train_test_split(images,class_name,test_size=0.5,random_state=42)

x1,x_validation,y1,y_validation=train_test_split(x1,y1,test_size=0.2,random_state=42)


print(f"Resim Sayısı :{images.shape[0]}\nEğitim Verisi Sayısı: {x1.shape[0]}\nTest Verisi Sayısı: {x2.shape[0]}\nDoğrulama Versisi Sayısı:{x_validation.shape[0]}")

#preprocessing
def pre_process(img) :
    img=img/255 #normalize
    return img




x1=np.array(list(map(pre_process,x1)))
x2=np.array(list(map(pre_process,x2)))
x_validation=np.array(list(map(pre_process,x_validation)))

x1=x1.reshape(-1,32,32,3)
x2=x2.reshape(-1,32,32,3)
x_validation=x_validation.reshape(-1,32,32,3)

#data generate

data_gen=ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            rotation_range=90)

data_gen.fit(x1)


#encoding label encoeder kullandık
lb=LabelEncoder()
encoder=lb.fit(class_name)
#daha sonra kullanacağımız için verinin classlarını kaydettik
np.save("classes.npy",encoder.classes_)
#sözel veriyi sayısal veriye çevirdik
y1=lb.transform(y1)
y2=lb.transform(y2)
y_validation=lb.transform(y_validation)

#process
model=Sequential()
model.add(Conv2D(input_shape=(32,32,3),filters=32, kernel_size=(3,3),activation="relu",padding="same"))
#padding pixel ekleme yaparak kayıp yaşanamasını engelliyor
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu",padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu",padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))


#düzleştirme
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))

#çıkış katı
model.add(Dense(units=classes, activation="softmax"))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer="adam",metrics=["accuracy"])

batch_size=150


hist=model.fit_generator(data_gen.flow(x1,y1,batch_size=batch_size),
                                       validation_data=(x_validation,y_validation),epochs=8,
                                       steps_per_epoch=x1.shape[0]//batch_size,shuffle=1)





open("model.json","w").write(model.to_json())
model.save_weights("fruit_classification.h5")


#%%

hist.history.keys()

mp.figure()
mp.plot(hist.history["loss"],label="Eğitim Loss")
mp.plot(hist.history["val_loss"],label="Val Loss")
mp.legend()

mp.figure()
mp.plot(hist.history["accuracy"],label="Eğitim Doğruluk")
mp.plot(hist.history["val_accuracy"],label="Val Doğruluk")
mp.legend()


score=model.evaluate(x2,y2,verbose=1)
print("Test loss:",score[0])
print("Test Doğruluk",score[1])


tahmin=model.predict(x_validation)
tahmin_class=np.argmax(tahmin,axis=1)




cm=confusion_matrix(y_validation, tahmin_class)




