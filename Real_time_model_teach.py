# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:16:11 2022

@author: Mehmet
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sbn
import matplotlib.pyplot as mp
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
import warnings

warnings.filterwarnings("ignore")


path="mnist_data"

mylist=os.listdir(path)

classes=len(mylist)

print("Label (Sınıf) Sayısı",classes)


images=[]
class_no=[]

for i in range(classes):
    my_img_list=os.listdir(path+"\\"+str(i))
    for j in my_img_list:
        img=cv2.imread(path+"\\"+str(i)+"\\"+j,1)
        img=cv2.resize(img,(32,32))
        images.append(img)
        class_no.append(i)
        
print(len(images))
print(len(class_no)) 

images=np.array(images)
class_no=np.array(class_no)       


print(images.shape)
print(class_no.shape) 



x1,x2,y1,y2=train_test_split(images,class_no,test_size=0.5,random_state=42)
x1,x_validation,y1,y_validation=train_test_split(x1,y1,test_size=0.2,random_state=42)

print(images.shape)
print(x1.shape)
print(x2.shape)
print(x_validation.shape)

#visualization
fig,axes=mp.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sbn.countplot(y1,ax=axes[0])
axes[0].set_title("y_train")

sbn.countplot(y2,ax=axes[1])
axes[1].set_title("y_test")

sbn.countplot(y_validation,ax=axes[2])
axes[2].set_title("y_validation")



#preprocessing
def pre_process(img) :
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255 #normalize
    return img


"""
img=pre_process(x1[105])
img=cv2.resize(img, (300,300))
cv2.imshow("winname", img)    
"""


x1=np.array(list(map(pre_process,x1)))
x2=np.array(list(map(pre_process,x2)))
x_validation=np.array(list(map(pre_process,x_validation)))

x1=x1.reshape(-1,32,32,1)
x2=x2.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-1,32,32,1)

#data generate
data_gen=ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            rotation_range=10)

data_gen.fit(x1)

#encode işlemi

y1= to_categorical(y1,classes)
y2= to_categorical(y2,classes)
y_validation= to_categorical(y_validation,classes)

#process
model=Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5),input_shape=(32,32,1),
                 activation="relu",padding="same"))
#padding pixel ekleme yaparak kayıp yaşanamasını engelliyor
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(5,5),activation="relu",padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))

#ezberlemeyi engellemek için dropout
model.add(Dropout(0.2))
#düzleştirme
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.2))
#çıkış katı
model.add(Dense(units=classes, activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=("adam"),metrics=["accuracy"])

batch_size=250


hist=model.fit_generator(data_gen.flow(x1,y1,batch_size=batch_size),
                                       validation_data=(x_validation,y_validation),epochs=10,
                                       steps_per_epoch=x1.shape[0]//batch_size,shuffle=1)


"""
#pickle ile model kaydetme
with open("trained_model.p","wb") as pickle_out :
    pickle.dump(model,pickle_out)
    pickle_out.close()

"""



open("model.json","w").write(model.to_json())
model.save_weights("trained_model.h5")


#%% değerlendirme

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
y_gercek=np.argmax(y_validation,axis=1)



cm=confusion_matrix(y_gercek, tahmin_class)


f,ax=mp.subplots(figsize=(8,8))
sbn.heatmap(cm,annot=True,linewidths=0.01,linecolor="gray",fmt=".1f",cmap="Greens",ax=ax)
mp.xlabel("Tahmin Değerleri")
mp.ylabel("Gerçek Değerler")


