# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:35:46 2022

@author: Mehmet

"""
import cv2
import matplotlib.pyplot as mp
import numpy as np

img=cv2.imread("datai_team.png",0)

mp.figure()
mp.imshow(img,cmap="gray")
mp.axis("off")
mp.title("orjinal resim")



#Morfolojik Operasyonlar : 5 çeşittir başlık başlık açacak olursak

#erezyon :ön plandaki nesnenin sınırlarını aşındırır
kernel = np.ones((5,5),dtype=np.uint8)
e_img=cv2.erode(img, kernel,iterations=2) #burada iterasyon yükseltilirse görüntüdeki kalınlık düşer
mp.figure()
mp.imshow(e_img,cmap="gray")
mp.axis("off")
mp.title("erezyonlu resim")

#genişleme(dilation) : görüntüdeki beyaz bölgeyi artırır erezyonun tam tersidir
d_img=cv2.dilate(img, kernel,iterations=2)
mp.figure()
mp.imshow(d_img,cmap="gray")
mp.axis("off")
mp.title("genişletilmiş resim")

#açılma : sırayla erezyon + genişleme uygulanmasıdır gürültülü resimlerde gürültüyü azaltmak için kullanılabilir küçük beyaz noktaları kapatmak için kullanılır.
white_noise=np.random.randint(0,2,size=img.shape[:2])
white_noise=white_noise*255
 
n_img=white_noise+img

#resime gürültü ekledik
erezyon=cv2.erode(n_img.astype(np.uint8), kernel, iterations=2)
genisleme=cv2.dilate(erezyon,kernel,iterations=2)
""" YADA"""
opening=cv2.morphologyEx(n_img.astype(np.float32), cv2.MORPH_OPEN, kernel)

#aşamaların görselleştirilmesi
fig=mp.figure()
ax=fig.add_subplot(3,1,1)
mp.imshow(n_img,cmap="gray")
mp.axis("off")
ax.set_title("gürültülü görüntü")
ax=fig.add_subplot(3,1,2)
mp.imshow(erezyon,cmap="gray")
mp.axis("off")
ax.set_title("erezyonlanmış görüntü")
ax=fig.add_subplot(3,1,3)
mp.imshow(genisleme,cmap="gray")
mp.axis("off")
ax.set_title("açılmış görüntü ")

#sonuç görseli
fig=mp.figure()
ax=fig.add_subplot(1,3,1)
mp.imshow(n_img,cmap="gray")
mp.axis("off")
ax.set_title("gürültülü görüntü")
ax=fig.add_subplot(1,3,2)
mp.imshow(genisleme,cmap="gray")
mp.axis("off")
ax.set_title("açılmış görüntü ")
ax=fig.add_subplot(1,3,3)
mp.imshow(opening,cmap="gray")
mp.axis("off")
ax.set_title("açılmış görüntü\n(opening)")

#kapatma : açılmanın tam tersine sırayla genişleme + erezyon uygulanmasıdır resimdeki siyah noktaları kapatmak için kullanılabilir .
black_noise=np.random.randint(0,2,size=img.shape[:2])
black_noise=black_noise*-255

 
bn_img=black_noise+img
bn_img[bn_img<=-245]=0
#resime gürültü ekledik
genisleme2=cv2.dilate(bn_img.astype(np.float32),kernel,iterations=2)
erezyon2=cv2.erode(genisleme2, kernel, iterations=2)

"""YADA"""
closing=cv2.morphologyEx(bn_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
#aşamaların görselleştirilmesi
fig=mp.figure()
ax=fig.add_subplot(3,1,1)
mp.imshow(bn_img,cmap="gray")
mp.axis("off")
ax.set_title("gürültülü görüntü")
ax=fig.add_subplot(3,1,2)
mp.imshow(genisleme2,cmap="gray")
mp.axis("off")
ax.set_title("genisletilmis görüntü")
ax=fig.add_subplot(3,1,3)
mp.imshow(erezyon2,cmap="gray")
mp.axis("off")
ax.set_title("kapanmış görüntü ")

fig=mp.figure()
ax=fig.add_subplot(1,3,1)
mp.imshow(bn_img,cmap="gray")
mp.axis("off")
ax.set_title("gürültülü görüntü")
ax=fig.add_subplot(1,3,2)
mp.imshow(erezyon2,cmap="gray")
mp.axis("off")
ax.set_title("kapanmış görüntü ")
ax=fig.add_subplot(1,3,3)
mp.imshow(closing,cmap="gray")
mp.axis("off")
ax.set_title("kapanmış görüntü\n(closing)")

#morfolojik gradyan : teknik olarak genişletilmiş görüntü ile erezyona uğramış görüntünün farkının alınmasıdır.
gradient=cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
mp.figure()
mp.imshow(gradient,cmap="gray")
mp.axis("off")
mp.title("Morfolojik Gradyan resim")