# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:09:28 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#bluring iştemi detay azaltır ve gürültüleri önler

img=cv2.imread("newyork_city.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

mp.figure()
mp.imshow(img)
mp.title("Orjinal Resim")
mp.axis("off")

#ortalama bulanıklaştırma
dst=cv2.blur(img, ksize=(3,3))
mp.figure()
mp.imshow(dst)
mp.title("ortalama bulanııklaştrılmış Resim")
mp.axis("off")


#Gaussian blur 

gb=cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
mp.figure()
mp.imshow(gb)
mp.title("Gaus Yöntemi İle Bulanıklaştırılmış Resim")
mp.axis("off")


#median blur : genelde salty (tencikli) gürültüden kurtulmak için kullanılır.
#median blur resmi mutlaka 32 bitlik float olarak kabul eder  onun dışında hata verebilir.

mb=cv2.medianBlur(img, ksize=3)
mp.figure()
mp.imshow(mb)
mp.title("Median Yöntemi İle Bulanıklaştırılmış Resim")
mp.axis("off")

#resmin üstüne gürültü ekleme

def gaussianNoise (image):
    row,col,ch=image.shape
    mean=0
    var=0.05
    sigma=var**0.5
    
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss=gauss.reshape(row,col,ch)
    noisy=gauss+image
    
    return noisy

#resmi 0 ile bir aralığına normalize ediyoruz 
img2=cv2.imread("newyork_city.jpg")
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)/255 #1 ile 255 aralığında olduğu için değerler 255 e bölersek tüm değerler 0-1 aralığına normalize etmiş oluruz

noisy_img=gaussianNoise(img2)

mp.figure()
mp.imshow(noisy_img)
mp.title("Orjinal Resim(gürültülü)")
mp.axis("off")

#şimdi yöntemleri bu resim üzerinde deneyelim


#ortalama bulanıklaştırma
dst2=cv2.blur(noisy_img, ksize=(3,3))
mp.figure()
mp.imshow(dst2)
mp.title("ortalama bulanııklaştrılmış Gürültülü Resim")
mp.axis("off")


#Gaussian blur 

gb2=cv2.GaussianBlur(noisy_img, ksize=(3,3), sigmaX=7)
mp.figure()
mp.imshow(gb2)
mp.title("Gaus Yöntemi İle Bulanıklaştırılmış Gürültülü Resim")
mp.axis("off")


#median blur

mb2=cv2.medianBlur(noisy_img.astype(np.float32), ksize=3)
mp.figure()
mp.imshow(mb2)
mp.title("Median Yöntemi İle Bulanıklaştırılmış Gürültülü Resim")
mp.axis("off")


#saltpaper noisy resim elde etme ve bulanıklaştırma ile çözme 

def saltPaperNoise(image):
    row,col,ch=image.shape
    s_vs_p=0.5
    amount=0.004
    noisy=np.copy(image)
    
    #salt(beyaz noktalar)
    num_salt=np.ceil(amount*image.size*s_vs_p)
    coords=[np.random.randint(0,i-1,int(num_salt)) for i in image.shape]
    noisy[coords]=1
    
    #paper(siyah noktalar)
    num_paper=np.ceil(amount*image.size*(1-s_vs_p))
    coords=[np.random.randint(0,i-1,int(num_paper)) for i in image.shape]
    noisy[coords]=0
    
    return noisy


s_p_img=saltPaperNoise(img2)
mp.figure()
mp.imshow(s_p_img)
mp.title("Siyah Ve Beyaz Lekeli Resim")
mp.axis("off")


#median blur

mb3=cv2.medianBlur(s_p_img.astype(np.float32), ksize=3)
mp.figure()
mp.imshow(mb3)
mp.title("Median Yöntemi İle Bulanıklaştırılmış lekeli Resim")
mp.axis("off")
