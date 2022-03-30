# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:48:36 2022

@author: Mehmet

"""


#özellik eşleştirme : karmaşık sahnede belli özelliğe sahip hedeflerin tespiti için kullanılan yöntemdir.
#yavaştır

import cv2 
import matplotlib.pyplot as mp


 
img=cv2.imread("cikolata_sepeti.jpg",0)
#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
nesne=cv2.imread("oreo.jpg",0)
#nesne=cv2.cvtColor(nesne, cv2.COLOR_BGR2RGB)

mp.figure()
mp.imshow(img,cmap="gray")
mp.axis("off")

mp.figure()
mp.imshow(nesne,cmap="gray")
mp.axis("off")

#tanımlayıcılar: orb tanımlayıcı görüntü ve aranan nesne arasındaki özellikleri tespit etmek için

orb=cv2.ORB_create()

#anahtar nokta tespiti:
kp1,des1=orb.detectAndCompute(nesne,None)
kp2,des2=orb.detectAndCompute(img,None)

#bf matcher :brootforce matcher
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
#noktaları eşleştir
matches =bf.match(des1,des2)  

#mesafeye göre sıralayalım

matches=sorted(matches,key=lambda x: x.distance)

mp.figure()
img_match =cv2.drawMatches(nesne, kp1, img, kp2, matches[:20], None,flags=2)

mp.imshow(img_match)
mp.axis("off")
mp.title("ORB Eşleşme")
#orb tanımlayıcı çok daha net train verisine ihtiyaç duyduğundan (arama yapılan resimde train edilen resmin renleri şekilleri durdukları açılar) bu durumdaki verilerde başarısız olurlar

#sift tanımlayıcı: opencv ye dışardan eklendiğinden kurulum gerekir :pip install opencv-contrib-python --user

sift=cv2.SIFT_create()

#bf
bf=cv2.BFMatcher()

#anahtar nokta tespiti sift ile
kp3,des3=sift.detectAndCompute(nesne,None)
kp4,des4=sift.detectAndCompute(img,None)

eslesme=bf.knnMatch(des3, des4, k=2)

g_eslesme=[]

for match1,match2 in eslesme:
    if match1.distance <0.75*match2.distance:
        g_eslesme.append([match1])
        

mp.figure()
sift_match =cv2.drawMatchesKnn(nesne, kp3, img, kp4, g_eslesme, None,flags=2)


mp.imshow(sift_match)
mp.axis("off")
mp.title("Sift Eşleşme")


#FLANN BAsed Matcher

#FLANN Parametreleri

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params=dict(checks=50)

fln=cv2.FlannBasedMatcher(index_params,search_params)

fln_match=fln.knnMatch(des3, des4, k=2)

mask=[[0,0] for i in range(len(fln_match))]


for i ,(m,n) in enumerate(fln_match):
    if m.distance < 0.7*n.distance:
        mask[i]=[1,0]
        

draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=mask,flags=0)
img3=cv2.drawMatchesKnn(nesne, kp3, img, kp4, fln_match, None,**draw_params)


mp.figure()
mp.imshow(img3)
mp.axis("off")
mp.title("FLANN Eşleşme")



