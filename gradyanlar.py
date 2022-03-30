# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:11:08 2022

@author: Mehmet
"""

import cv2
import matplotlib.pyplot as mp

img=cv2.imread("sudoku.png")
mp.figure()
mp.imshow(img,cmap="gray")
mp.title("Orjinal Resim")
mp.axis("off")

# x gradyanı : gradyanlar genelde kenar bulmada kullanılır 

sobelX=cv2.Sobel(img, ddepth=cv2.CV_16S, dx=1, dy=0,ksize=5)
mp.figure()
mp.imshow(sobelX,cmap="gray")
mp.title("Sobel X")
mp.axis("off")

# y gradyanı :
    
sobelY=cv2.Sobel(img, ddepth=cv2.CV_16S, dx=0, dy=1,ksize=5)
mp.figure()
mp.imshow(sobelY,cmap="gray")
mp.title("Sobel Y")
mp.axis("off")

#laplace gradyanı : hem x hemde y yönünde tespit için

laplacian=cv2.Laplacian(img, ddepth=cv2.CV_16S)
mp.figure()
mp.imshow(laplacian,cmap="gray")
mp.title("Laplace Gradyanı")
mp.axis("off")