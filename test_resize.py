import numpy as np
import cv2

img = cv2.imread('D:\\1.JPG')
cv2.imshow('image0',img)
img1 = cv2.resize(img,(160,120)) #前面为宽  col x，后面为高 row y
cv2.imshow('image1',img1)
cv2.waitKey(0)
