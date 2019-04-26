import cv2

img = cv2.imread('D:\\1.JPG')
cv2.imshow('image0',img)
img1 = cv2.resize(img,(224,224))
img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
cv2.imshow('image1',img2)
cv2.waitKey(0)