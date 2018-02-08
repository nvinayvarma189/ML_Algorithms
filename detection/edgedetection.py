import cv2
import numpy

img=cv2.imread(r'C:\images\large.png')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray=numpy.float32(gray)

dst=cv2.cornerHarris(gray,2,3,0.04)

dst=cv2.dilate(dst,None)

#threshold for a particaular value
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
