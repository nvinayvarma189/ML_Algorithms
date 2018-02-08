import cv2
import numpy

###reading an image from computer
##img=cv2.imread(r'C:\images\image2.png',0)
##cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
##
##cv2.imwrite('C:\images\output.jpg',img)
##cv2.imshow('Image',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

#working with realtime images

##cap=cv2.VideoCapture(0)
##while True:
##    ret,frame=cap.read()
##
##    cv2.imshow('frame',frame)
##    if cv2.waitKey(1) & 0xFF==ord('q'):
##        break
##cap.release()
##cv2.destroyAllWindows()                   


img=cv2.imread(r'C:\images\image2.png')
img=cv2.line(img,(10,12),(200,200),(255,0,0),5)
img=cv2.rectangle(img,(50,0),(200,180),(0,0,255),3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print img.shape


















