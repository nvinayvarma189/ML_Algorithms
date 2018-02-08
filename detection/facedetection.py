import numpy
import cv2
face_cascade=cv2.CascadeClassifier(r'C:\python35\images\haarcascade_frontalface_alt2.xml')
ed=cv2.CascadeClassifier(r'C:\python35\images\haarcascade_eye.xml')
                         
img=cv2.imread(r'C:\python35\images\dk.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    eyes=ed.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('gray',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
