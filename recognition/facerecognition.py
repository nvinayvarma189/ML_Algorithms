import cv2
import time
import numpy
cap=cv2.VideoCapture(0)
fc=cv2.CascadeClassifier(r'C:\python35\images\haarcascade_upperbody.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
while (cap.isOpened()):
   ret,frame=cap.read()
   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   faces=fc.detectMultiScale(gray,1.2,5)
   #for (x,y,w,h) in faces:
      #frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      #cv2.putText(frame,'Anshu Pandey',(x,y-20), font, 2,(0,0,255),2,cv2.LINE_AA)
   cv2.imshow('frame',frame)
   time.sleep(0.018)
   if cv2.waitKey(10) & 0xFF==ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
