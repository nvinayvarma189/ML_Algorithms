from skimage import io
from sklearn.externals import joblib
import os
import sys
import time
import serial
import numpy
import cv2

global url

url="http://192.168.1.6:8080/shot.jpg"
s=serial.Serial("COM13",9600)
time.sleep(1)

alg=joblib.load('C:\python35\nnmodelforsdc.pkl')
print('Model Loaded')

def drive():
   img=io.imread(url)
   cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   img=cv2.blur(img,(5,5))
   retval,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)
   img=cv2.resize(img,(24,24))
   retval,img=cv2.threshold(img,210,255,cv2.THRESH_BINARY)
   imgarray=numpy.ndarray.flatten(numpy.array(img))
   result=alg.predict(imgarray)[0]
   if result=='forward':
      s.write(b'f')
      time.sleep(1)
   elif result=='left':
      s.write(b'l')
      time.sleep(1)
   elif result=='right':
      s.write(b'r')
      time.sleep(1)
   elif result=='backward':
      s.write(b'b')
      time.sleep(1)
   elif result=='stop':
      s.write(b's')
      time.sleep(1)
   time.sleep(1)
   print(result)
   drive()
print("Start Driving")
drive()
s.close()



   
