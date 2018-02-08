from PIL import Image
import face_recognition
import cv2
import numpy

# Load the jpg file into a numpy array
#image = face_recognition.load_image_file(r'C:\python35\data\image3.jpeg')
cap=cv2.VideoCapture(r'C:\Python35\data\VID-20170713-WA0010.mp4')

# Find all the faces in the image
#face_locations = face_recognition.face_locations(image)

#print("I found {} face(s) in this photograph.".format(len(face_locations)))
while (cap.isOpened()):
   ret,frame=cap.read()
   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   face_locations = face_recognition.face_locations(frame)
   for face_location in face_locations:
      top, right, bottom, left = face_location
      frame=cv2.rectangle(frame,(top,left),(bottom,right),(255,0,0),2)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF==ord('q'):
         break
cap.release()
cv2.destroyAllWindows()
      


      

    # Print the location of each face in this image
    #top, right, bottom, left = face_location
    #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    #face_image = image[top:bottom, left:right]
    #pil_image = Image.fromarray(face_image)
    #pil_image.show()
