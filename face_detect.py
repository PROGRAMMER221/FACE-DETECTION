import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while(True):

      ret , frame = cap.read()
      
      gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
      
      faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05,minNeighbors = 5)
      
      for x,y,w,h in faces:
            color = (255,0,0)
            stroke = 2
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)

      cv2.imshow('frame',frame)
      
      if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
