import cv2
import numpy as np
video = cv2.VideoCapture('signal270.mp4')
var = 1
while(1):
    _,frame = video.read()    
    if cv2.waitKey(3) & 0xFF == ord('s'):
      cv2.imwrite('framefirst__'+str(var)+'.jpg',frame);
      print ("image written %d"%var)
      var = var + 1
    cv2.imshow('image',frame)
    if cv2.waitKey(4) & 0xFF == 27:
   	 break
cv2.destroyAllWindows()
video.release()


