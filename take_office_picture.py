import cv2
import numpy as np

cap = cv2.VideoCapture(0)
obj = 1
while (1):
  _,frame = cap.read()
  cv2.imshow('frame',frame)
  if cv2.waitKey(2) & 0xFF == ord('q'):
    cv2.imwrite(str(obj)+'_frame.png',frame)
    obj  = obj + 1
  if cv2.waitKey(2) & 0xFF == 27:
    break

cv2.destroyAllWindows()
cap.release()
