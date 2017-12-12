import cv2
import numpy as np
cam_left = cv2.VideoCapture(2)
cam_right = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('office_18_left.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter('office_18_right.avi',fourcc,20.0,(640,480))
flag = 1
while(1):
    _,frame_left = cam_left.read()
    _,frame_right = cam_right.read()
    
    #*****passing value to check and display depth****
    cv2.imshow('image_left',frame_left)
    cv2.imshow('image_right',frame_right)
    out.write(frame_left)
    out2.write(frame_right)
    if cv2.waitKey(4) & 0xFF == 27:
   	 break
cv2.destroyAllWindows()


