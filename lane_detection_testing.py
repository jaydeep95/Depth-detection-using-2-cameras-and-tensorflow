import cv2
import numpy as np
# read image
def lane_detection(in_image):
  #in_image = cv2.imread('road.jpg')
  height,width,_ = in_image.shape
  # convert image to hsv for better results(for yellow)
  gray_image = cv2.cvtColor(in_image,cv2.COLOR_BGR2GRAY)
  mask_white = cv2.inRange(gray_image,150,255)
  hsv_image = cv2.cvtColor(in_image,cv2.COLOR_BGR2HSV)
  # hsv range for yellow
  lower_yellow = np.array([0,100,100], dtype = "uint8")
  upper_yellow = np.array([30,255,255], dtype = "uint8")
  # masking image with respect to yellow color
  mask_yellow = cv2.inRange(hsv_image,lower_yellow,upper_yellow)
  mask_YW = cv2.bitwise_or(mask_yellow,mask_white)

  mask_yw_image = cv2.bitwise_and(gray_image, mask_YW)
  mask_yw_image[0:int(height*0.7),0:width] = np.zeros([int(height*0.7),width],dtype=np.uint8)
  ##cv2.imshow("mask yellow and white image",mask_yw_image)
  # blurring the image
  blurred_image = cv2.GaussianBlur(mask_yw_image,(5,5),0)
  edges_blurred = cv2.Canny(blurred_image,80,255,apertureSize = 3)
  edges_mask_yw = cv2.Canny(mask_yw_image,80,255,apertureSize = 3)
  cv2.imshow("edges blurred",edges_blurred)
  #cv2.imshow("edges mask yw",edges_mask_yw)
  minLineLength = 10
  maxLineGap = 5
  lines = cv2.HoughLinesP(blurred_image,10,np.pi/160,100,minLineLength,maxLineGap)
  for i in lines:
    for x1,y1,x2,y2 in i:
        cv2.line(in_image,(x1,y1),(x2,y2),(0,255,0),2)

  cv2.imshow('houghlines5 jpg',in_image)
  # white and yellow color separation
def main():
  cap = cv2.VideoCapture('signal900.mp4')
  while(1):
    _,frame = cap.read()
    lane_detection(frame)
    if cv2.waitKey(2) and 0xFF == 27:
      break
lane_detection(cv2.imread('road.jpg')
cv2.destroyAllWindows()
