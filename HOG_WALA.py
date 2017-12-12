0#**********importing packages*********
import cv2
import numpy as np
import timeit
e1 = cv2.getTickCount()
# *********inilizing parameters (making list for center co-ordinates)*******
lx = []
ly = []
l1x = []
l1y = []
height = []#height and width of image on left side
width = []
height1 = []#height and width of image on right side
width1 = []
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())

#***********Pedestrian detection function*************
def pedestrian(img,flag):

# **** this function detects the pedestrian in image and return it's centre,height and width
    found,_=hog.detectMultiScale(img, winStride=(4,4), padding=(32,32), scale=1.05)
    for x,y,w,h in found:
   	 pad_w, pad_h = int(0.15*w), int(0.05*h)
   	 cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 0),1)
   	 if (flag == 1):
   		 lx.append(int(x+w/2))
   		 ly.append(int(y+h/2))
   		 height.append(int (h))
   		 width.append(int (w))
   	 else:
   		 l1x.append(int(x+w/2))
   		 l1y.append(int(y+h/2))
   		 height1.append(int (h))
   		 width1.append(int (w))

def display_depth(frame_left,frame_right):
    flag = 1 # if pedestrian found than value will be changed
    hist_list = []# list of histogram
    ratio = [[0 for x in range(len(l1x))] for y in range(len(lx))]#column x row(but access is reverse)

# ****************making list*****************
    for i in range(len(lx)):#iterating through length of centers image in left image
   	 for j in range(len(l1x)):#iterating through length of centers of image in right image
   		 ratio[i][j]=int(height[i]*width[i]/(height1[j]*width1[j])*100)# saving area ratio of pedestrina detected in both the images
    print (ratio)
# *************sorting out list*************
    for i in range(len(ratio)):
   	 number = 0
   	 tem_j = []
   	 for j in range(len(ratio[i])):
   		 #print ("ratio = %f"%ratio[i][j])
   		 if ratio [i][j] <130  and ratio[i][j] >70:# *** sorting after having area ratio
   			 #print (lx[i] ,l1x[j])
   			 if (lx[i] != l1x[j]):
   				 number = number + 1
   				 tem_j.append(j)#number of common area founded
   	 if (number>1):
   				 #******find histogram and comparing ratio*****
   				 # ** cut the image and save it in grayscale***
   		 number = 0
   		 t_left = frame_left[(ly[i]-height[i]/2):(ly[i]+height[i]/2),(lx[i]-width[i]/2):(lx[i]+width[i]/2)]
   		 tg_left = cv2.cvtColor(t_left,cv2.COLOR_BGR2GRAY)
   		 hist_left = cv2.calcHist([tg_left],[0],None,[256],[0,256])# histogram value
   		 for v in tem_j:
   			 t_right = frame_right[(l1y[v]-height1[v]/2):(l1y[v]+height1[v]/2),(l1x[v]-width1[v]/2):(l1x[v]+width1[v]/2)]
   			 tg_right = cv2.cvtColor(t_right,cv2.COLOR_BGR2GRAY)
   			 rdx = str(v)
   			 # cv2.calcHist returns comparison value
   			 hist_right = cv2.calcHist([tg_right],[0],None,[256],[0,256])
   			 # *** making list of histogram comparison values
   			 hist_list.append(cv2.compareHist(hist_left,hist_right,cv2.HISTCMP_CORREL))#adding value to new list of histogram
   		 print ("hist comp value is %f"%hist_list)
   		 maxv = 0#setting to default
   		 i_value = 0#index value
   		 for index,val in enumerate(hist_list):
   			 if val > maxv:
   				 i_value = index
   				 maxv = val
   				 # ** comparing histogram values
   		 distance = str(int(abs((72*400)/(2*0.252*(lx[i]-l1x[tem_j[i_value]])))))
   		 display = "d = "+distance
   		 #print (distance)
   		 flag = 0
   		 font = cv2.FONT_HERSHEY_SIMPLEX
   		 cv2.putText(frame_left,display,(lx[i]-int(width[i]/2),ly[i]-int(height[i]/2)), font, 1,(0,255,0),1,cv2.LINE_AA)
   				 # *** changing value for rest of ratio(setting it to zero)***
   		 for v in range(len(ratio)):#flushing other column values    						 #if boxes are same
   			 ratio[v][i_value] = 0
   	 elif number == 1:
   		 number = 0
   		 #change 400 to L value if resolution is LxH
   		 distance = str(int(abs((72*400)/(2*0.252*(lx[i]-l1x[tem_j[0]])))))
   		 display = "d = "+distance
   		 #print (distance)
   		 flag = 0
   		 font = cv2.FONT_HERSHEY_SIMPLEX
   		 cv2.putText(frame_left,display,(lx[i]-int(width[i]/2),ly[i]-int(height[i]/2)+20), font, 1,(0,255,0),1,cv2.LINE_AA)
   		 for v in range(len(ratio)):#flushing other column values    						 #if boxes are same
   			 ratio[v][tem_j[0]] = 0
   	 hist_list[:] = []
   	 tem_j[:] =[]
    if (flag):
   	 print ("didn't find any pedestrian")
#********reseting values
    for i in range(len(ratio)):
   	 for j in range(len(ratio[i])):
   		 ratio[i][j] = 0
    lx[:] = ly[:] = l1x[:] = l1y[:] = height[:] = height1[:] =width[:] = width1[:] =[]   		 

def tracking_color(frame_left,frame_right):
    
    #********* colorspace conversion
    hsvl = cv2.cvtColor(frame_left,cv2.COLOR_BGR2HSV)
    hsvr = cv2.cvtColor(frame_right,cv2.COLOR_BGR2HSV)
    
    #for blue colour or you can pass RGB values to program
    lower_range = np.array([160,100,100])
    upper_range = np.array([179,255,255])
    #lower_range = hsv_calc(R,B,G)
    #upper_range = hsv_calc(R,B,G)
    # ***** extracting only perticular part of image (on basis of color)
    maskl = cv2.inRange(hsvl,lower_range,upper_range)
    maskr = cv2.inRange(hsvr,lower_range,upper_range)
    _,contoursl,_ =cv2.findContours(maskl,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _,contoursr,_ =cv2.findContours(maskr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contoursl)>0 and len(contoursr)>0:#if there are contours
   	 print("you have %d and %d contours"%(len(contoursl),len(contoursr)))
   	 areal = 0;
   	 for i in range(0,len(contoursl)):
   		 c1 = contoursl[i]
   		 m1 = cv2.moments(c1)# moments to calc area of segmented section(c1 in this case)
   		 if areal < m1['m00']:
   			 cntl = c1
   			 areal = m1['m00']
   	 if areal >0 :#if it found legitimate contour
   		 m1 = cv2.moments(cntl)
   		 xl = int (m1['m10']/m1['m00'])#centroid x co-ordinates
   		 y1 = int (m1['m01']/m1['m00'])# centroid point Y of image
   		 cv2.drawContours(frame_left,[cntl],0,(255,0,0),3)
   		 print ("here is your co-ordinates on left side are %d,%d "%(xl,y1))
   		 arear = 0
   		 for i in range (0,len(contoursr)):
   			 c2 = contoursr[i]
   			 cr1 = cv2.moments(c2)
   			 area = cr1['m00']
   			 if arear<area:
   				 '''y21 = int (cr1['m01']/cr1['m00'])
   				 if ((y21-y1)<yr1):
   					 yr1 = y21 -y1
   					 cntr = c2
   					 #print (cntr)'''
   				 arear = area
   				 cntr = c2
   					 
   		 if arear >0:
   			 cr = cv2.moments(cntr)
   			 #print ("area on right side is %d " %arear)
   			 xr = int (cr['m10']/cr['m00'])
   			 yr = int (cr['m01']/cr['m00'])
   			 print ("here is your co-ordinates on right side are %d,%d "%(xr,yr))   			 
   			 if (xl-xr)!=0:
   				 depth = (72*640)/(2*0.252*(xl-xr))
   				 print("distance of object from camera is %d "%depth)
   			 cv2.drawContours(frame_right,[cntr],0,(0,0,255),2)
   		 else:
   			 print ("right camera didn't recognise object")
   	 else:
   		 print("left camera didn't recognise any object")

    else:
   	 print(" no contour found")
   	 print("you have %d and %d contours"%(len(contoursl),len(contoursr)))
def hsv_calc(r,g,b):
    r1 = r/255
    g1 = g/255
    b1 = b/255
    cmax = max(r1,g1,b1)
    cmin = min(r1,g1,b1)
    
    dif = cmax-cmin

    if dif==0:
   	 hue = 0
    elif cmax == r1:
   	 hue = 60*(((g1-b1)/dif)%6)
    elif cmax == g1:
   	 hue = 60*((b1-g1)/dif + 2)
    elif cmax == b1:
   	 hue = 60*((r1-g1)/dif +4)
    if cmax == 0:
   	 sat = 0
    else:
   	 sat = dif/cmax
    val = cmax
    arr = np.array([int(hue),int(sat*255),(val*255)])
    return arr

	#  	*******************main program*******************
cam_left = cv2.VideoCapture('cutnw_left.avi')
cam_right = cv2.VideoCapture('cutnw_right.avi')
R=123
G = 231
B = 224
R1 = 232    
G1 = 235
B1 = 24
while(1):
    e1 = cv2.getTickCount()
    #********capturing frames and resizing it*********
    _,frame_left = cam_left.read()
    _,frame_right = cam_right.read()
    #frame_left = cv2.imread('jd_left.jpg',-1)
    #frame_right = cv2.imread('jd_right.jpg',-1)
    frame_left = cv2.resize(frame_left,(400,300))# resizing image
    frame_right = cv2.resize(frame_right,(400,300))
    pedestrian(frame_left,1)
    pedestrian(frame_right,2)

    #*****passing value to check and display depth****
    display_depth(frame_left,frame_right)
    tracking_color(frame_left,frame_right)#** this show depth on terminal(need to edit)
    cv2.imshow('image',frame_left)
    cv2.imshow('image_right',frame_right)
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print ("time taken by program is %f"%time)
    if cv2.waitKey(4) & 0xFF == 27:
   	 break
cv2.destroyAllWindows()

   		 
   			 
   				 
   		 





