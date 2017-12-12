import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import timeit

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]
from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#


# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'E:/TensorFl/models-master/object_detection/ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# running model directory and decoding using tar_file
tar_file = tarfile.open("ssd_mobilenet_v1_coco_11_06_2017.tar.gz")
#print (tar_file)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph() #JD tf.Graph defines namespace for tf.Operation object it contains
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:
def newfunction(image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.( returned value from function is class, scale, co-ordinates)
    # co- ordinate are in 0-1 so this thing needed rescaling  by multiplying with resolution
    val = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return val

''' sorting of classes and than subclasses ( if more than one object is found in same class than sorting them by histogram ) ,sorting by distance ,sorting by size '''
def classification(val1,val2,cam_property,frame_left,frame_right):
  lst_of_mat_row = [] # making list of matching objects
  row1 = 0
  row2 = 0
  number_of_match = 0
  flag = 0
  min_hist = 0.04 # min satisfying value for histogram value
  font = cv2.FONT_HERSHEY_SIMPLEX # font for writing depth and etc
  
  for i in val1:# iterating through first set of values camera 1
    row1 = row1 + 1# increamenting by one to keep track of row numbers
    if i[0] != 0:# if first element is not class
      cut_left = frame_left[int(val1[row1-1][3] * cam_property[2]):int(val1[row1-1][5] * cam_property[2]),int(val1[row1-1][2] * cam_property[3]):int(val1[row1-1][4] * cam_property[2])]
      gray_left = cv2.cvtColor(cut_left,cv2.COLOR_BGR2GRAY) # time taken by color image was too much so i converted it in grayScale
      hist_left = cv2.calcHist([gray_left],[0],None,[256],[0,256])# histogram value
      for m in val2: # iterating through second set of values of camera 2
        row2 = row2 + 1
        if m[0] != 0:
          if i[0] == m[0]:# if both classes match
            number_of_match =  number_of_match + 1# if more than one class are same
            lst_of_mat_row.append(row2 - 1)# appending row number in list of matching class
            #print (i[0])
          else:
            break
      row2 = 0
      match_flag = 0
    if number_of_match >= 1:
      test_value = 0 # check node for further reference( to check if histogram matches or not)
      if number_of_match > 1:# agar 1 se jyada object same class ke present he to
          # differenciate by area
          ratio = []
          for i in lst_of_mat_row:
            val = abs(((val1[row1 - 1][5] - val1[row1 - 1][3])*(val1[row1 - 1][4] - val1[row1 - 1][2]))/((val2[i][5] - val2[i][3])*(val2[i][4] - val2[i][2])))# ratio of area between multiple pbject which are in match with perticuler class
            ratio.append(100*val)
          abs_rat_match = []# stores the row number list of absolute matching area of class
          for i in range(len(lst_of_mat_row)):
            if ratio[i] > 85 and ratio[i] < 115:
              abs_rat_match.append(lst_of_mat_row[i])
          Ws = []# temporary list histogram compare
          for i in abs_rat_match:
            cut_right = frame_right[int(val2[i][3] * cam_property[2]):int(val2[i][5] * cam_property[2]),int(val2[i][2] * cam_property[3]):int(val2[i][4] * cam_property[2])]
            gray_right = cv2.cvtColor(cut_right,cv2.COLOR_BGR2GRAY)
            hist_right = cv2.calcHist([gray_right],[0],None,[256],[0,256])# histogram value
            Ws.append(abs(cv2.compareHist(hist_left,hist_right,cv2.HISTCMP_CORREL)))
          for index,i in enumerate(Ws):
            min_hist1 = min_hist
            if i >min_hist1:
              min_hist1 = i
              temp_row2 = abs_rat_match[index]
              match_flag = 1 # all set to display value
          print ("number of object are more than one")
      else:
        temp_row2 = lst_of_mat_row[0]
        val = 100*abs(((val1[row1 - 1][5] - val1[row1 - 1][3])*(val1[row1 - 1][4] - val1[row1 - 1][2]))/((val2[temp_row2][5] - val2[temp_row2][3])*(val2[temp_row2][4] - val2[temp_row2][2])))
        print ("ratio value is %f"%val)
        if val>85 and val <115:
          cut_right = frame_right[int(val2[temp_row2][3] * cam_property[2]):int(val2[temp_row2][5] * cam_property[2]),int(val2[temp_row2][2] * cam_property[3]):int(val2[temp_row2][4] * cam_property[2])]
          gray_right = cv2.cvtColor(cut_right,cv2.COLOR_BGR2GRAY)
          hist_right = cv2.calcHist([gray_right],[0],None,[256],[0,256])# histogram value
          comp_histV = abs(cv2.compareHist(hist_left,hist_right,cv2.HISTCMP_CORREL))
          print ("value of hist matching is %f"%comp_histV)
          if (comp_histV > min_hist):
            match_flag = 1 # all set to display value
            #print ("value of hist matching is %f"%comp_histV)
      if match_flag == 1:
        match_flag = 0
        x1 = int(abs(((val1[row1-1][5] - val1[row1-1][3])/2 + val1[row1-1][3]) *cam_property[2]))# column 5 and 3 have Xmin and Xmax and cam_property [2] have width
        x2 = int(abs(((val2[temp_row2][5] - val2[temp_row2][3])/2 + val2[temp_row2][3]) *cam_property[2]))
        y1 = int(abs(((val1[row1-1][4] - val1[row1-1][2])/2 + val1[row1-1][2]) *cam_property[3]))
        y2 = int(abs(((val2[temp_row2][4] - val2[temp_row2][2])/2 + val2[temp_row2][2]) *cam_property[3]))
                # ******* depth from camera*********
        depth = (cam_property[0] * cam_property[2])/(2*cam_property[1]*(x1 - x2))# depth of object from cameras
        #***********thresholding to depth***********
        if depth <= 1100:
          display  = " D = " + str(int(abs(depth))) + "CM"
          label_d = "obj " + str(row1-1)
          #print ("%s is %s cm away"%(val1[row1 - 1][0],int(abs(depth))))
          cv2.rectangle(frame_left,(x1-15,y1-10),(x1+15,y1+10),(0,0,255),20)
          cv2.rectangle(frame_right,(x2-15,y2-10),(x2+15,y2+10),(0,0,255),20)
          cv2.putText(frame_left,label_d,(x1-20,y1), font, 0.5,(255,255,255),1,cv2.LINE_AA)#
          cv2.putText(frame_right,label_d,(x2-20,y2), font, 0.5,(255,255,255),1,cv2.LINE_AA)#
          x_point = int(abs(cam_property[2]*val1[row1 - 1][5]))
          y_point = int(abs(cam_property[3]*val1[row1-1][4]))
          cv2.putText(frame_left,display,(x1-30,y1+14), font, 0.3,(255,255,255),1,cv2.LINE_AA)#
          
          flag = 1
      number_of_match = 0
      lst_of_mat_row[:] = []
    else:
      break
    row1 = 0
  if flag == 0:
    return "No common object were found"
  else:
    return " Object/objects were found"

#************* compare it later************
#def compare_histogram(frame_left,frame_right,Hval1,Hval2):
  



left = 1
right = 0
cap1 = cv2.VideoCapture(left)#(left)#('cutnw_left.avi')
cap2 = cv2.VideoCapture(right)#(right)#('cutnw_right.avi')
cam_property = [72 ,0.252] # camera property [ distance b/w camera , focal length]
flag = 1
with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while True:
          t1 = cv2.getTickCount()
          ret, frame_left = cap1.read()
          ret, frame_right = cap2.read()
          val1 = newfunction(frame_left)
          val2 = newfunction(frame_right)
          height,width,_ = frame_left.shape
          cam_property.append(width)# adding width of frame in camera property
          cam_property.append(height)
          classification(val1,val2,cam_property,frame_left,frame_right)
          t2 = cv2.getTickCount()
          #print (" time taken to process data is %f"%((t2-t1)/cv2.getTickFrequency()))
          cv2.imshow('frame_left',frame_left )# use cv2.resize(frame_left, (800,600))
          cv2.imshow('frame_right',frame_right )
          if cv2.waitKey(25) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
