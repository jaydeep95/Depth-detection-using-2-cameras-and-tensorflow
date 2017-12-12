# How to run the program
## Set cam_property function (only last two are neccessory )
### cam_property[ distance between camera(optional), focal length (optional), height (required),width(required)]
## for IP Camera set ulr and run the program
## For USB camera use opencv (capture image , read image and pass it to newfunction)


import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import timeit
import urllib.request as urllib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

# ## Object detection imports
# Here are the imports from the object detection module.


from utils import label_map_util

from utils import visualization_utils as vis_util
#************jd*******model given below is for 90 default objects***

# What model to download.
MODEL_NAME = 'E:/TensorFl/models-master/object_detection/ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
'''
#*********jd********** model given below is for Traffic light detection
# What model to download.
MODEL_NAME = 'traffic_light_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1'''

# running model directory and decoding using tar_file
tar_file = tarfile.open("ssd_mobilenet_v1_coco_11_06_2017.tar.gz")# 

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

#  This function is core function to determine the class of object it will return the class, scale (how much % for this class) and co-ordinates of image
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
  # *************** This function detects the color of signal whether it is red or green and shows an image after masking and print output of color ********
def traffic_color_detection(image):
  # inside the np.range this is HSV values of particular color
  red_area = 0
  green_area = 0
  min_area = 5 # min threshold area of any color
  lower_red = np.array([100,100,100])# lower and upper range for red color( a little modification needs to be done)
  upper_red = np.array([360,255,255])
  lower_green = np.array([30,100,100])# lower and upper range for cyan color(may require modification)
  upper_green = np.array([100,255,255])
  cv2.imshow("original",image)
  hsvl = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  maskr = cv2.inRange(hsvl,lower_red,upper_red)
  _,contours,_ =cv2.findContours(maskr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  for i in contours:
    m = cv2.moments(i)
    red_area = red_area + m['m00']
  cv2.imshow("red signal",maskr)
  maskg = cv2.inRange(hsvl,lower_green,upper_green)
  _,contours,_ =cv2.findContours(maskg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  for i in contours:
    m = cv2.moments(i)
    green_area = green_area + m['m00']
  cv2.imshow("green signal",maskg)
  print (green_area,red_area)
  if green_area >= min_area or red_area >= min_area:
    if red_area > green_area:
      print ("red signal")
    elif red_area == green_area:
      print ("can't define signal")
    else:
      print ("green signal")

      
cam_property = [72 ,0.252,480,854] # [distance between camera, focal length, height, width]
url = 'http://192.168.1.41/snap.jpg'
with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while(1):
          #_,image = cap.read()
          imgResp=urllib.urlopen(url)
          imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
          image=cv2.imdecode(imgNp,-1)
          val1 = newfunction(image)# class,percentage, Ymax,Xmax,Ymin,Xmin
          cv2.imshow('frame_left',image )
          for v in val1:
            if v[0] == 'traffic light':
              # find the area and cut and form new image
              print(v)
              color_image = image2[int(v[2]*cam_property[2]):int(v[4]*cam_property[2]),int(v[3]*cam_property[3]):int(v[5]*cam_property[3])]
              traffic_color_detection(color_image)
              #cv2.imshow("traffic light",color_image)
            elif v[0] == 0:
              break
          if cv2.waitKey(2) and 0xFF == 27:
            break
cv2.destroyAllWindows()
