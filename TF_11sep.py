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

#************sorting according to class *************** right now the resolution is 640x480
def sorting(val1,val2,cam_property):
  lst_of_mat_row = [] # making list of matching objects
  row1 = 0
  row2 = 0
  number_of_match = 0
  flag = 0
  for i in val1:# iterating through first set of values camera 1
    row1 = row1 + 1# increamenting by one to keep track of row numbers
    if i[0] != 0:# if first element is not class
      for m in val2: # iterating through second set of values of camera 2
        row2 = row2 + 1
        if m[0] != 0:
          if i[0] == m[0]:# if both classes match
            number_of_match =  number_of_match + 1# if more than one class are same
            lst_of_mat_row.append(row2 - 1)# appending row number in list of matching class
            flag = 1
          else:
            break
    if number_of_match >= 1:
      if number_of_match > 1:
        temp_row2 = lst_of_mat_row[0]# pending (apply sorting in it)
          # sort by area or histogram
      else:
        temp_row2 = lst_of_mat_row[0]# temp row that will be used while final countdown
            
      x1 = ((val1[row1-1][5] - val1[row1-1][3])/2 + val1[row1-1][3]) *cam_property[2]# column 5 and 3 have Xmin and Xmax and cam_property [2] have FOV
      x2 = ((val2[temp_row2][5] - val2[temp_row2][3])/2 + val2[temp_row2][3]) *cam_property[2]
              # ******* depth from camera*********
      depth = (cam_property[0] * cam_property[2])/(2*cam_property[1]*(x1 - x2))# depth of object from cameras
      font = cv2.FONT_HERSHEY_SIMPLEX
      display  = " D = " + str(int(abs(depth)))
      print ("%s is %s cm away"%(val1[row1 - 1],display))
      x_point = int(abs(cam_property[2]*val1[row1 - 1][5]))
      y_point = int(abs(cam_property[3]*val1[row1-1][4]))
      cv2.putText(frame_left,display,(x_point,y_point), font, 1,(0,255,0),1,cv2.LINE_AA)#
      number_of_match = 0
    else:
      break
  if flag == 0:
    return "No common object were found"
  else:
    return " Object/objects were found"
              
                
                
                
            
              
left = 1
right = 0
cap1 = cv2.VideoCapture('newofc_left.avi')
cap2 = cv2.VideoCapture('newofc_right.avi')
cam_property = [72 ,0.252] # camera property [ distance b/w camera , focal length]
flag = 1
with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while True:
          t1 = cv2.getTickCount()
          ret, frame_left = cap1.read()
          ret, frame_right = cap2.read()
          if cv2.waitKey(2) & 0xFF == ord('q'):
            flag = 0
          if flag != 0:
            cv2.imshow('frame_left',frame_left )# use cv2.resize(frame_left, (800,600))
            cv2.imshow('frame_right',frame_right )
          else:
            val1 = newfunction(frame_left)
            val2 = newfunction(frame_right)
            height,width,_ = frame_left.shape
            cam_property.append(width)# adding width of frame in camera property
            cam_property.append(height)
            sorting(val1,val2,cam_property)
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
