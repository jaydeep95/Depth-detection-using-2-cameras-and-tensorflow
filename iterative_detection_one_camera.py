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
#************jd*******model given below is for 90 default objects***

# What model to download.
MODEL_NAME = 'E:/TensorFl/models-master/object_detection/ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
'''
#*********jd********** model given below is for mac and cheese only
# What model to download.
MODEL_NAME = 'mac_cheese_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
'''
# running model directory and decoding using tar_file
tar_file = tarfile.open("ssd_mobilenet_v1_coco_11_06_2017.tar.gz")

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
def centroid(val1,index1,cam_property):
  x_left = int(abs(((val1[index1][5] - val1[index1][3])/2 + val1[index1][3]) *cam_property[2]))# column 5 and 3 have Xmin and Xmax and cam_property [2] have width
  y_left = int(abs(((val1[index1][4] - val1[index1][2])/2 + val1[index1][2]) *cam_property[3]))
  centroid =[ x_left,y_left]
  return centroid

def extract_object(frame,val1,ids,cam_property):
  new_frame = frame[int(val1[ids][2]*cam_property[3]):int((val1[ids][4])*cam_property[3]),int((val1[ids][3])*cam_property[2]):int((val1[ids][5])*cam_property[2])]
  return new_frame

def interative_detection(frame,val1,cam_property):
  for index,i in enumerate(val1):
    if i[0] != 0:
      frame_new = extract_object(frame,val1,index,cam_property)
      vd = newfunction(frame_new)
pnt_val = ["CLASS","CENTROID(X,Y)"]
var_val = 0
index = 0
cam_property = [72 ,0.252,854,480]
frames_time = 0
array_to_store = [[0 for x in range (3)] for y in range(40)]
cam = cv2.VideoCapture('signal270.mp4')
with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while(True):
          var_val = var_val + 1
          _,frame_left = cam.read()
          t1 = cv2.getTickCount()
          val1 = newfunction(frame_left)
          interative_detection(frame_left,val1,cam_property)
          t2 = cv2.getTickCount()
          for index0,i in enumerate(val1):
            if i[0] != 0:
              #print (i[0])
              for index,n in enumerate(array_to_store):
                if n[0] == 0 :
                  array_to_store[index][0] = i[0]
                  array_to_store[index][1],array_to_store[index][2] = centroid(val1,index0,cam_property)
                  break
            else:
              array_to_store[index+1][:] = "****"
              break          
          #print (" time taken to process data is %f"%((t2-t1)/cv2.getTickFrequency()))
          frames_time = frames_time + (t2-t1)/cv2.getTickFrequency()
          if var_val%5 ==0:
            for i in range(2):
              print (pnt_val[i],"\t\t\t", end = '')
            print ("\n")
            for i in array_to_store:
              if i[0] != 0:
                for j in i:
                  print (j,"\t\t",end='')
                print ("\n")
            array_to_store = [[0 for x in range (3)] for y in range(40)]
            print ("total time taken for 5 frame %f"%frames_time)
            frames_time = 0
          cv2.imshow('frame_left',frame_left )
          if cv2.waitKey(2) & 0xFF == ord('s'):
            while (1):
              if cv2.waitKey(2) & 0xFF ==ord('r'):
                break
          if cv2.waitKey(25) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
cam.release()
			  
