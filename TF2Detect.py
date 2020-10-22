#Based on https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html#sphx-glr-auto-examples-plot-object-detection-saved-model-py
#This code applies TF2 object detection on a video, outputs detected people count and time taken for each frame. Also creates video for outputs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import cv2

PATH_TO_SAVED_MODEL = 'trained models/zoo2/efficientdet_d0_coco17_tpu-32/saved_model'
PATH_TO_VIDEO = 'Crowd_PETS09/S1/L1/Time_13-59/View_001/S1L1-1359-201frames.avi'
#Width of the video frame (neeeded for creating output video)
WIDTH = 768
#Height of the video frame
HEIGHT = 576

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

#load label data from tensorflow object detection API
PATH_TO_LABELS = 'TensorFlow/workspace/training_model/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(cv2.imread(path))


#Load video
video = cv2.VideoCapture(PATH_TO_VIDEO)
#Set size
size = (WIDTH,HEIGHT)
#Set up video writer
out = cv2.VideoWriter('odCounted.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
#Create arrays for storing output of the program
peopleCount = []
timeTaken = []

while True:
    ret, frame=video.read()
    #break if the video is over
    if frame is None:
        break
    #load frame as numpy array
    image_np = np.array(frame)
    #Log current time
    start_time2 = time.time()
    #Detect people
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #Count how long it took to process
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print(elapsed_time2)
    #Count how many people were detected
    count = 0
    for x in range(num_detections):
        if ((detections['detection_classes'][x] == 1) and (detections['detection_scores'][x]>=0.5)):
            count = count + 1
    #Draw detection boxes, classes and scores on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)
    #Add image to the video
    out.write(image_np)
    #Add count to the output
    peopleCount.append(count)
    #Add time taken to process a frame to the output
    timeTaken.append(elapsed_time2)
#Save output
np.savetxt('peopleCount.csv', peopleCount, delimiter=',') 
np.savetxt('timeTaken.csv', timeTaken, delimiter=',')
#Release statements
out.release()
video.release()   
    