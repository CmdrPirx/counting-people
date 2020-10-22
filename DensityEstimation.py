import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import time


PATHTOVIDEO = 'Crowd_PETS09/S1/L1/Time_13-59/View_001/S1L1-1359-201frames.avi'
VIDEOWIDTH = 768
VIDEOHEIGHT = 576
#Load video
video = cv2.VideoCapture(PATHTOVIDEO)
video2 = cv2.VideoCapture(PATHTOVIDEO)

#load pixel counts from training run of this program and manual count of people corresponding to counted pixels
pixelCount = np.loadtxt('C:/Users/ekorz/Desktop/Thesis/Crowd_PETS09/S0/Regular_Flow/Time_14-03/View_001/xpixelCount.csv', delimiter=',')
actualCount = np.loadtxt('C:/Users/ekorz/Desktop/Thesis/Crowd_PETS09/S0/Regular_Flow/Time_14-03/View_001/zActualCount.csv', delimiter=',')
#train linear regresison
reg = LinearRegression().fit(pixelCount.reshape(-1, 1), actualCount)
#Initialize background substraction
size = (VIDEOWIDTH,VIDEOHEIGHT)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=35, detectShadows = True)
#prepare output
out = cv2.VideoWriter('counted.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
#load perspective normalization map values for view 1
grid = np.loadtxt('view1density2cubicfilled.csv', delimiter=',')

#Create array for storing output of the program
pixelCount = []
peopleCount = []
timeTaken = []

#Since the given video is not a live feed, go over video once to create background model
while True:
    ret, frame=video2.read()
    #break if the video is over
    if frame is None:
        break
    fgmask = fgbg.apply(frame)


#Actual people count happens here
while True:
    ret, frame=video.read()
    #break if the video is over
    if frame is None:
        break
    #Get size of video frame
    start_time2 = time.time()
    height2, width2, layer = frame.shape
    #Apply background substraction
    fgmask = fgbg.apply(frame)
    #Get rid of shadows
    ret, fgmask2 = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY) 
    #Detect edges
    edges = cv2.Canny(fgmask2,100,200)
    edges = cv2.dilate(edges, None, iterations=2)
    edgescopy = edges.copy()
    height, width = edges.shape[:2]
    #Fill in the edges
    mask = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(edgescopy, mask, (0,0), 255)
    inv = cv2.bitwise_not(edgescopy)
    motion = edges | inv

    #Display motion pixels after multiplying them by perspective normalization maps
    text2 = str(np.round(np.sum((motion/255)*grid)))
    #Use linear regression to predict amount of people
    prediction = reg.predict(np.sum((motion/255)*grid).reshape(-1, 1))
    #Round it to nearest number
    count = np.round(prediction)
    #Count time elapsed
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    text = str(count)
    #Add counts to outputs
    peopleCount.append(count)
    timeTaken.append(elapsed_time2)
    
    #Convert image to BGR for video export
    color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    #Put text into the modified frame
    cv2.putText(color, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(color, text2, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    #Put text into the frame with original image
    cv2.putText(frame, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, text2, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    #Write out statement
    out.write(frame)
    
    #Add pixel count to ouput
    pixelCount.append(np.round(np.sum((motion/255)*grid)))
    
    #Show output as it's running, comment out to obtain accurate time counts
    cv2.imshow('frame',motion )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release statements
out.release()
video.release()
cv2.destroyAllWindows()

#Save output
np.savetxt('pixelCount.csv', pixelCount, delimiter=',')
np.savetxt('peopleCount.csv', peopleCount, delimiter=',')
np.savetxt('timeTaken.csv', timeTaken, delimiter=',')


    