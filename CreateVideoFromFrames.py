import cv2
import glob

#Path to images, include /^.jpg at the end to access all images in given directory
PATH_TO_IMAGES = 'Crowd_PETS09/S1/L1/Time_13-59/View_001/201 frames/*.jpg'

#Create video based on input frames
img_array = []
for filename in glob.glob(PATH_TO_IMAGES):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

#Output the video
out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()