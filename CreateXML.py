import xml.etree.ElementTree as gfg  
import glob
import numpy as np

###THIS CODE CREATES XML FILE COMPATIBILE FOR CONVERSION TO TENSORFLOW .RECORD FILES FOR EACH IMAGE FROM A CSV FILE OF GROUND TRUTH DATA
PATH_TO_IMAGE_FOLDER = 'Crowd_PETS09/S2/L1/Time_12-34/View_001/'
count = 0

PATH_TO_GROUND_TRUTH_CSV = 'Crowd_PETS09/S2/L1/Time_12-34/View_001/S2L1-1234-Count.csv'
PART_OF_PATH_TO_BE_REPLACED = 'Crowd_PETS09/S2/L1/Time_12-34/View_001\\'
groundTruth = np.genfromtxt(PATH_TO_GROUND_TRUTH_CSV, delimiter=',')
groundTruth[0][0]=0
convertedGroundTruth = np.empty((0,5), int)

for row in groundTruth:
    topLeftY = np.round(row[5] - row[2]/2)
    topLeftX = np.round(row[4] - row[3]/2)
    bottomRightY = np.round(row[5] + row[2]/2)
    bottomRightX = np.round(row[4] + row[3]/2)
    convertedGroundTruth = np.append(convertedGroundTruth, [[row[0], topLeftY, topLeftX, bottomRightY, bottomRightX]], axis=0)

print(convertedGroundTruth)
for filename in glob.glob(PATH_TO_IMAGE_FOLDER + '/*.jpg'):
    #print(filename.replace('Crowd_PETS09/S1/L1/Time_13-59/View_001\\', ''))
    #print(count)
    root = gfg.Element("annotation")

    m1 = gfg.Element("folder")
    m1.text = "images"
    root.append(m1)
    
    m2 = gfg.Element("filename")
    m2.text = filename.replace(PART_OF_PATH_TO_BE_REPLACED, '')
    root.append(m2)
    
    m3 = gfg.Element("path")
    m3.text = "C:/Users/ekorz/Desktop/Thesis/TensorFlow/workspace/training_demo/images/" + filename.replace(PART_OF_PATH_TO_BE_REPLACED, '')
    root.append(m3)
    
    m4 = gfg.Element("source")
    m5 = gfg.Element("database")
    m5.text = "Unknown"
    
    m4.append(m5)
    root.append(m4)
    
    m6 = gfg.Element("size")
    m7 = gfg.Element("width")
    m7.text = "768"
    m8 = gfg.Element("height")
    m8.text = "576"
    m9 = gfg.Element("depth")
    m9.text = "3"
    
    m6.append(m7)
    m6.append(m8)
    m6.append(m9)
    root.append(m6)
    
    m10 = gfg.Element("segmented")
    m10.text = "0"
    root.append(m10)
    
    for row in convertedGroundTruth: #blue
        if(row[0]==count):
            m11 = gfg.Element("object")
            m12 = gfg.Element("name")
            m12.text = "person"
            m13 = gfg.Element("pose")
            m13.text = "Unspecified"
            m14 = gfg.Element("truncated")
            m14.text = "0"
            m15 = gfg.Element("difficult")
            m15.text = "0"
            m11.append(m12)
            m11.append(m13)
            m11.append(m14)
            m11.append(m15)
            
            m16 = gfg.Element("bndbox")
            m17 = gfg.Element("xmin")
            m17.text = str(int(row[2]))
            m18 = gfg.Element("ymin")
            m18.text = str(int(row[1]))
            m19 = gfg.Element("xmax")
            m19.text = str(int(row[4]))
            m20 = gfg.Element("ymax")
            m20.text = str(int(row[3]))
            m16.append(m17)
            m16.append(m18)
            m16.append(m19)
            m16.append(m20)
            m11.append(m16)
            root.append(m11)
            
    tree = gfg.ElementTree(root)
    with open (PATH_TO_IMAGE_FOLDER + filename.replace(PART_OF_PATH_TO_BE_REPLACED, '').replace('.jpg','') + ".xml", "wb") as files : 
        tree.write(files)
    
    count = count + 1
