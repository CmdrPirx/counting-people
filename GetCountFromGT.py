import numpy as np

#THIS CODE CREATES COUNT FOR EACH FRAME FROM CSV FILE WITH GROUND TRUTH VALUES FOR OBJECT DETECTION
PATH_TO_GROUND_TRUTH_CSV = 'Crowd_PETS09/S1/L1/Time_13-59/View_001/S1L1-1359-Counts.csv'
groundTruth = np.genfromtxt(PATH_TO_GROUND_TRUTH_CSV, delimiter=',')
groundTruth[0][0]=0
realCount = np.zeros((201,1))

for row in groundTruth:
    if row[0] > 200:
        break
    print(row[0])
    realCount[int(row[0])] = realCount[int(row[0])] + 1

np.savetxt('GTpeopleCount.csv', realCount, delimiter=',') 
    
    
