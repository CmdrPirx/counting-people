import numpy as np
import matplotlib.pyplot as plt

#Generate figures for report

#S1L11efnet = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-57/View_001/efnetOUTPUT/peopleCount.csv', delimiter=',')
#S1L11myefnet = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-57/View_001/my_efnetOUTPUT/peopleCount.csv', delimiter=',')
#S1L11density = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-57/View_001/PixelCountOUTPUT/peopleCount.csv', delimiter=',')
#S1L11real = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-57/View_001/S1L1-1357-GTpeopleCount.csv', delimiter=',')

#S1L11efnet = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-59/View_001/efnetOUTPUT/peopleCount.csv', delimiter=',')
#S1L11myefnet = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-59/View_001/my_efnetOUTPUT/peopleCount.csv', delimiter=',')
#S1L11density = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-59/View_001/PixelCountOUTPUT/peopleCount.csv', delimiter=',')
#S1L11real = np.loadtxt('Crowd_PETS09/S1/L1/Time_13-59/View_001/GTpeopleCount.csv', delimiter=',')

S1L11efnet = np.loadtxt('Crowd_PETS09/S2/L1/Time_12-34/View_001/efnetOUTPUT/peopleCount.csv', delimiter=',')
S1L11myefnet = np.loadtxt('Crowd_PETS09/S2/L1/Time_12-34/View_001/my_efnetOUTPUT/peopleCount.csv', delimiter=',')
S1L11density = np.loadtxt('Crowd_PETS09/S2/L1/Time_12-34/View_001/PixelCountOUTPUT/peopleCount.csv', delimiter=',')
S1L11real = np.loadtxt('Crowd_PETS09/S2/L1/Time_12-34/View_001/GTpeopleCount.csv', delimiter=',')

fig = plt.figure()
plt.plot(S1L11efnet)
plt.plot(S1L11myefnet)
plt.plot(S1L11density)
plt.plot(S1L11real)
plt.xlabel('frame')
plt.ylabel('number of people counted')
plt.legend(('EfficientDet', 'FT-EfficientDet', 'Density estimation', 'Real count'))
fig.savefig('S2L1.png')




