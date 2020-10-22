import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#Create an array containing pixel positions and pixel weights at extreme points of the view
n = np.array([(0, 168, 0.93),
              (0, 238, 0.93),
              (768, 285, 0.56),
              (768, 492, 0.56),
              (580, 144, 1),
              (768, 144, 1),
              (357, 421, 0.89),
              (768,576, 0.5),
              (0, 576, 0.8)
              ])

#Create an empty matrix for with the size of the used image
grid = np.zeros((576,768))


#Perform cubic interpolation
for i in range(0,768):
    for j in range(0,576):
        print(i , " " , j)
        grid[j][i] = griddata(n[:,0:2], n[:,2], [(i, j)], method='cubic')


#Save obtained pixel 
np.savetxt('view1density2cubic.csv', grid, delimiter=',')
#Show plot
plt.imshow(grid)
plt.show()