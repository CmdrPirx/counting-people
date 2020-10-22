import numpy as np
import matplotlib.pyplot as plt

#Load results from NormalizationMapGeneration1
grid = np.loadtxt('view1density2cubic.csv', delimiter=',')
#plt.imshow(grid)
#plt.show()

#Fill missing values with column means
col_mean = np.nanmean(grid, axis=0)
inds = np.where(np.isnan(grid))
grid[inds] = np.take(col_mean, inds[1])


#Show plot
fig = plt.figure()
plt.imshow(grid)
plt.colorbar()
plt.show()

#Save figure
#fig.savefig('pnm.png')


#Save perspective normalization map
np.savetxt('view1density2cubicfilled.csv', grid, delimiter=',')
#print(col_mean)