#from matplotlib import mpl, pyplot
import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np

# make values from -5 to 5, for this example
zvals = np.random.rand(100,100)*10-5

# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['blue','black','red'])
bounds=[-6,-2,2,6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = plt.imshow(zvals,interpolation='nearest',
                    cmap = cmap,norm=norm)

# make a color bar
plt.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[-5,0,5])

plt.show()



fig = plt.figure(2)

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['blue','black','red'],
                                           256)

img2 = plt.imshow(zvals,interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')

plt.colorbar(img2,cmap=cmap2)
plt.show()

fig.savefig("image2.png")
