import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure()

gs = gridspec.GridSpec(1, 2, width_ratios=[10,.5])

ax = np.empty([1,2],dtype=np.object)
im = np.empty([1,1],dtype=np.object)

ax[0,0] = plt.subplot(gs[0,0])
im[0,0] = ax[0,0].imshow(np.random.rand(10,10))

ax[0,1] = plt.subplot(gs[0,1])
plt.colorbar(im[0,0],cax=ax[0,1])

plt.show()

