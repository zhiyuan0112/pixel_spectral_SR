import imp

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

data = loadmat('/home/liangzhiyuan/Code/spectralSR/data/A.mat')
data = data['A']
print(data.shape)
x = range(100)
for i in range(16):
    plt.plot(x, data[i], linewidth=0.7)
plt.savefig('data/T/a.png')
