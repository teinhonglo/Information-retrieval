#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

x = np.linspace(0,1, num=10000)
x = -x[1:] * np.log(x[1:])
plt.figure(8)
plt.plot(x, label='entropy')
plt.show()