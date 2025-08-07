"""
Plots the hit map to characterise
the gamma profile that comes from the collimator.
Attempt to replicate the plot from 2018

Timothy Chew
7/8/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('g4bl_stuff/g4beamlinesfiles/gamma_profile_Det.txt', skiprows=1)

x = np.arctan2(data[:,1],85+50+45+100+25+400+160) * 1000
y = np.arctan2(data[:,0],85+50+45+100+25+400+160) * 1000

plt.title('hit map of gamma beam source at the interaction point')
plt.xlabel('y/mrad')
plt.ylabel('x/mrad')
plt.hist2d(x, y, bins=500, cmap='gnuplot')
plt.colorbar(label='Counts (AU)')
plt.xlim(-6,4)
plt.ylim(-5,5)
plt.show()