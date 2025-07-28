"""
Check simulation over different input angles and see for differences in maximum yield

Timothy Chew
22/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('length_optimisation\\length_optimise.csv', delimiter=',', skiprows=1)

length = data[:,0]
npos = data[:,1]
npos_err = data[:,2]

fig, ax = plt.subplots() #pylint: disable=unused-variable
ax.set_title('Positron count vs Length of xray source')
ax.set_xlabel('Length (mm)')
ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

ax.plot(
    length, npos,
    '-o',
    label = 'Positron yield',
    color = 'blue'
)

ax.fill_between(
    x = length,
    y1 = npos - npos_err,
    y2 = npos + npos_err,
    label = 'Uncertainty',
    color = 'blue',
    alpha = 0.3
)

ax.axvline(x = 0.8,
    ymin = 0, ymax = 1,
    label = 'Length used in 2018', color = 'orange')

# ax.set_ylim(2e-5, 4e-5)

ax.set_axisbelow(True)
ax.grid()
ax.legend()

plt.show()