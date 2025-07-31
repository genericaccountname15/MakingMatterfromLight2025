"""
Optimises the parameter f
f is the distance of the tungsten block shielding to the centre
of the gamma profile
Basically how 'non-semicircle' it becomes

Timothy Chew
22/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('f_optimisation\\f_optimise.csv', delimiter=',', skiprows=1)

f = data[:,0]
npos = data[:,1] * 2
npos_err = data[:,2] * 2

fig, ax = plt.subplots() #pylint: disable=unused-variable
ax.set_title('Positron count vs shielding')
ax.set_xlabel('f /mm')
ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

ax.plot(
    f, npos,
    '-o',
    label = 'Positron yield',
    color = 'blue'
)

ax.fill_between(
    x = f,
    y1 = npos - npos_err,
    y2 = npos + npos_err,
    label = 'Uncertainty',
    color = 'blue',
    alpha = 0.3
)

ax.axvline(x = 0,
    ymin = 0, ymax = 1,
    label = 'f used in 2018', color = 'orange')

ax.set_axisbelow(True)
ax.grid()
ax.legend()

plt.show()