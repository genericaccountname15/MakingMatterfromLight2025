"""
Check simulation over different input angles and see for differences in maximum yield

Timothy Chew
22/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('angle_yield.csv', delimiter=',', skiprows=1)

angle = data[:,0]
npos = data[:,1]
npos_err = data[:,2]

fig, ax = plt.subplots() #pylint: disable=unused-variable
ax.set_title('Positron count vs kapton tape angle')
ax.set_xlabel('Angle (Degrees)')
ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

ax.plot(
    angle, npos,
    '-o',
    label = 'Positron yield',
    color = 'blue'
)

ax.fill_between(
    x = angle,
    y1 = npos - npos_err,
    y2 = npos + npos_err,
    label = 'Uncertainty',
    color = 'blue',
    alpha = 0.3
)

ax.axvline(x = 40,
    ymin = 0, ymax = 1,
    label = 'Angle used in 2018', color = 'orange')

ax.set_axisbelow(True)
ax.grid()
ax.legend()

plt.show()