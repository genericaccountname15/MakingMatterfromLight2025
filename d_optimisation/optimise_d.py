"""
Check simulation over different off axial distances and see for differences in maximum yield

Timothy Chew
22/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('d_optimisation\\optimise_d.csv', delimiter=',', skiprows=1)

d = data[:,0]
npos = data[:,1]
npos_err = data[:,2]

fig, ax = plt.subplots() #pylint: disable=unused-variable
ax.set_title('Positron count vs Pulse displacement')
ax.set_xlabel('d (mm)')
ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

ax.plot(
    d, npos,
    '-o',
    label = 'Positron yield',
    color = 'blue'
)

ax.fill_between(
    x = d,
    y1 = npos - npos_err,
    y2 = npos + npos_err,
    label = 'Uncertainty',
    color = 'blue',
    alpha = 0.3
)

ax.axvline(x = 1,
    ymin = 0, ymax = 1,
    label = 'd used in 2018', color = 'orange')

ax.set_axisbelow(True)
ax.grid()
ax.legend()

plt.show()