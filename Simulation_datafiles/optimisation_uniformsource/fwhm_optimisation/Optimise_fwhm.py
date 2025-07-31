"""
Check simulation over different FWHM and see for differences in maximum yield

Timothy Chew
28/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('fwhm_optimisation\\FWHM_optimise.csv', delimiter=',', skiprows=1)

fwhm = data[:,0]
npos = data[:,1]
npos_err = data[:,2]

fig, ax = plt.subplots() #pylint: disable=unused-variable
ax.set_title('Positron count vs FWHM of xray source')
ax.set_xlabel('FWHM (ps)')
ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

ax.plot(
    fwhm, npos,
    '-o',
    label = 'Positron yield',
    color = 'blue'
)

ax.fill_between(
    x = fwhm,
    y1 = npos - npos_err,
    y2 = npos + npos_err,
    label = 'Uncertainty',
    color = 'blue',
    alpha = 0.3
)

ax.axvline(x = 40,
    ymin = 0, ymax = 1,
    label = 'fwhm used in 2018', color = 'orange')

# ax.set_ylim(2e-5, 4e-5)

ax.set_axisbelow(True)
ax.grid()
ax.legend()

plt.show()