"""
Attempted analytical modelling of the optimal timing
"""

import numpy as np
import matplotlib.pyplot as plt
import optimisation_simulation.theory.values as values

def get_tmin(t_delay, d, width):
    output =  ( t_delay ** 2 + ( d / values.c ) ** 2 ) / (
        2 * ( t_delay + width )
    )
    return output


def get_tmax(t_delay, d, R, width):
    output =  ( t_delay ** 2 + ( ( d + R ) / values.c ) ** 2 ) / (
        2 * ( t_delay - width )
    )
    return output

# initial conditions
d = values.off_axial_dist * 1e-3 * 40
R = values.gamma_radius * 1e-3 * 40
N = values.xray_number_density * 2/3 * np.pi * 1e-3 ** 3
width = 20e-12

# arguments
t_delay = np.linspace(0, 500, 100) * 1e-12 # time at which which the beam hits the xray bath (ps)
t_delay_scaled = t_delay * 1e12

# values
tmin = get_tmin(t_delay, d, width)
tmax = get_tmax(t_delay, d, R, width)

plt.title('Analytical time delay hit count relation')
plt.xlabel('time delay (ps)')
plt.ylabel('hit count (arbritary units)')
# plt.ylim(-1e13, 3e13)
# plt.plot(t_delay_scaled, 1/tmin, label='tmin')
# plt.plot(t_delay_scaled, 1/tmax, label='tmax2')
test = (1/tmin-1/tmax)
plt.plot(t_delay_scaled, test, label='est npairs')
plt.legend()
plt.show()