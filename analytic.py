"""
Analytic solution for the gamma xray collision

Timothy Chew
24/7/25
"""

import numpy as np
import matplotlib.pyplot as plt
import Final_simulation.values as values


def func(t, n, d, beam_radius):
    #unit conversion
    t_delay = t *1e-12
    d *= 1e-3
    beam_radius *= 1e-3
    N = n * 2/3 * np.pi * 1e-3 ** 3
    t0 = 4e-11
    t0 = t_delay + (d + beam_radius ) ** 2 / ( values.c ** 2 * t_delay)
    t0 = 2e-11


    hits = N / (2 * np.pi * (values.c * t_delay ** 2) ) * (
        np.arcsin( (d + beam_radius) / ( values.c * ( t_delay + t0 ) )
        - np.arcsin( d / ( values.c * t_delay ) ) )
    )
    return hits

tmin =  values.off_axial_dist / values.c * 1e-3 * 1e12
t = np.linspace(tmin, 200, 1000) #time delay (ps)
Npairs = func( t, values.xray_number_density, values.off_axial_dist, values.gamma_radius )

plt.ylim(-1e23, 3e24)
plt.plot(t, Npairs, 100)
plt.show()