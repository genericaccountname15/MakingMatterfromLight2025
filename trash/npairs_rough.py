"""
Rough calculation and estimation of number of positron pairs
detected at the CsI detector

Timothy Chew
16/07/25
"""

import numpy as np
from counter_hits_v4 import simulate_hits
from cross_section import c_BW

# experimental conditions
nx = 1.4e21 #m^-3 photon number density of X-ray beam 1mm from target (+/-0.5)
Ny = 7e6 #number of gamma photons (+/-1)
AMS_transmission = 0.25 #estimated AMS transition (+/- 0.05)

#get cross section
cs = c_BW(1.0247) #get cross sec for 230MeV root s
cs *= 1e-28 #convert from barns to m^2

# simulate hits
x0 = -10e-12 * 3e8 * 1e3 #experimental delay of 10 ps
beam_length = 45e-15 * 3e8 * 1e3
beam_radius = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
d = 1
hit_ratio = simulate_hits(x0, beam_length, beam_radius, d) / 40000

#circular ratio
max_angle = np.arctan(beam_radius/d) 
hit_ratio = hit_ratio * (max_angle / np.pi)

# estimate number of positrons
N_p = nx * Ny * AMS_transmission * cs * hit_ratio

#get uncertainty
uncertainty = np.sqrt(
    (0.5/1.4) ** 2 + (1/7) ** 2 + (0.05/0.25) ** 2 
) * N_p

print(f'Estimated number of positrons {N_p} +/- {uncertainty} postirons/pC') 