"""
Rough calculation and estimation of number of positron pairs
detected at the CsI detector

Timothy Chew
16/07/25
"""

import numpy as np
from counter_hits import simulate_hits
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
hit_ratio = sum(simulate_hits(x0)) / 4000

# estimate number of positrons
N_p = nx * Ny * AMS_transmission * cs * hit_ratio

#get uncertainty
uncertainty = np.sqrt(
    (0.5/1.4) ** 2 + (1/7) ** 2 + (0.05/0.25) ** 2 
) * N_p

print(f'Estimated number of positrons {N_p} +/- {uncertainty}') 