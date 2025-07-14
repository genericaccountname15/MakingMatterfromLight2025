"""
Calculation to find number of BW pairs produced

Timothy Chew
14/07/25
"""

import numpy as np

from scipy.integrate import quad

def get_cs(theta):
    C = np.cosh(theta)
    S = np.sinh(theta)
    cs_parallel = 2 * np.pi * ( e * e / ( m * c * c ) ) ** 2 * (
        -S / (C ** 3) - 1.5 * S / (C ** 5) - 1.5 * theta / (C ** 6)
        + 2 * theta / (C ** 4) + 2 * theta / (C ** 2)
    )
    return cs_parallel



# constants
e = 1.60e-19
c = 3e8
m = 9.11e-31    # electron mass

#values
N_gamma = 3.5e8 # +/- 0.5
n_xray = 1.4e12 / (1e3  ** 3)
L = 2 * c * 40e-12

#cross section
cs = quad(get_cs, 0, np.pi)[0]

print(cs)

npairs = N_gamma * n_xray * L
print(200 / npairs)