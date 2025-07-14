"""
Calculation to find how many pairs produced

Timothy Chew
14/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
e = 1.60e-19
c = 3e8
m = 9.11e-31    # electron mass


def get_cs(theta):
    C = np.cosh(theta)
    S = np.sinh(theta)
    cs_parallel = 2 * np.pi * ( e * e / ( m * c * c ) ) ** 2 * (
        -S / (C ** 3) - 1.5 * S / (C ** 5) - 1.5 * theta / (C ** 6)
        + 2 * theta / (C ** 4) + 2 * theta / (C ** 2)
    )

    cs_perp =  2 * np.pi * ( e * e / ( m * c * c ) ) ** 2 * (
        -S / (C ** 3) - 0.5 * S / (C ** 5) - 0.5 * theta / (C ** 6)
        + 2 * theta / (C ** 4) + 2 * theta / (C ** 2)
    )

    return cs_parallel, cs_perp

theta = np.linspace(0, np.pi)

cs_ll, cs_L = get_cs(theta)

plt.title('cross sections')
plt.xlabel('$\\theta$')
plt.ylabel('$\\bar{\sigma}$')
plt.plot(theta/np.pi, cs_ll, label = 'parallel')
plt.plot(theta/np.pi, cs_L, label = 'perpendicular')
plt.legend()
plt.show()


