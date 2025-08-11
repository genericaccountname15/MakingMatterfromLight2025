"""
Counts the number of Bethe-Heitler positrons
detected at the virtual detector

Timothy Chew
7/8/25
"""
import numpy as np

data = np.loadtxt('Gamma_profile_Det_LWFA_100mil.txt', skiprows=1)
mask = (data[:,7] == -11)
print(len(data[mask]))
