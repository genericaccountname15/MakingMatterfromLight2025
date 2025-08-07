"""
Counts the number of Bethe-Heitler positrons
detected at the virtual detector

Timothy Chew
7/8/25
"""
import numpy as np

data = np.loadtxt('g4bl_stuff/g4beamlinesfiles/noise_measure_Det.txt', skiprows=1)
mask = (data[:,7] == -11)
print(len(data[mask]))
