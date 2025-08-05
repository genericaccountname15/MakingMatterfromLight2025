"""
Rough file made to plot all csv files on the same figure

Timothy Chew
5/8/25
"""

import numpy as np
import matplotlib.pyplot as plt

plt.title('Positron count vs Kapton tape angle')
plt.xlabel('Angle (degrees)')
plt.ylabel('Maximum number of positrons/pC incident on CsI')

data1 = np.loadtxt('optimise_angle_1mm.csv', delimiter=',', skiprows=1)
data2 = np.loadtxt('optimise_angle_2mm.csv', delimiter=',', skiprows=1)
data3 = np.loadtxt('optimise_angle_3mm.csv', delimiter=',', skiprows=1)
data4 = np.loadtxt('optimise_angle4mm.csv', delimiter=',', skiprows=1)
data5 = np.loadtxt('optimise_angle5mm.csv', delimiter=',', skiprows=1)

plt.plot(
    data1[:,0], data1[:,1],
    '-o',
    label = 'line source 1mm',
    color = '#292f56'
)

plt.fill_between(
    x = data1[:,0],
    y1 = data1[:,1] - data1[:,2],
    y2 = data1[:,1] + data1[:,2],
    color = '#292f56',
    alpha = 0.3
)

plt.plot(
    data2[:,0], data2[:,1],
    '-o',
    label = 'line source 2mm',
    color = '#006290'
)

plt.fill_between(
    x = data2[:,0],
    y1 = data2[:,1] - data2[:,2],
    y2 = data2[:,1] + data2[:,2],
    color = '#006290',
    alpha = 0.3
)

plt.plot(
    data3[:,0], data3[:,1],
    '-o',
    label = 'line source 3mm',
    color = '#0097a3'
)

plt.fill_between(
    x = data3[:,0],
    y1 = data3[:,1] - data3[:,2],
    y2 = data3[:,1] + data3[:,2],
    color = '#0097a3',
    alpha = 0.3
)

plt.plot(
    data4[:,0] * 180 / np.pi, data4[:,1],
    '-o',
    label = 'line source 4mm',
    color = '#00cf97'
)

plt.fill_between(
    x = data4[:,0] * 180 / np.pi,
    y1 = data4[:,1] - data4[:,2],
    y2 = data4[:,1] + data4[:,2],
    color = '#00cf97',
    alpha = 0.3
)

plt.plot(
    data5[:,0] * 180 / np.pi, data5[:,1],
    '-o',
    label = 'line source 5mm',
    color = '#acfa70'
)

plt.fill_between(
    x = data5[:,0] * 180 / np.pi,
    y1 = data5[:,1] - data5[:,2],
    y2 = data5[:,1] + data5[:,2],
    color = '#acfa70',
    alpha = 0.3
)

plt.axvline(x = 40,
        ymin = 0, ymax = 1,
        label = 'Angle used in 2018', color = 'orange')

plt.legend()
plt.grid()

plt.show()

