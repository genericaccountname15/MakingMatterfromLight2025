"""
Simulated beam crossing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

#create x-ray distribtion
x = []
y = []
for theta in range(0,180):
    ndist = np.random.normal(100,10,10)

    #filtering
    ndist = ndist[ndist >= 0]


    theta = theta * np.pi/180
    rot_matrix = np.array(
        [ [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)] ])

    for k in ndist:
        rotated_coords = np.matmul(rot_matrix, [k, 0])
        x.append(rotated_coords[0])
        y.append(rotated_coords[1])


#gamma beam
x_init = -50
d = 3
gamma = Rectangle([x_init, d], width=100, height=1, facecolor=(0, 0, 1, 0.5), edgecolor=(1, 0, 0), linewidth = 2, label='gamma beam')

beam_bounds = [gamma.get_y(), gamma.get_y() + gamma.get_height()]

#check amount of points in beam
y = np.array(y)
y_lower = y[ beam_bounds[0] <= y]
points_in_beam = len(y_lower[ y_lower <= beam_bounds[1]])
print(points_in_beam)


#plotting
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25) #making space for sliders

line, = ax.plot(x, y, 'o', label = 'X-ray bath')

#time step
time_slider = Slider(
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03]),
    label = 'time',
    valmin = 0,
    valmax = 10,
    valinit = 0
)

def update(val):
    x = []
    y = []
    for theta in range(0,180):
        ndist = np.random.normal(time_slider.val*10,time_slider.val,10)

        #filtering
        ndist = ndist[ndist >= 0]


        theta = theta * np.pi/180
        rot_matrix = np.array(
            [ [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)] ])

        for k in ndist:
            rotated_coords = np.matmul(rot_matrix, [k, 0])
            x.append(rotated_coords[0])
            y.append(rotated_coords[1])

    line.set_xdata(x)
    line.set_ydata(y)

    fig.canvas.draw_idle()

    y = np.array(y)
    y_lower = y[ beam_bounds[0] <= y]
    points_in_beam = len(y_lower[ y_lower <= beam_bounds[1]])
    print(points_in_beam)

time_slider.on_changed(update)



ax.add_patch(gamma)
ax.set_xlim(-20,20)
ax.set_ylim(-1,20)  
ax.set_aspect('equal')
ax.legend()


plt.show()