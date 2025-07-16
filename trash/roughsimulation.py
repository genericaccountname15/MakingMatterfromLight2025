"""
Rough simulation
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle, Circle

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25) #making space for sliders

#initial values
x_init = -100
d = 1
r_init = 0

#beams
gamma = Rectangle([x_init, d], width=100, height=1, facecolor=(0, 0, 1, 0.5))
xray = Circle([0,0], r_init, facecolor=(1, 0, 0, 0.5))

#time step
time_slider = Slider(
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03]),
    label = 'time',
    valmin = 0,
    valmax = 10,
    valinit = 0
)

def update(val):
    r = time_slider.val
    x = x_init + time_slider.val

    gamma.set_x(x)
    xray.set_radius(r)
    fig.canvas.draw_idle()

time_slider.on_changed(update)


ax.add_patch(gamma)
ax.add_patch(xray)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_aspect('equal')
ax.plot()


plt.show()