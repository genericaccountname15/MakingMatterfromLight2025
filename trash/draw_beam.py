import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=plt.figaspect(0.5))
fig.subplots_adjust(bottom=0.25) #making space for sliders
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')
ax.set_title('Gamma and X-ray simulation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def draw_cylinder(ax, d, radius, length, resolution=50, color='blue'):
    #curved surface
    x = np.linspace(0, length, resolution)

    theta = np.linspace(0, np.pi, resolution)
    theta_grid, x_grid = np.meshgrid(theta, x)

    z_grid = radius * np.cos(theta_grid)
    y_grid = d + radius * np.sin(theta_grid)

    #plot surface
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.7)

    # Create the end caps
    # Common semi-circle profile (shared by front and back caps)
    cap_theta = np.linspace(0, np.pi, resolution)
    cap_z = radius * np.cos(cap_theta)
    cap_y = d + radius * np.sin(cap_theta)
    cap_yz = list(zip(cap_y, cap_z))
    cap_yz.append((d, 0))  # Close the semi-circle at the base

    # Front cap at x = 0
    front_cap = [(0, y, z) for y, z in cap_yz]
    # Back cap at x = length
    back_cap = [(length, y, z) for y, z in cap_yz]

    # Add to plot
    ax.add_collection3d(Poly3DCollection([front_cap], color=color, alpha=0.7))
    ax.add_collection3d(Poly3DCollection([back_cap], color=color, alpha=0.7))

draw_cylinder(ax, 1, 1, 2)

plt.show()