"""
Simulation moving to 3 dimensions!

Timothy Chew
17/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def gen_Xray_seed(mean, n_angular_samples = 400, n_samples = 10):
    """Generates distribution of X ray pulse in 3D

    Args:
        mean (float): mean of distribution, radial position
        n_angular_samples (int, optional): Number of angles to sample. Defaults to 400
        n_samples (int, optional): Number of samples per angle. Defaults to 10.

    Returns:
        list: list of coordinates for distribtution points
    """
    #variance of distribution is related to FWHM of 40ps, convertion to variance is here: https://mathworld.wolfram.com/GaussianFunction.html
    FWHM = 40e-12 * 3e8 # multiply by c to get spacial width
    std_dev = ( FWHM / 2.3548 * 1e3 ) # value is in mm
    variance = std_dev ** 2

    coords = []

    #rotate distribution 180 degrees
    angles = np.linspace(0,np.pi, n_angular_samples)
    angles_azim = np.append(
        np.linspace(0, np.pi/2, int(n_angular_samples/2)),
        np.linspace(np.pi/2, int(n_angular_samples/2))
    )
    for phi in angles_azim:
        flat_coords = []
        for theta in angles:
            ndist = np.random.normal(mean,variance,n_samples)

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                flat_coords.append([rotated_coords[0], rotated_coords[1], theta])
        
        rot_matrix = np.array(
            [ [1, 0, 0], 
             [0, np.cos(phi), -np.sin(phi)],
             [0, np.sin(phi), np.cos(phi)] ] 
             )
        for k in flat_coords:
            rotated_coords = np.matmul(rot_matrix, [k[0], k[1], 0])
            coords.append([rotated_coords[0], rotated_coords[1], rotated_coords[2], k[2], phi])
    
    return np.array(coords)



def move_Xrays(coords, t):
    #iterate through Xray coordinates
    t *= 1e-9 #mm and picosecond, unit conversion
    moved_coords = []
    for r in coords:
        #calculate distance to move
        dx = 3e8 * t * np.cos(r[2]) * np.sin(r[3])
        dy = 3e8 * t * np.sin(r[2]) * np.sin(r[3])
        dz = 3e8 * t * np.cos(r[3])

        moved_coords.append([r[0] + dx, r[1] + dy, r[2] + dz])
    
    return np.array(moved_coords)

def plotter(xray_coords, bath_vis = True):
    """Plots simulation

    Args:
        xray_coords (numpy.ndarray): of xray distribution points
        gamma (_type_): 
        x0 (float): initial x-coordinate
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.subplots_adjust(bottom=0.25) #making space for sliders
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Gamma and X-ray simulation')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')
    ax.set_zlabel('z/mm')

    xray_bath, = ax.plot(xray_coords[:, 0], xray_coords[:, 1], xray_coords[:,2], 'o', label = 'X-ray bath', alpha=0.1)

    if bath_vis:
        ax.set_xlim(-500, 500)
        ax.set_ylim(0,500)  
        ax.set_zlim(-500, 500)
        ax.set_aspect('equal')
        ax.legend(loc = 'upper right')
    else:
        ax.set_xlim(-4, 4)
        ax.set_ylim(0,2)
        ax.set_zlim(-4, 4)
        ax.set_aspect('equal')
        ax.legend(loc = 'upper right')


    #time step slider
    time_slider = Slider(
        ax = fig.add_axes([0.25, 0.1, 0.65, 0.03]),
        label = 'time(ps)',
        valmin = 0,
        valmax = 1000,
        valinit = 0
    )

    def update(val):
        #note: time is in pico seconds and distance in mm

        coords = move_Xrays(xray_coords, time_slider.val)
        
        xray_bath.set_xdata(coords[:, 0])
        xray_bath.set_ydata(coords[:, 1])
        xray_bath.set_3d_properties(coords[:, 2])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    plt.show()

if __name__ == '__main__':
    FWHM = 40e-12 * 3e8 * 1e3 #FWHM of X-ray bath in mm
    beam_length = 45e-15 * 3e8 * 1e3
    beam_height = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
    d = 1
    x0 = -10e-12 * 3e8 * 1e3 

    xray_coords = gen_Xray_seed(-FWHM, n_angular_samples=40, n_samples=2) #start on small edge of distribution
    plotter(xray_coords, bath_vis=True)
    # print(xray_coords)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xray_coords[:,0], xray_coords[:,1], xray_coords[:,2], marker = 'o')
    # plt.show()