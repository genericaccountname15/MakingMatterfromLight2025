"""
Another attempted simulation, this time using gaussian temporal profile
which does not expand in time

Timothy Chew
16/7/25
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle


def gen_Xray_dist(mean, n_angular_samples = 400, n_samples = 10):
    """Generates distribution of X ray pulse in 2D

    Args:
        mean (float): mean of distribution, radial position
        variance (float): spread of distribution
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
    for theta in angles:
        ndist = np.random.normal(mean,variance,n_samples)

        #filtering out negative values
        ndist = ndist[ndist >= 0]

        #rotation matrix
        rot_matrix = np.array(
            [ [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)] ])

        #append coords
        for k in ndist:
            rotated_coords = np.matmul(rot_matrix, [k, 0])
            coords.append([rotated_coords[0], rotated_coords[1]])
    
    return np.array(coords)

def create_gamma_beam(x, dimensions, d):
    """Creates gamma beam axes object and returns bounds of rectangular beam in 2D

    Args:
        x (float): beam coordinate (bottom-left corner)
        dimensions (list): [width, height] dimensions of beam
        d (float): off axis distance

    Returns:
        matplotlib patches object: Rectangular shape of beam
        list: Boundaries of beam [xmin, xmax, ymin, ymax]
    """
    #gamma beam
    gamma = Rectangle([x, d], width=dimensions[0], height=dimensions[1], facecolor=(0, 0, 1, 0.5), edgecolor=(1, 0, 0), linewidth = 2, label='gamma beam')

    beam_bounds = [
        gamma.get_x(), gamma.get_x() + gamma.get_width(),
        gamma.get_y(), gamma.get_y() + gamma.get_height()
        ]

    return gamma, beam_bounds

def get_overlap(coords, beam_bounds):
    """Count number of points of x-ray pulse overlap with gamma pulse

    Args:
        coords (list): list of coords for x-ray distribution
        beam_bounds (list): gamma beam boundary

    Returns:
        integer: number of overlapping points
    """
    n = 0 #points in beam counter

    #nested iterations
    for r in coords:
        if beam_bounds[0] <= r[0] <= beam_bounds[1]:
            if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                n += 1
    
    return n

def get_overlap_coords(coords, beam_bounds):
    """Get coordinates of overlapping points of x-ray and gamma pulses

    Args:
        coords (list): Coordinates of x-ray pulse points
        beam_bounds (list): Boundaries of gamma pulse [xmin, xmax, ymin, ymax]

    Returns:
        numpy.ndarray: Array of overlapping coordinates
    """
    overlap_coords = []
    for r in coords:
        if beam_bounds[0] <= r[0] <= beam_bounds[1]:
            if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                overlap_coords.append([r[0], r[1]])
    
    return np.array(overlap_coords)

def plotter(xray_coords, gamma, x0, beam_bounds, bath_vis = False):
    """Plots simulation

    Args:
        xray_coords (numpy.ndarray): coordinates of xray distribution points
        gamma (matplotlib.patches object): patch object for gamma beam
        x0 (float): initial x-coordinate
    """
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25) #making space for sliders
    ax.set_title('Gamma and X-ray simulation')
    ax.set_xlabel('x/mm')
    ax.set_ylabel('y/mm')

    xray_bath, = ax.plot(xray_coords[:, 0], xray_coords[:, 1], 'o', label = 'X-ray bath')
    
    overlap_coords = get_overlap_coords(xray_coords, beam_bounds)
    if len(overlap_coords) != 0:
        overlap, = ax.plot(overlap_coords[:, 0], overlap_coords[:, 1], 'o', label="overlap")
    else:
        overlap, = ax.plot(0,-10,'o', label='overlap')

    ax.add_patch(gamma)

    if bath_vis:
        ax.set_xlim(-500, 500)
        ax.set_ylim(0,500)  
        ax.set_aspect('equal')
        ax.legend(loc = 'upper right')
    else:
        ax.set_xlim(-4, 4)
        ax.set_ylim(0,2)  
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
        r = time_slider.val * 1e-12 * 3e8 * 1e3#radial distance travelled by shell, r = ct
        x = time_slider.val * 1e-12 * 3e8 * 1e3 + x0 #x_coordinate of gamma beam

        coords = gen_Xray_dist(r)
        
        xray_bath.set_xdata(coords[:, 0])
        xray_bath.set_ydata(coords[:, 1])

        gamma.set_x(x)
        
        bounds = [
            gamma.get_x(), gamma.get_x() + gamma.get_width(),
            gamma.get_y(), gamma.get_y() + gamma.get_height()
            ]
        overlap_coords = get_overlap_coords(coords, bounds)
        if len(overlap_coords) != 0:
            overlap.set_xdata(overlap_coords[:, 0])
            overlap.set_ydata(overlap_coords[:, 1])
            print('overlap')
        else:
            overlap.set_xdata([0])
            overlap.set_ydata([-10])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)


    plt.show()

if __name__ == '__main__':
    FWHM = 40e-12 * 3e8 * 1e3 #FWHM of X-ray bath in mm

    xray_coords = gen_Xray_dist(-FWHM) #start on small edge of distribution
    beam_length = 45e-15 * 3e8 * 1e3
    beam_height = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
    d = 1
    x0 = -10e-12 * 3e8 * 1e3 
    gamma, bounds = create_gamma_beam(x0, [beam_length,beam_height], 1)
    plotter(xray_coords, gamma, x0, bounds, bath_vis=False)