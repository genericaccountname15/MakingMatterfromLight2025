"""
plot_bw_cs.py

Plots the Breit-Wheeler cross section along with the 
differential cross section

Timothy Chew
1/8/25
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from theory.cross_section import c_bw, dc_bw    #pylint: disable=import-error

def plot_formulae():
    """
    Plots the formulae for the breit wheeler cross section
    and differential cross section
    """
    # setup figure ########################################################
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(elev=15, azim=60, roll=0)
    ax1.set_title('Differential Breit-Wheeler scattering cross-section')
    ax1.set_xlabel('$\\theta$(rad)')
    ax1.set_ylabel('$\\sqrt{s}$(MeV)')
    ax1.set_zlabel('$d\\sigma_{\\gamma\\gamma}/d\\Omega (b)$')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Total Breit-Wheeler scattering cross-section')
    ax2.set_xlabel('$\\sqrt{s}$(MeV)')
    ax2.set_ylabel('$\\sigma_{\\gamma\\gamma}(b)$')
    ax2.set_xlim(1e-1, 1e3)

    # generate values ######################################################
    # differential cross section #############
    theta = np.linspace(0, np.pi, 100)
    root_s_differential = np.linspace(1, 5, 100)
    theta, root_s_differential = np.meshgrid(theta, root_s_differential) #for array dimensions

    diff_cross_sec = dc_bw(root_s_differential, theta)
    
    # total cross section ###################
    root_s = np.logspace(0, 3, 1000)
    cross_sec = c_bw(root_s)

    # plot values #########################################################
    ax1.plot_surface(theta, root_s_differential, diff_cross_sec, cmap = cm.gnuplot) #pylint: disable = no-member
    ax2.loglog(root_s, cross_sec, color='red')

    plt.show()