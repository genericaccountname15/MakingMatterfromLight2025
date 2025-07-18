"""
Formula for the Breit Wheeler cross-section taken from
BKettle et al. A laser-plasma platform for photon-photon physics. 2018. New J. Phys 23 115006
Section 8

Timothy Chew
18/07/25
"""

import numpy as np
import values as values

def dc_BW(root_s, theta):
    """Breit Wheeler differential cross section
    We get invalid values into the square root sometimes
    Supposed to set the value of d_cs to 0
    But lazy to handle the meshgrid array

    Args:
        s (float): CM energy
        theta (float): Collision and scattering angle
    """
    root_s = root_s * values.e * 1e6 #Convert out of MeV to SI
    s = root_s**2
    beta =  np.sqrt( 1 - 4 * values.m ** 2 * values.c ** 4 / s )

    d_cs = beta * values.re ** 2 * values.m ** 2 * values.c ** 4 / s  * ( ( 1 + 2 * beta * np.sin(theta) ** 2 
                                      - beta ** 4 -beta ** 4 * np.sin(theta) ** 4 ) /
                                      ( 1- beta * beta * np.cos(theta) ** 2 ) ** 2)
    
    d_cs *= 1e28 #conversion to barns

    return d_cs

def c_BW(root_s):
    """Breit Wheeler total cross section

    Args:
        s (float): CM energy
    """
    # unit conversion #################################
    root_s = root_s * values.e * 1e6 #Convert out of MeV
    s_list = root_s**2

    # check if root_s is singular/float #############
    if isinstance(s_list, float):
        s_list = [s_list]

    c_BW_list = []

    for s in s_list:
        if  1 - 4 * values.m ** 2 * values.c ** 4 / s >= 0:
            beta =  np.sqrt( 1 - 4 * values.m ** 2 * values.c ** 4 / s )

            cs = np.pi * values.re ** 2 * ( 1 - beta * beta ) / 2 * (
                ( 3 - beta ** 4 ) * np.log( ( 1 + beta ) / ( 1 - beta ) )
                - 2 * beta * ( 2 - beta * beta )
            )

            cs *= 1e28 #conversion to barns
            c_BW_list.append(cs)
        
        else:
            cs = 0.0 #reaction doesn't happen
            c_BW_list.append(cs)

    return np.array(c_BW_list)


def plot_formulae():
    """
    Plots the formulae for the breit wheeler cross section
    and differential cross section
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
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

    diff_cross_sec = dc_BW(root_s_differential, theta)
    
    # total cross section ###################
    root_s = np.logspace(0, 3, 1000)
    cross_sec = c_BW(root_s)

    # plot values #########################################################
    ax1.plot_surface(theta, root_s_differential, diff_cross_sec, cmap = cm.gnuplot) #pylint: disable = no-member
    ax2.loglog(root_s, cross_sec, color='red')

    plt.show()


if __name__ == "__main__":
    plot_formulae()