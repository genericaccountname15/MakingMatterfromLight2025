"""
Cross section calculation using the 2018 paper

Timothy Chew
16/07/25
"""

import numpy as np
# constants
e = 1.60e-19
c = 299792458
m = 9.11e-31    # electron mass
re = 2.8179403205e-15 #electron radius (classical)

def dc_BW(root_s, theta):
    """Breit Wheeler differential cross section

    Args:
        s (float): CM energy
        theta (float): Collision and scattering angle
    """
    root_s = root_s * e * 1e6 #Convert out of MeV to SI
    s = root_s**2
    beta =  np.sqrt( 1 - 4 * m * m * c ** 4 / s )

    dc_BW = beta * re * re * m * m * c ** 4 / s  * ( ( 1 + 2 * beta * np.sin(theta) ** 2 
                                      - beta ** 4 -beta ** 4 * np.sin(theta) ** 4 ) /
                                      ( 1- beta * beta * np.cos(theta) ** 2 ) ** 2)
    
    dc_BW *= 1e28 #conversion to barns

    return dc_BW

def c_BW(root_s):
    """Breit Wheeler total cross section

    Args:
        s (float): CM energy
    """
    root_s = root_s * e * 1e6 #Convert out of MeV
    s = root_s**2
    beta =  np.sqrt( 1 - 4 * m * m * c ** 4 / s )

    c_BW = np.pi * re * re * ( 1 - beta * beta ) / 2 * (
        ( 3 - beta ** 4 ) * np.log( ( 1 + beta ) / ( 1 - beta ) )
        - 2 * beta * ( 2 - beta * beta )
    )

    c_BW *= 1e28 #conversion to barns

    return c_BW

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.transforms import Affine2D
    #cross_sec = c_BW(root_s ** 2)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # differential cross section
    #generate values
    theta = np.linspace(0, np.pi, 100)
    root_s = np.linspace(1, 5, 100)
    theta, root_s = np.meshgrid(theta, root_s) #for array dimensions

    diff_cross_sec = dc_BW(root_s, theta)

    #add axes
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(elev=15, azim=60, roll=0)
    ax1.set_title('Differential Breit-Wheeler scattering cross-section')
    ax1.set_xlabel('$\\theta$(rad)')
    ax1.set_ylabel('$\\sqrt{s}$(MeV)')
    ax1.set_zlabel('$d\\sigma_{\\gamma\\gamma}/d\\Omega (b)$')
    
    ax1.plot_surface(theta, root_s, diff_cross_sec, cmap = cm.gnuplot)

    # total cross section
    #generate values
    root_s = np.logspace(0, 3, 1000)
    cross_sec = c_BW(root_s)

    #add axes
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Total Breit-Wheeler scattering cross-section')
    ax2.set_xlabel('$\\sqrt{s}$(MeV)')
    ax2.set_ylabel('$\\sigma_{\\gamma\\gamma}(b)$')
    ax2.loglog(root_s, cross_sec, color='red')
    ax2.set_xlim(1e-1, 1e3)

    plt.show()

    