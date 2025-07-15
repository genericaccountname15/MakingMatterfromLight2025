"""
Cross section calculation using Watt's PhD thesis

Timothy Chew
15/07/25
"""

import numpy as np
# constants
e = 1.60e-19
c = 3e8
m = 9.11e-31    # electron mass
alpha = 1/137

def dc_BW(s, theta):
    """Breit Wheeler differential cross section

    Args:
        s (float): CM energy
        theta (float): Collision and scattering angle
    """
    beta =  np.sqrt( 1 - 1 / s )
    dc_BW = beta * alpha * alpha / ( m * m * s ) * ( ( 1 + 2 * beta * np.sin(theta) ** 2 
                                      - beta ** 4 -beta ** 4 * np.sin(theta) ** 4 ) /
                                      ( 1- beta * beta * np.cos(theta) ** 2 ) ** 2)
    
    dc_BW *= 1e-56

    return dc_BW

def c_BW(s):
    """Breit Wheeler total cross section

    Args:
        s (float): CM energy
    """
    beta = np.sqrt( 1 - 1 / s )
    c_BW = np.pi * alpha * alpha * ( 1 - beta * beta ) / (2 * m * m) * (
        ( 3 - beta * beta ) * np.log( ( 1 + beta ) / ( 1 - beta ) )
        - 2 * beta * ( 2 - beta * beta )
    )

    c_BW *= 1e-56

    return c_BW

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #cross_sec = c_BW(root_s ** 2)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # differential cross section
    #generate values
    theta = np.linspace(0, np.pi, 100)
    root_s = np.linspace(0, 5, 100)
    theta, root_s = np.meshgrid(theta, root_s) #for array dimensions

    diff_cross_sec = dc_BW(root_s ** 2, theta)

    #add axes
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(elev=15, azim=60, roll=0)
    ax1.set_title('Differential photon-photon scattering cross-section')
    ax1.set_xlabel('$\\theta$(rad)')
    ax1.set_ylabel('$\\sqrt{s}$(MeV)')
    ax1.set_zlabel('$d\\sigma_{\\gamma\\gamma}/d\\Omega (b)$')
    
    ax1.plot_surface(theta, root_s, diff_cross_sec, cmap = cm.gnuplot)

    # total cross section
    #generate values
    root_s = np.logspace(0, 3, 100)
    cross_sec = c_BW(root_s ** 2)

    #add axes
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Total photon-photon scattering cross-section')
    ax2.set_xlabel('$\\sqrt{s}$(MeV)')
    ax2.set_ylabel('$\\sigma_{\\gamma\\gamma}(b)$')
    ax2.loglog(root_s, cross_sec, color='red')
    ax2.set_xlim(1e-1, 1e3)

    plt.show()

    