"""
Counts number of hits on gamma pulse
"""

import numpy as np
import matplotlib.pyplot as plt
import trash.simulationv2_units as sim

#initial conditions
beam_width = 45e-15 * 3e8 * 1e3
beam_height = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
d = 1


def simulate_hits(x0):
    gamma, bounds = sim.create_gamma_beam(x0, [beam_width, beam_height], d)

    n = []
    # time iteration
    time = np.linspace(0,10,300)
    for t in time:
        #note: time is in pico seconds and distance in mm
        r = t * 1e-12 * 3e8 * 1e3#radial distance travelled by shell, r = ct
        var = t * 1e-12 / 40e-12 #variance of shell, time/FWHM
        x = t * 1e-12 * 3e8 * 1e3 + x0 #x_coordinate of gamma beam

        coords = sim.gen_Xray_dist(r, var)
        
        bounds = [
            x, x + gamma.get_width(),
            gamma.get_y(), gamma.get_y() + gamma.get_height()
            ]
        n.append(sim.get_overlap(coords, bounds))

    return n

    

if __name__ == '__main__':
    exp_x0 = -10e-12 * 3e8 * 1e3 #experimental delay
    x0 = np.linspace(-20e-12 * 3e8 * 1e3, 10e-12 * 3e8 * 1e3) #-10ps to 20ps delay
    hits = []
    for x in x0:
        hits.append(sum(simulate_hits(x)))

    delay = - x0 / 3e8 * 1e12 / 1e3 #in picoseconds

    np.savetxt('counter_hits_data.csv', np.array([delay, hits]), delimiter=',')
    
    plt.title('hit count against time delay')
    plt.xlabel('Delay (ps)')
    plt.ylabel('Number of hits')
    plt.plot(delay, hits, 'x')
    plt.show()
