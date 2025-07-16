"""
Hit counter for the v3 simulation

Timothy Chew
16/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
import simulationv3 as sim

#initial conditions
beam_width = 45e-15 * 3e8 * 1e3
beam_height = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
d = 1


def simulate_hits(x0, n_angular_samples = 400, n_samples = 10):
    gamma, bounds = sim.create_gamma_beam(x0, [beam_width, beam_height], d)

    n = []
    # time iteration
    time = np.linspace(0,10,300)
    for t in time:
        #note: time is in pico seconds and distance in mm
        r = t * 1e-12 * 3e8 * 1e3#radial distance travelled by shell, r = ct
        x = t * 1e-12 * 3e8 * 1e3 + x0 #x_coordinate of gamma beam

        coords = sim.gen_Xray_dist(r, n_angular_samples, n_samples)
        
        bounds = [
            x, x + gamma.get_width(),
            gamma.get_y(), gamma.get_y() + gamma.get_height()
            ]
        n.append(sim.get_overlap(coords, bounds))

    return n

    

if __name__ == '__main__':
    exp_x0 = -10e-12 * 3e8 * 1e3 #experimental delay
    x0 = np.linspace(-120e-12 * 3e8 * 1e3, 10e-12 * 3e8 * 1e3) #-10ps to 20ps delay
    hits = []
    for x in x0:
        hits.append(sum(simulate_hits(x, 4000, 10)))

    delay = - x0 / 3e8 * 1e12 / 1e3 #in picoseconds
    
    plt.title('hit count against time delay')
    plt.xlabel('Delay (ps)')
    plt.ylabel('Number of hits')
    plt.plot(delay, hits, 'x')
    plt.show()
