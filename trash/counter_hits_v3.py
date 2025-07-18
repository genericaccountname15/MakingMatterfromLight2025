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
    FWHM = 40e-12 * 3e8 * 1e3 #FWHM of X-ray bath in mm
    xray_coords = sim.gen_Xray_seed(-FWHM, n_angular_samples, n_samples)
    gamma, bounds = sim.create_gamma_beam(x0, [beam_width, beam_height], d)

    hit_count, hit_coords = sim.find_hits(xray_coords, bounds)

    return hit_count

    

if __name__ == '__main__':
    exp_x0 = -10e-12 * 3e8 * 1e3 #experimental delay
    x0 = np.linspace(-500e-12 * 3e8 * 1e3, 10e-12 * 3e8 * 1e3) #-10ps to 20ps delay

    n_samples = 10
    n_angular_samples = 400

    hits = []
    for x in x0:
        hits.append(simulate_hits(x, n_angular_samples, n_samples))

    delay = - x0 / 3e8 * 1e12 / 1e3 #in picoseconds
    
    #save data to a csv file
    #np.savetxt('counter_hitsv3_data.csv', np.array([delay, hits]), delimiter=',')

    plt.title('hit count against time delay')
    plt.xlabel('Delay (ps)')
    plt.ylabel('Number of hits')
    plt.plot(delay, hits, 'x')
    plt.show()
