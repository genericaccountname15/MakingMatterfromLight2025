import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from tqdm import tqdm

import optimisation_simulation.theory.values as values  #pylind: disable=import-error

def xray_gauss(t, x, y):
    return (
        np.exp( - ( np.sqrt( x**2 + y**2 ) - values.c * t - values.xray_FWHM * 1e-3 / 2.3548 ) ** 2 / (2 *  ( values.xray_FWHM * 1e-3 / 2.3548 ) ** 2 ) )
    )

def area_func(t, delay):
    xmin = values.c * ( t - delay )
    xmax = xmin + values.gamma_length * 1e-3
    ymin = values.off_axial_dist * 1e-3
    ymax = ymin + values.gamma_radius * 1e-3
    return (
        dblquad(lambda y, x: xray_gauss(t, x, y), xmin, xmax, ymin, ymax)[0]
    )

def sum_area_func(t_range, delay):
    hit_count = 0
    hit_count_list = []
    for t in t_range:
        hit_count += area_func(t, delay)
        hit_count_list.append(area_func(t, delay)[0])
    return np.max(hit_count_list)

def int_area_func(delay):
    return (
        quad(lambda t: area_func(t, delay), 0, np.inf)[0]
    )


if __name__ == '__main__':
    time = np.linspace(0,1000, 100) * 1e-12
    # time = np.arange(0, 500, 45e-3) * 1e-12
    delay_list = np.linspace(0,500) * 1e-12
    
    hit_list = []
    for t0 in tqdm(delay_list, desc='Integrating'):
        hit_list.append(int_area_func(t0))
    
    plt.title('estimated hit counting')
    plt.xlabel('delay/ps')
    plt.ylabel('BW pairs (AU)')

    plt.plot(delay_list * 1e12, hit_list)
    plt.show()