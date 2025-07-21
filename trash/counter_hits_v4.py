"""
Count number of hits but we do 3d approximation
Semi circular beam

Timothy Chew
17/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
import trash.simulationv3 as sim
from trash.cross_section import c_BW

def simulate_hits(delay, beam_length, beam_radius, d, n_azimuthal_samples=10, n_angular_samples = 400, n_samples = 10):
    """Counts number of hits between the gamma pulse and the X-ray bath

    Args:
        delay(float): time delay between X-ray pulse and gamma pulse passing overhead (ps)
        beam_length (float): length of gamma pulse (mm)
        beam_radius (float): radius of gamma pulse (mm)
        d (float): off-axial displacement (mm)
        n_azimuthal_samples (int, optional): number of azimuthal angles (perpendicular to beam direction) to sample. Defaults to 10.
        n_angular_samples (int, optional): number of angles to sample (parallel to beam direction). Defaults to 400.
        n_samples (int, optional): number of samples per angle. Defaults to 10.

    Returns:
        int: total number of hits incident on target
        list: list of numpy arrays
    """
    #delay to distance in mm
    x0 = -delay * 1e-12 * 3e8 * 1e3

    #azimuthal angles to sweep
    max_angle = np.arctan(beam_radius/d)
    angles_azim = np.linspace(0, max_angle, n_azimuthal_samples)

    FWHM = 40e-12 * 3e8 * 1e3 #FWHM of X-ray bath in mm

    total_hit_count = 0
    total_hit_coords = []


    for phi in angles_azim:
        hit_count = 0
        hit_coords = np.array([])
        # use line and circle intersection for length
        xray_coords = sim.gen_Xray_seed(-FWHM, n_angular_samples, n_samples)
        if phi == 0:
            gamma, bounds = sim.create_gamma_beam(x0, [beam_length, beam_radius], d)

            hit_count, hit_coords = sim.find_hits(xray_coords, bounds)
        else:
            r = beam_radius # for simplicity
            m = abs( np.tan(phi) ) #gradient of line, direction don't matter cuz symmetric
            if d * d - ( 1 + m * m) * ( d * d - r * r ) < 0:
                pass # ignore not intersecting
            else:
                x1 = ( d + np.sqrt( d * d 
                                        - ( 1 + m * m) * ( d * d - r * r ))
                                        ) / ( 1 + m * m )
                y1 = m * x1

                x2 = d
                y2 = m * d

                dist = np.sqrt( ( x1 - x2 ) ** 2 + ( y1 - y2 ) ** 2 )

                gamma, bounds = sim.create_gamma_beam(x0, [beam_length, dist], d)

                hit_count, hit_coords = sim.find_hits(xray_coords, bounds)
            
        
        if hit_count != 0:
            total_hit_count += hit_count
            if len(total_hit_coords) == 0:
                total_hit_coords = hit_coords
            else:
                total_hit_coords = np.append(total_hit_coords, hit_coords, axis=0)


    return total_hit_count, total_hit_coords


def npairs(hit_count, angles, n_azimuthal_samples, n_angular_samples, n_samples):
    """Converts the hit count to the number of positrons generated

    Args:
        hit_count (_type_): _description_
        angles (list): list of angles between hits
        n_azimuthal_samples (_type_): _description_
        n_angular_samples (_type_): _description_
        n_samples (_type_): _description_

    Returns:
        _type_: _description_
    """
    # experimental conditions
    nx = 1.4e21 #m^-3 photon number density of X-ray beam 1mm from target (+/-0.5)
    Ny = 7e6 #number of gamma photons (+/-1)
    AMS_transmission = 0.25 #estimated AMS transition (+/- 0.05)

    # hit ratio
    hit_ratio = hit_count / n_azimuthal_samples / n_angular_samples / n_samples

    #circular ratio
    beam_radius = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
    max_angle = np.arctan( beam_radius / d )
    hit_ratio = hit_ratio * ( max_angle / np.pi )

    cs_list = []
    for angle in angles:
        #get cross section
        s = 2 * ( 1 - np.cos(angle + np.pi/2) ) * 300 * 2e-3 #using 300 MeV cuz fk it
        cs = c_BW(np.sqrt(s)) #get cross sec for 230MeV root s
        cs *= 1e-28 #convert from barns to m^2
        cs_list.append(cs)

    # estimate number of positrons
    N_pos = nx * Ny * AMS_transmission * sum(cs_list) / n_azimuthal_samples / n_angular_samples / n_samples * ( max_angle / np.pi )

    #get uncertainty
    uncertainty = np.sqrt(
        (0.5/1.4) ** 2 + (1/7) ** 2 + (0.05/0.25) ** 2 
    ) * N_pos
    return np.array([N_pos, uncertainty]) # * 1e5 for readability

if __name__ == '__main__':
    #initial conditions
    beam_length = 45e-15 * 3e8 * 1e3
    beam_radius = 44e-6 * 1e3 #44 micrometres FWHM of drive laser, should use 0.6mrad instead
    # beam_radius = 3.1e-3 #3.1mm from Brendan
    d = 1
    FWHM = 40e-12 * 3e8 * 1e3 #FWHM of X-ray bath in mm


    exp_x0 = -10e-12 * 3e8 * 1e3 #experimental delay
    delay = np.linspace(-10, 500) #ps #experiment was 10ps

    n_samples = 10
    n_angular_samples = 400
    n_azimuthal_samples = 1

    hits = []
    N_pos = []

    #progress bar
    p = 0
    print('#' * p + '-' * ( len(delay) - p ) + f'{ p / len(delay) * 100 }%')

    for T in delay:
        p += 1
        hit_count, hit_coords = simulate_hits(T, beam_radius=beam_radius, beam_length=beam_length, d=d,
                                   n_azimuthal_samples=n_azimuthal_samples, n_angular_samples=n_angular_samples, n_samples=n_samples)
        hits.append(hit_count)
        N_pos.append(npairs(hit_count, hit_coords[:, 3], n_azimuthal_samples, n_angular_samples, n_samples))
        print('#' * p + '-' * ( len(delay) - p ) + f'{ p / len(delay) * 100 :.1f}%')
    N_pos = np.array(N_pos)

    #save data to a csv file
    #np.savetxt('counter_hitsv3_data.csv', np.array([delay, hits]), delimiter=',')
    fig, ax = plt.subplots()
    ax.set_title('Hit count against time delay')
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Number of hits')
    hits, = ax.plot(delay, hits, '-x', label='Hit count', color='red')

    twin = ax.twinx()
    twin.set_ylabel('Number of positrons/pC incident on CsI')
    positrons, = twin.plot(delay, N_pos[:,0,0], '-o', label='Number of positrons/pC incident on CsI', color='blue')
    fill_band = twin.fill_between(delay, N_pos[:,0,0]-N_pos[:,1,0], N_pos[:,0,0]+N_pos[:,1,0], color='blue', alpha=0.3, label='Uncertainty')

    ax.legend(handles=[hits, positrons, fill_band])
    ax.grid()

    plt.show()