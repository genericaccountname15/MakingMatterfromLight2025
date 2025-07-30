"""
Simulates a lambertian xray source
I = I_0 cos(theta)

Timothy Chew
30/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Xray, Gamma, Visualiser
from hit_counter import Hit_counter

class Xray_lambertian(Xray):
    def __init__(self, FWHM, rotation=0, n_samples_angular=400, n_samples=10):
        super().__init__(FWHM, rotation, n_samples_angular, n_samples)

        self.xray_coords, self.n_samples_lambert = self.gen_Xray_seed(
            mean = -self.get_FWHM(),
            variance = self.get_variance(),
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples,
            get_n_lambert = True
        )

    def gen_Xray_seed(self, mean, variance, rotation=0, n_samples_angular = 400, n_samples = 10, get_n_lambert = False):
        """Generates a lambertian distribution of X ray pulse in 2D

        Args:
            mean (float): mean of distribution, radial position (m)
            variance (float): variance of x-ray distribution (mm^2)
            n_samples_angular (int, optional): Number of angles to sample. Defaults to 400
            n_samples (int, optional): Number of samples per angle. Defaults to 10.

        Returns:
            list: list of coordinates for distribution points
        """
        coords = []
        n_lambert = 0

        #rotate distribution 180 degrees
        angles = np.linspace(0 + rotation, np.pi + rotation, n_samples_angular)
        for theta in angles:
            lambert_samples = round(n_samples * np.cos( np.pi/2 - theta + rotation )) # estimated lambert distribution
            ndist = np.random.normal(mean, variance, lambert_samples) # random distribution centred at 0

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                coords.append([rotated_coords[0], rotated_coords[1], theta])
            n_lambert += lambert_samples
        
        if get_n_lambert:
            return np.array(coords), n_lambert     
        else:
            return np.array(coords)
    
    def get_n_lambert(self):
        return self.n_samples_lambert
    

class Hit_counter_lambertian(Hit_counter):
    def est_npairs(self, angles):
        Npos = super().est_npairs(angles)
        Npos *= self.get_xray_bath().get_n_samples_angular() + self.get_xray_bath().get_n_samples()
        Npos /= self.get_xray_bath().get_n_lambert()
        return Npos


class Test:
    """
    For running tests on the simulation
    """
    def __init__(self):
        pass

    ############ METHODS #####################################################################################
    def test_values(self):
        """
        Runs the simulation using experiment accurate values
        """
        import values
        xray = Xray_lambertian(
            FWHM = values.xray_FWHM,
            rotation=40 * np.pi / 180
        )

        gamma = Gamma(
            x_pos = -values.delay_experiment * 1e-12 * values.c * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = values.off_axial_dist
        )

        vis = Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = False
        )

        vis.plot()
    
    def test_sim(self):
        """
        Runs the simulation using more visually appealing values
        """
        xray = Xray_lambertian(
            FWHM = 10,
            rotation = 40 * np.pi / 180
        )

        gamma = Gamma(
            x_pos = -300,
            pulse_length = 100,
            height = 50,
            off_axis_dist = 100
        )

        vis = Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = True
        )

        vis.plot()

    def test_hit_counter(self):
        """
        Runs hit counter on experimental values
        """
        import values
        xray = Xray_lambertian(
            FWHM = values.xray_FWHM,
            rotation= 40 * np.pi / 180,
            n_samples_angular = 400,
            n_samples = 20,
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius, 
            off_axis_dist = values.off_axial_dist
        )

        counter = Hit_counter_lambertian(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 5
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 100,
            show_exp_value = True,
            save_data = True,
            plot_wait = 0.5
        )


if __name__ == '__main__':
    import os
    test = Test()
    for i in range(1, 4):
        test.test_hit_counter()
        os.rename('Npos_plot_data.pickle', f'Npos_plot_data{i}.pickle')