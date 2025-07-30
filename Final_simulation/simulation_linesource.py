"""
Runs the simulation using a 'line source'

Timothy Chew
28/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Xray
from hit_counter import Hit_counter

class Xray_line(Xray):
    def __init__(self, FWHM, line_length,rotation=0, n_samples_angular=400, n_samples=10, n_line_samples=10):
        super().__init__(FWHM, rotation, n_samples_angular, n_samples)
        self.n_line_samples = n_line_samples
        self.line_length = line_length

        self.xray_coords = self.gen_Xray_seed_line()
    
    def gen_Xray_seed_line(self):
        coords = []
        
        for i in range(self.get_n_line_samples()):
            # generate point source coordinates
            gen_coords = self.gen_Xray_seed(
                mean = -self.get_FWHM(),
                variance = self.get_variance(),
                rotation=self.get_rotation(),
                n_samples_angular = self.get_n_samples_angular(),
                n_samples = self.get_n_samples()
            )

            #shift point source coordinates
            shift = self.get_line_length() / self.get_n_line_samples() * i
            gen_coords[:, 0] -= shift * np.cos(self.get_rotation())
            gen_coords[:, 1] -= shift * np.sin(self.get_rotation())

            # append to coords
            if len(coords) == 0:
                coords = gen_coords
            else:
                coords = np.append(coords, gen_coords, axis=0)

        return coords
    
    def resample(self):
        """Resamples x-ray distribution
        """
        self.xray_coords = self.gen_Xray_seed_line()


    def get_n_line_samples(self):
        return self.n_line_samples
    
    def get_line_length(self):
        return self.line_length
    
class Hit_counter_line(Hit_counter):
    def __init__(self, xray_bath, gamma_pulse, n_samples_azimuthal=1):
        super().__init__(xray_bath, gamma_pulse, n_samples_azimuthal)
    
    def est_npairs(self, angles):
        Npos = super().est_npairs(angles)
        Npos /= self.get_xray_bath().get_n_line_samples()
        return Npos


class Test:
    def __init__(self):
        pass

    def test_bath_vis(self):
        import values
        from simulation import Gamma, Visualiser
        xray = Xray_line(
            FWHM = values.xray_FWHM,
            line_length = 0.8,
            rotation= 0 * np.pi / 180,
            n_line_samples = 1
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
    
    def test_hit_counter(self):
        """
        Runs hit counter on experimental values
        """
        import values as values
        from simulation import Gamma
        xray = Xray_line(
            FWHM = values.xray_FWHM,
            line_length = 10.0,
            rotation= 0 * np.pi / 180,
            n_line_samples = 20,
            n_samples_angular = 100,
            n_samples = 10,
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius, 
            off_axis_dist = values.off_axial_dist
        )

        counter = Hit_counter_line(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 20
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 100,
            show_exp_value = True,
            save_data = True,
            plot_wait = 3
        )

if __name__ == '__main__':
    import os
    test = Test()
    for i in range(1, 4):
        test.test_hit_counter()
        os.rename('Npos_plot_data.pickle', f'Npos_plot_data{i}.pickle')

