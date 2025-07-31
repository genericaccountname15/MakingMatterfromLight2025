"""
Runs the simulation using a 'line source'

Timothy Chew
28/07/25
"""

import os
from tqdm import tqdm
import numpy as np

from simulation_lambertian import XrayLambertian
from hit_counter import HitCounter
import values
from simulation import Gamma, Visualiser
from optimise_delay import write_data_csv
from plot_optimised_data import plot_optimised_data

class XrayLine(XrayLambertian):
    """Generates a line source by setting up an
    array of lambertian point sources

    Redefined methods:

    New methods:
    """
    def __init__(self, FWHM, line_length,rotation=0,
                 n_samples_angular=400, n_samples=10, n_line_samples=10):
        super().__init__(FWHM, rotation, n_samples_angular, n_samples)
        self.n_line_samples = n_line_samples
        self.line_length = line_length

        self.xray_coords, self.n_samples_total = self.gen_xray_seed_line( get_total_samples = True )

    def gen_xray_seed_line(self, phi = 0, get_total_samples=False):
        """Generates xray coordinates for a line source

        Returns:
            np.ndarray: array of line source generated xray coordinates
        """
        coords = []
        n_samples_total = 0

        if phi == 0:
            n_samples = self.get_n_samples()
        else:
            n_samples = round( self.get_n_samples() * np.cos( phi ) )


        for i in range(self.get_n_line_samples()):
            # generate point source coordinates
            gen_coords, n_samples_lambert = self.gen_xray_seed(
                mean = -self.get_fwhm(),
                variance = self.get_variance(),
                rotation=self.get_rotation(),
                n_samples_angular = self.get_n_samples_angular(),
                n_samples = n_samples,
                get_n_lambert = True
            )

            #shift point source coordinates
            shift = self.get_line_length() / self.get_n_line_samples() * i
            gen_coords[:, 0] -= shift * np.cos(self.get_rotation())
            gen_coords[:, 1] -= shift * np.sin(self.get_rotation())

            # append to coords
            if len(coords) == 0:
                coords = np.array(gen_coords)
            else:
                coords = np.append(coords, gen_coords, axis=0)

            n_samples_total += n_samples_lambert

        if get_total_samples:
            return coords, n_samples_total
        else:
            return coords

    def resample(self, phi=None):
        """Resamples x-ray distribution
        """
        if phi is None:
            self.xray_coords = self.gen_xray_seed_line(phi=0)
        else:
            self.xray_coords = self.gen_xray_seed_line(phi)

    def get_n_samples_total(self):
        return self.n_samples_total


    def get_n_line_samples(self):
        """Access method for number of line samples

        Returns:
            int: number of line samples
        """
        return self.n_line_samples

    def get_line_length(self):
        """Access method for line_length

        Returns:
            float: length of line source
        """
        return self.line_length


class HitCounterLine(HitCounter):
    def get_params(self):
        params = {
            'FWHM (mm)': self.get_xray_bath().get_fwhm(),
            'Rotation (rad)': self.get_xray_bath().get_rotation(),
            'Length of Xray source (mm)': self.get_xray_bath().get_line_length(),
            'Number of samples on line': self.get_xray_bath().get_n_line_samples(),
            'Number of sampled angles': self.get_xray_bath().get_n_samples_angular(),
            'Number of samples per angle': self.get_n_samples(),
            'Height of gamma pulse (mm)': self.get_gamma_pulse().get_height(),
            'Length of gamma pulse (mm)': self.get_gamma_pulse().get_pulse_length(),
            'Off axial distance (mm)': self.get_gamma_pulse().get_off_axis_dist()
        }
        
        return params

class Test:
    """Class for testing module
    And also collecting data
    """
    def __init__(self):
        pass

    def test_bath_vis(self):
        """Opens the visualiser for the experiment
        """
        xray = XrayLine(
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
        xray = XrayLine(
            FWHM = values.xray_FWHM,
            line_length = 1,
            rotation= 0 * np.pi / 180,
            n_line_samples = 5,
            n_samples_angular = 100,
            n_samples = 5,
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = values.off_axial_dist
        )

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 5
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 50,
            show_exp_value = True
        )

    def collect_data(self, var):
        """Runs hit counter on experimental values
        Saves and plots data for analysis

        Args:
            var (list): variable we are changing
            Edit class definition params for this purpose
        """
        xray = XrayLine(
            FWHM = values.xray_FWHM,
            line_length = var, # VARIABLE WE CHANGING, REMEMBER TO MOVE
            rotation= 0 * np.pi / 180,
            n_line_samples = 10,
            n_samples_angular = 100,
            n_samples = 10,
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = values.off_axial_dist
        )

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 50
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 50,
            show_exp_value = True,
            save_data = True
        )

def run_data_collection():
    """Runs the data collection process for HPC systems
    """
    # INPUT PARAMETERS #############################################################################
    variables = np.linspace(0.8, 5.0, 20) #variable list
    variable_file_name = variables #what to label each individual file
    variable_name = 'line_source_length' # no spaces
    units = 'mm'
    old_value = 0.8 #value in 2018

    test = Test()
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)

    datadir = f'{variable_name}_optimisation'
    os.makedirs(datadir)


    for i, var in enumerate(tqdm(variables, desc = 'Data collection progress')):
        # directory name
        dir_name = f'{datadir}/sim_datafiles_{variable_name}_{variable_file_name[i]}_{units}'
        os.makedirs(dir_name)
        #repeat simulation 3 times
        for i in tqdm(range(1, 4), desc = 'Repeating simulations', leave = False):
            test.collect_data(var)
            os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')

    print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

    # WRITING DATA TO CSV ########################################################################
    write_data_csv(
        variable_name = f'{variable_name} ({units})',
        variable_list = variables,
        datadir = f'{variable_name}_optimisation_lambert',
        csvname = f'{datadir}/optimise_{variable_name}.csv'
    )

    plot_optimised_data(
        filename = f'{datadir}/optimise_{variable_name}.csv',
        variable_name = variable_name,
        xlabel = f'{variable_name} ({units})',
        old_value = old_value,
        save_fig = True,
        fig_location = f'{datadir}'
    )

if __name__ == '__main__':
    run_data_collection()
