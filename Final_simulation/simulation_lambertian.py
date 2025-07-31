"""
Simulates a lambertian xray source
I = I_0 cos(theta)

Timothy Chew
30/07/25
"""

import os
import numpy as np
import values
from simulation import Xray, Gamma, Visualiser
from hit_counter import HitCounter

from tqdm import tqdm
from optimise_delay import write_data_csv
from plot_optimised_data import plot_optimised_data


class XrayLambertian(Xray):
    """Coordinates and distribution of the X-ray bath
    Inherits from the Xray bath class
    Uses a lambertian distribution for points instead

    Redefined methods:
        gen_xray_seed, resample, get_n_samples_total: 
                    redefined appropriately for a lambertian distribution
    """
    def __init__(self, FWHM, rotation=0, n_samples_angular=400, n_samples=10):
        super().__init__(FWHM, rotation, n_samples_angular, n_samples)

        self.xray_coords, self.n_samples_lambert = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples,
            get_n_lambert = True
        )

    def gen_xray_seed(self, mean, variance, rotation=0,
                      n_samples_angular = 400, n_samples = 10, get_n_lambert = False):
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
            # estimated lambert distribution
            lambert_samples = round(n_samples * np.cos( np.pi/2 - theta + rotation ))
            # random distribution centred at 0
            ndist = np.random.normal(mean, variance, lambert_samples)

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

    def resample(self, phi=0):
        """Resamples xray distribution depending
        on azimuthal angle

        Args:
            phi (_type_): _description_
        """
        if phi == 0:
            n_samples = self.get_n_samples()
        else:
            n_samples = round( self.get_n_samples() * np.cos( phi ) )

        self.xray_coords, self.n_samples_lambert = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation = self.get_rotation(),
            n_samples_angular = self.get_n_samples_angular(),
            n_samples = n_samples,
            get_n_lambert = True
        )

    def get_n_samples_total(self):
        return self.n_samples_lambert


class Test:
    """
    For running tests on the simulation
    """
    def __init__(self):
        pass

    ############ METHODS ###########################################################################
    def test_values(self):
        """
        Runs the simulation using experiment accurate values
        """
        xray = XrayLambertian(
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
        xray = XrayLambertian(
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
        xray = XrayLambertian(
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

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 10
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 50,
            show_exp_value = True,
            save_data = True
        )


    def collect_data(self, var):
        """Runs hit counter on experimental values
        Saves and plots data for analysis

        Args:
            var (list): variable we are changing
            Edit class definition params for this purpose
        """
        xray = XrayLambertian(
            FWHM = values.xray_FWHM,
            rotation= 40 * np.pi / 180,
            n_samples_angular = 400,
            n_samples = 10,
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = var #VALUE BEING VARIED BE SURE TO CHANGE
        )

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = 10
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 50,
            show_exp_value = True,
            save_data = True,
            save_params = True,
            plot_wait = 0.5
        )



###################################################################################################
def run_data_collection():
    """Runs a data collection algorithm for HPC runs
    """
    # INPUT PARAMETERS #############################################################################
    variables = np.linspace(2.0, 3.0, 3) #variable list
    variable_file_name = np.around(variables, 2) #what to label each individual file
    variable_name = 'd' # no spaces
    units = 'mm'
    old_value = 1 #value in 2018

    test = Test() #pylint: disable=redefined-outer-name
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)

    datadir = f'{variable_name}_optimisation_lambert'
    os.makedirs(datadir)


    for i, var in enumerate(tqdm(variables, desc = 'Data collection progress')):
        # directory name
        dir_name = f'{datadir}/sim_datafiles_{variable_name}_{variable_file_name[i]}_{units}'
        os.makedirs(dir_name)
        #repeat simulation 3 times
        for i in tqdm(range(1, 4), desc = 'Repeating simulations', leave = False):
            test.collect_data(var)
            os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')
        os.rename('Simulation_parameters.csv', f'{dir_name}/Simulation_parameters.csv')

    print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

  # WRITING DATA TO CSV #####################################################################
    write_data_csv(
        variable_name = f'{variable_name} ({units})',
        variable_list = variables,
        datadir = f'{variable_name}_optimisation_lambert',
        csvname = f'{datadir}/optimise_{variable_name}'
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
