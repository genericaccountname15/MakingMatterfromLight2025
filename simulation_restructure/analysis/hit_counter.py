"""
hit_counter.py

Defines the HitCounter class which inherits from the Simulation class.
Count the number of collisions between the gamma pulse and Xray bath by 
calculating future positions.
Predicts number of positron pairs to be generated and received by
the CsI detector.

Timothy Chew
1/8/25
"""
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from core.simulation import Simulation, Xray, Gamma                 #pylint: disable=import-error
from visualisation.visualisation import Visualiser                  #pylint: disable=import-error
import theory.values as values                                      #pylint: disable=import-error
from theory.cross_section import c_bw                               #pylint: disable=import-error
from theory.energy_spectra.spectral_data import xray_spectra, gamma_spectra     #pylint: disable=import-error

class HitCounter(Simulation):
    """Counts the number of collisions between the X-ray bath
    and Gamma pulse

    Attributes:
        n_samples_azimuthal (float): number of azimuthal samples to take for width of Xray pulse
    
    Methods:
        count_hits: Counts the total number of collisions for a given pulse timing
        calc_effective_height: Solves a geometric problem to find the new effective height of the 
                                gamma pulse when looking at an azimuthal plane of the x-ray bath.
        calc_effective_d: Solves geometric problem to calculate effective off-axis
                        displacement when looking at another azimuthal plane paramaterised by phi
        est_npairs: Estimates the number of positron pairs produced and lands on the CsI detector
        plot_hit_count: Plots the hit count and estimated number of pairs for a range of delays
                        (with option to save data)
    """
    def __init__(self, xray_bath, gamma_pulse, n_samples_azimuthal = 1):
        super().__init__(xray_bath, gamma_pulse)
        self.n_samples_azimuthal = n_samples_azimuthal


    ############ METHODS ######################################################################
    def count_hits(self, delay):
        """Counts the total number of collisions for a given pulse timing

        Args:
            delay (float): time delay of gamma pulse to x-ray ignition (ps)

        Returns:
            tuple (float, numpy.ndarray): number of collisions, 
                                        array of coordinates for each hit (x, y, angles)

        """
        #delay to distance in mm
        x0 = -delay * 1e-12 * 3e8 * 1e3
        self.gamma_pulse.set_x_pos(x0)

        #azimuthal angles to sweep
        max_angle = np.arctan( self.get_gamma_pulse().get_height() /
                              self.get_gamma_pulse().get_off_axis_dist() )
        angles_azim = np.linspace(-max_angle, max_angle, self.get_n_samples_azimuthal())

        total_hit_count = 0
        total_hit_coords = []
        samples = 0



        for phi in angles_azim:
            hit_count = 0
            hit_coords = np.array([])

            # resample xray distribution ###########################
            self.xray_bath.resample(phi = phi)

            # calculate 'effective gamma pulse parameters' ##########
            eff_height = self.calc_effective_height(
                r = self.get_gamma_pulse().get_height(),
                phi = phi,
                d = self.get_gamma_pulse().get_off_axis_dist()
            )

            eff_d = self.calc_effective_d(
                phi = phi,
                d = self.get_gamma_pulse().get_off_axis_dist()
            )
            hit_count, hit_coords = self.find_hits(eff_height = eff_height, eff_d = eff_d)

            if hit_count != 0:
                total_hit_count += hit_count
                if len(total_hit_coords) == 0:
                    total_hit_coords = hit_coords
                else:
                    total_hit_coords = np.append(total_hit_coords, hit_coords, axis=0)

            samples += self.get_xray_bath().get_n_samples_total()


        return total_hit_count, total_hit_coords, samples

    def calc_effective_height(self, r, phi, d):
        """
        Solves a geometric problem to find the new effective height of the 
        gamma pulse when looking at an azimuthal plane of the x-ray bath.
        When phi = 0 the effective height is the radius of the beam

        Args:
            r (float): radius of gamma pulse (mm)
            phi (float): azimuthal angle being considered (radians)
            d (float): off-axial displacement (mm)

        Returns:
            float: new effective height
        """
        dist = r

        if phi != 0:
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

        return dist

    def calc_effective_d(self, phi, d):
        """Solves geometric problem to calculate effective off-axis
        displacement when looking at another azimuthal plane paramaterised
        by phi

        Args:
            phi (float): azimuthal angle being considered (radians)
            d (float): off-axial displacement (mm)

        Returns:
            float: effective off-axis displacement
        """
        eff_d = d * np.sqrt(
            1 + np.tan(phi) ** 2
        )

        return eff_d

    def est_npairs(self, angles, samples):
        """Estimates the number of positron pairs produced
        and lands on the CsI detector

        Returns:
            angles: list of angles for each hit
        """

        # calculate cross section of each hit and sum #####################################
        xray_data = xray_spectra('Final_simulation/data_read/data/XrayBath/XraySpectra/',
                                 resolution=0.5)
        gamma_data = gamma_spectra(
            'Final_simulation/data_read/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')

        xray_energy_sample = xray_data.sample_pdf(
            min_energy = values.xray_spectra_min,
            max_energy = values.xray_spectra_max,
            n = len(angles)
        )

        gamma_energy_sample = gamma_data.sample_pdf(n_samples = len(angles))

        cs_list = []
        for i, angle in enumerate(angles):
            #get cross section
            # s = 2 * ( 1 - np.cos(angle) ) * 230 * 1.38e-3
            s = 2 * ( 1 - np.cos(angle) ) * gamma_energy_sample[i] * xray_energy_sample[i]/1e6
            cs = c_bw(np.sqrt(s)) #get cross sec for 230MeV root s
            cs *= 1e-28 #convert from barns to m^2
            cs_list.append(cs)

        # get maximum azimuthal angle we calculate ######################################
        max_angle = np.arctan( self.get_gamma_pulse().get_height() /
                        self.get_gamma_pulse().get_off_axis_dist() )

        # estimate number of positrons #################################################
        n_pos = ( values.xray_number_density * values.gamma_photons_number
                  * values.AMS_transmision * sum(cs_list)
                  / samples
                  * ( max_angle / np.pi ) )

        # estimate uncertainty ##########################################################
        uncertainty = np.sqrt(
        (values.gamma_photons_number_err/values.gamma_photons_number) ** 2
        + (values.xray_number_density_err/values.xray_number_density) ** 2
        + (values.gamma_length_err / values.gamma_length) ** 2
        ) * n_pos

        return np.array([n_pos, uncertainty])

    def plot_hit_count(self, min_delay, max_delay, samples=50, **kwargs):
        """Plots the hit count and estimated number of pairs for a
        range of delays

        Args:
            min_delay (float): Minumum pulse delay (ps)
            max_delay (float): Maximum pulse delay (ps)
            samples (int, optional): Number of delays to check.
            Defaults to 50.
            **kwargs: optional
                show_exp_value (bool, optional): Whether to plot the delay used in 2018.
                                                Defaults to False.
                save_data (bool, optional): Whether to save the plot data to a csv.
                                            Defaults to False
                plot_wait (float, optional): Time to leave plot open.
                                            Defaults to None
                save_params (bool, optional): Whether to save parameters for Xray and Gamma objects.
                                            Defaults to False
        """
        # kwaargs ##########################################################
        show_exp_value = kwargs.get('show_exp_value', False)
        save_data = kwargs.get('save_data', False)
        save_data_filename = kwargs.get('save_data_filename', 'Npos_plot_data')
        plot_wait = kwargs.get('plot_wait', None)
        save_params = kwargs.get('save_params', False)
        save_params_filename = kwargs.get('save_params_fname', 'Simulation_parameters')

        # set up figure ############################################
        fig, ax = plt.subplots()
        ax.set_title('Hit count against time delay')
        ax.set_xlabel('Delay (ps)')
        ax.set_ylabel('Number of hits')

        twin = ax.twinx()
        twin.set_ylabel('Number of positrons/pC incident on CsI')


        # generate values #########################################
        delay_list = np.linspace(min_delay, max_delay, samples)

        n_pos_list = []
        hit_count_list = []

        for delay in tqdm(delay_list, desc='Simulation', leave=False):
            hit_count, hit_coords, samples = self.count_hits(delay)
            #avoid numpy [:,3] splitting error when no hits are counted
            if hit_count == 0:
                n_pos_list.append(np.array([[0], [0]]))
            else:
                n_pos_list.append( self.est_npairs(angles = hit_coords[:, 3], samples = samples) )
            hit_count_list.append(hit_count)

            self.xray_bath.resample(phi = 0) #resample x-ray distribution

        n_pos_list = np.array(n_pos_list)


        # plot values ##################################################
        hits, = ax.plot(delay_list, hit_count_list, '-x',
                        label='Hit count', color='red')

        positrons, = twin.plot(delay_list, n_pos_list[:,0,0], '-o',
                               label='Number of positrons/pC incident on CsI', color='blue')
        fill_band = twin.fill_between(delay_list, n_pos_list[:,0,0] - n_pos_list[:,1,0],
                                      n_pos_list[:,0,0] + n_pos_list[:,1,0],
                                      color='blue', alpha=0.3, label='Uncertainty')

        if show_exp_value:
            exp_value = ax.axvline(x = values.delay_experiment,
                                    ymin = 0, ymax = 1,
                                    label = 'Delay used in 2018', color = 'orange')
            ax.legend(handles=[hits, positrons, fill_band, exp_value])
        else:
            ax.legend(handles=[hits, positrons, fill_band])
        ax.grid()


        if save_data:
            data = {
                'delay' : delay_list,
                'hit_count': hit_count_list,
                'Npos_CsI': n_pos_list[:,0,0],
                'Npos_CsI_err': n_pos_list[:,1,0],
                'n_samples': samples
            }

            with open(f'{save_data_filename}.pickle', 'wb') as f:
                pickle.dump(data, f)

        if save_params:
            df = pd.DataFrame([self.get_params()])
            df.to_csv(f'{save_params_filename}.csv',index=False)

        if plot_wait is not None:
            plt.show(block=False)
            time.sleep(plot_wait)
            plt.close(fig)

        else:
            plt.show()


    def get_params(self):
        """Get parameters of Xray and Gamma objects

        Returns:
            dict: parameters of objects
        """
        params = {
            'FWHM (mm)': self.get_xray_bath().get_fwhm(),
            'Rotation (rad)': self.get_xray_bath().get_rotation(),
            'Number of sampled angles': self.get_xray_bath().get_n_samples_angular(),
            'Number of samples per angle': self.get_n_samples(),
            'Height of gamma pulse (mm)': self.get_gamma_pulse().get_height(),
            'Length of gamma pulse (mm)': self.get_gamma_pulse().get_pulse_length(),
            'Off axial distance (mm)': self.get_gamma_pulse().get_off_axis_dist()
        }

        return params


    def plot_ang_dist(self, delay):
        """Plots the angular distribution of hits for
        a pulse delay

        Args:
            delay (float): delay between gamma pulse and Xray ignition (ps)
        """
        _, hit_coords, _ = self.count_hits(delay)
        angles = hit_coords[:,3]
        #angles = self.get_xray_bath().get_xray_coords()[:,2]

        plt.title('Angular distribution of hits')
        plt.xlabel('Angle')
        plt.ylabel('Quantity')
        plt.hist(angles, bins=50)
        plt.show()


    ############ ACCESS METHODS ############################################################
    def get_n_samples_azimuthal(self):
        """Access method for n_samples_azimuthal

        Returns:
            int: number of azimuthal samples to take
        """
        return self.n_samples_azimuthal


class Test:
    """
    For running tests on the hit counter
    """
    def __init__(self):
        pass

    def check_ang_dist(self):
        """
        Checks angular distribution
        """
        xray = Xray(
            fwhm = 10,
            rotation = 0
        )

        gamma = Gamma(
            x_pos = -300,
            pulse_length = 200,
            height = 100,
            off_axis_dist = 100
        )

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal=1
        )

        counter.plot_ang_dist(
            delay = 5000
        )

    def test_hit_reg(self):
        """
        Checks for an accurate hit registration system
        using one x-ray point object
        """
        xray = Xray(
            fwhm = 10,
            rotation = 0
        )
        xray.xray_coords = np.array([[0,0,np.pi/2]])

        gamma = Gamma(
            x_pos = -300,
            pulse_length = 200,
            height = 100,
            off_axis_dist = 100
        )

        vis = Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = True
        )

        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal=1
        )

        vis.plot()
        print(counter.find_hits())

    def test_hit_counter(self):
        """
        Runs hit counter on experimental values
        """
        xray = Xray(
            fwhm = values.xray_FWHM ,
            rotation = 40 * np.pi / 180,
            n_samples_angular = 400,
            n_samples = 10
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
            show_exp_value = True,
            save_data = True
        )

    def collect_data(self):
        """
        Runs hit counter on experimental values
        More samples taken for data collection
        """
        xray = Xray(
            fwhm = values.xray_FWHM ,
            rotation = 40 * np.pi / 180,
            n_samples_angular = 400,
            n_samples = 20
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
            samples = 100,
            show_exp_value = True,
            save_data = True
        )