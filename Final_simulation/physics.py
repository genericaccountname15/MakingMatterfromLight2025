"""
For checking and confirming the 'physical reasoning' behind
the optimal hit count

Timothy Chew
24/7/25
"""

import numpy as np
import matplotlib.pyplot as plt
import simulation as sim


class Xray_flat(sim.Xray):
    def __init__(self, FWHM, rotation=0, n_samples_angular=50, n_samples=10):
        super().__init__(FWHM, rotation, n_samples_angular, n_samples)

        #use uniform distribution now
        self.xray_coords = self.gen_Xray_seed_flat(
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples
        )
    
    def gen_Xray_seed_flat(self, rotation=0, n_samples_angular=400, n_samples=10):
        coords = []

        #rotate distribution 180 degrees
        angles = np.linspace(0 + rotation, np.pi + rotation, n_samples_angular)
        for theta in angles:
            ndist = np.linspace(-self.get_FWHM(), self.get_FWHM()) # uniform distribution

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                coords.append([rotated_coords[0], rotated_coords[1], theta])
        
        return np.array(coords)

    def move_Xrays(self, t):
        t *= 1e-9 #mm and picosecond, unit conversion
        moved_coords = []
        for r in self.get_xray_coords():
            #calculate distance to move
            dx = 0
            dy = 0

            moved_coords.append([r[0] + dx, r[1] + dy, r[2]])
        
        return np.array(moved_coords)


class simulation_flat(sim.Simulation):
    def __init__(self, xray_bath, gamma_pulse, n_samples_angular=400, n_samples=10):
        super().__init__(xray_bath, gamma_pulse, n_samples_angular, n_samples)
    
    def find_hits(self, eff_height=None, eff_d=None):
        """Calculate hits using X-ray seed coordinates
        Check future position and see if registers a hit

        Args:
            eff_height (float): effective height if on an off-azimuthal position
            eff_d (float): effective off-axis displacement
        
        Returns:
            tuple (float, list): (number of ovelaps, coordinates of overlaps [time, x, y, angle])
        """
        seed_coords = self.get_xray_bath().get_xray_coords()
        bounds = self.get_gamma_pulse().get_bounds()
     
        if eff_d is not None:
            bounds[2] += eff_d

        if eff_height is not None:
            bounds[3] = bounds[2] + eff_height

        #unit conversion back to metres (m)
        beam_bounds = np.array(bounds) * 1e-3
        coords = seed_coords * np.array([1e-3, 1e-3, 1])


        n = 0 #hit counter
        overlap_coords = [] #coordinates of hits [time, x, y, theta]

        for r in coords:
            if beam_bounds[2] < r[1] < beam_bounds[3]:
                n += 1
                overlap_coords.append([0, r[0], r[1], r[2]])


        if len(overlap_coords) == 0:
            return 0, np.array([0,0,0,0])
        else:
            #unit conversion
            overlap_coords = np.array(overlap_coords)
            overlap_coords *= np.array([1e12, 1e3, 1e3, 1])


        return n, overlap_coords


class Hit_counter_flat(simulation_flat):
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
    def __init__(self, xray_bath, gamma_pulse, n_samples_angular=400, n_samples=10, n_samples_azimuthal = 1):
        super().__init__(xray_bath, gamma_pulse, n_samples_angular, n_samples)
        self.n_samples_azimuthal = n_samples_azimuthal

    ############ METHODS #####################################################################################
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
        angles_azim = np.linspace(0, max_angle, self.get_n_samples_azimuthal())

        total_hit_count = 0
        total_hit_coords = []


        for phi in angles_azim:
            hit_count = 0
            hit_coords = np.array([])
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


        return total_hit_count, total_hit_coords

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
            1 + np.tan(phi)
        )

        return eff_d
    
    def est_npairs(self, angles):
        """Estimates the number of positron pairs produced
        and lands on the CsI detector

        Returns:
            angles: list of angles for each hit
        """
        import values as values
        from cross_section import c_BW
        from data_read.spectral_data import xray_spectra, gamma_spectra
        
        # calculate cross section of each hit and sum #####################################
        xray_data = xray_spectra('Final_simulation\\data_read\\data\\XrayBath\\XraySpectra\\', resolution=0.5)
        gamma_data = gamma_spectra('Final_simulation/data_read/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
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
            cs = c_BW(np.sqrt(s)) #get cross sec for 230MeV root s
            cs *= 1e-28 #convert from barns to m^2
            cs_list.append(cs)

        # get maximum azimuthal angle we calculate ######################################
        max_angle = np.arctan( self.get_gamma_pulse().get_height() / 
                        self.get_gamma_pulse().get_off_axis_dist() )
        
        # estimate number of positrons #################################################
        N_pos = ( values.xray_number_density * values.gamma_photons_number
                  * values.AMS_transmision * sum(cs_list) 
                  / self.get_n_samples_azimuthal()
                  / self.get_n_samples_angular()
                  / self.get_n_samples()
                  * ( max_angle / np.pi ) )
        
        # estimate uncertainty ##########################################################
        uncertainty = np.sqrt(
        (values.gamma_photons_number_err/values.gamma_photons_number) ** 2
        + (values.xray_number_density_err/values.xray_number_density) ** 2
        + (values.gamma_length_err / values.gamma_length) ** 2
        ) * N_pos
        
        return np.array([N_pos, uncertainty])

    def plot_hit_count(self, min_delay, max_delay, samples=50, **kwargs):
        """Plots the hit count and estimated number of pairs for a
        range of delays

        Args:
            min_delay (float): Minumum pulse delay (ps)
            max_delay (float): Maximum pulse delay (ps)
            samples (int, optional): Number of delays to check. Defaults to 50.
            **kwargs: optional
                show_exp_value (bool, optional): Whether to plot the delay used in 2018. Defaults to False.
                save_data (bool, optional): Whether to save the plot data to a csv. Defaults to False
                show_progress_bar (bool, optional): Whether to print the progress bar. Defaults to True
        """
        # kwaargs ##########################################################
        show_exp_value = kwargs.get('show_exp_value', False)
        save_data = kwargs.get('save_data', False)
        save_data_filename = kwargs.get('save_data_filename', 'Npos_plot_data')
        show_progress_bar = kwargs.get('show_progress_bar', True)

        # set up figure ############################################
        fig, ax = plt.subplots() #pylint: disable=unused-variable
        ax.set_title('Hit count against time delay')
        ax.set_xlabel('Delay (ps)')
        ax.set_ylabel('Number of hits')

        twin = ax.twinx()
        twin.set_ylabel('Number of positrons/pC incident on CsI')


        # generate values #########################################
        delay_list = np.linspace(min_delay, max_delay, samples)

        N_pos_list = []
        hit_count_list = []

        # progress bar #########################
        if show_progress_bar:
            prog = 0
            print('#' * prog + '-' * ( len(delay_list) - prog ) + f'{ prog / len(delay_list) * 100 :.1f}%')

        for delay in delay_list:
            hit_count, hit_coords = self.count_hits(delay)
            N_pos_list.append( self.est_npairs(angles = hit_coords[:, 3]) )
            hit_count_list.append(hit_count)
            
            # progress bar updates #####################################
            if show_progress_bar:
                prog += 1
                print('#' * prog + '-' * ( len(delay_list) - prog ) + f'{ prog / len(delay_list) * 100 :.1f}%')
        N_pos_list = np.array(N_pos_list)

        
        # plot values ##################################################
        hits, = ax.plot(delay_list, hit_count_list, '-x', 
                        label='Hit count', color='red')

        positrons, = twin.plot(delay_list, N_pos_list[:,0,0], '-o', 
                               label='Number of positrons/pC incident on CsI', color='blue')
        fill_band = twin.fill_between(delay_list, N_pos_list[:,0,0] - N_pos_list[:,1,0], N_pos_list[:,0,0] + N_pos_list[:,1,0], 
                                      color='blue', alpha=0.3, label='Uncertainty')

        if show_exp_value:
            import values as values
            exp_value = ax.axvline(x = values.delay_experiment,
                                    ymin = 0, ymax = 1,
                                    label = 'Delay used in 2018', color = 'orange')
            ax.legend(handles=[hits, positrons, fill_band, exp_value])
        else:
            ax.legend(handles=[hits, positrons, fill_band])
        ax.grid()


        if save_data:
            import pickle
            data = {
                'delay' : delay_list,
                'hit_count': hit_count_list,
                'Npos_CsI': N_pos_list[:,0,0],
                'Npos_CsI_err': N_pos_list[:,1,0]
            }

            with open(f'{save_data_filename}.pickle', 'wb') as f:
                pickle.dump(data, f)

        plt.show()


    def plot_ang_dist(self, delay):
        """Plots the angular distribution of hits for
        a pulse delay

        Args:
            delay (float): delay between gamma pulse and Xray ignition (ps)
        """
        _, hit_coords = self.count_hits(delay)
        angles = hit_coords[:,3]
        #angles = self.get_xray_bath().get_xray_coords()[:,2]

        plt.title('Angular distribution of hits')
        plt.xlabel('Angle')
        plt.ylabel('Quantity')
        plt.hist(angles, bins=50)
        plt.show()
        

    ############ ACCESS METHODS ##############################################################################
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

    def test_hit_counter(self):
        """
        Runs hit counter on experimental values
        """
        import values as values
        from simulation import Xray, Gamma
        xray = Xray(
            FWHM = values.xray_FWHM,
            rotation = 40 * np.pi / 180
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius, 
            off_axis_dist = 3.0
        )

        counter = Hit_counter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_angular = 400,
            n_samples = 20,
            n_samples_azimuthal = 50
        )

        counter.plot_hit_count(
            min_delay = -10,
            max_delay = 500,
            samples = 100,
            show_exp_value = True,
            save_data = True
        )

    def check_ang_dist(self):
        """
        Checks angular distribution
        """
        from simulation import Xray, Gamma
        xray = Xray(
            FWHM = 10,
            rotation = 0
        )

        gamma = Gamma(
            x_pos = -300,
            pulse_length = 200,
            height = 100,
            off_axis_dist = 100
        )

        counter = Hit_counter(
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
        from simulation import Xray, Gamma, Visualiser
        xray = Xray(
            FWHM = 10,
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

        counter = Hit_counter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal=1
        )

        vis.plot()
        print(counter.find_hits())




class Test:
     def __init__(self):
          pass
     
     def test_geometry_vis(self):
        """Geometrical part
        Plots using the visiualiser
        """
        import values as values
        xray = Xray_flat(
            FWHM = values.xray_FWHM,
            rotation=0 * np.pi / 180
            )

        gamma = sim.Gamma(
            x_pos = -values.delay_experiment * 1e-12 * values.c * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius,
            off_axis_dist = values.off_axial_dist
            )

        vis = sim.Visualiser(
            xray_bath = xray,
            gamma_pulse = gamma,
            bath_vis = False
            )

        vis.plot()
     
     def test_geometry_hit(self):
        import values as values
        from simulation import Gamma
        xray = Xray_flat(
            FWHM = values.xray_FWHM,
            rotation = 40 * np.pi / 180
        )

        gamma = Gamma(
            x_pos = -10e-12 * 3e8 * 1e3,
            pulse_length = values.gamma_length,
            height = values.gamma_radius, 
            off_axis_dist = 3.0
        )

        counter = Hit_counter_flat(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_angular = 100,
            n_samples = 10,
            n_samples_azimuthal = 20
        )

        counter.plot_hit_count(
            min_delay = 0,
            max_delay = 500,
            samples = 10,
            show_exp_value = True,
            save_data = True
        )

if __name__ == '__main__':
    test = Test()
    test.test_geometry_hit()
