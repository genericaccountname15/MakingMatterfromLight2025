"""
Simulates a lambertian xray source
I = I_0 cos(theta)

Timothy Chew
30/07/25
"""

import numpy as np
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

    def resample(self, phi=np.pi/2):
        """Resamples xray distribution depending
        on azimuthal angle

        Args:
            phi (_type_): _description_
        """
        if phi == np.pi/2:
            n_samples = self.get_n_samples()
        else:
            n_samples = round( self.get_n_samples() * np.cos( phi ) )

        self.xray_coords = self.gen_Xray_seed(
            mean = -self.get_FWHM(),
            variance = self.get_variance(),
            rotation = self.get_rotation(),
            n_samples_angular = self.get_n_samples_angular(),
            n_samples = n_samples
        )
    
    def get_n_samples_total(self):
        return self.n_samples_lambert


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

    def test_hit_counter(self, VAR):
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
            off_axis_dist = VAR #VALUE BEING VARIED BE SURE TO CHANGE
        )

        counter = Hit_counter(
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


#####################################################################################################################
def run_data_collection():
    import os
    from tqdm import tqdm
    from Optimise_delay import write_data_csv
    from plot_optimised_data import plot_optimised_data

    # INPUT PARAMETERS ######################################################################################
    variables = np.linspace(2.0, 3.0, 20) #variable list
    variable_file_name = variables #what to label each individual file
    variable_name = 'd' # no spaces
    units = 'mm'
    old_value = 1 #value in 2018

    test = Test()
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)

    datadir = f'{variable_name}_optimisation_lambert'
    os.makedirs(datadir)


    for i, var in enumerate(tqdm(variables, desc = 'Data collection progress')):
        dir_name = f'{datadir}/sim_datafiles_{variable_name}_{variable_file_name[i]}_{units}' # directory name
        os.makedirs(dir_name)
        for i in tqdm(range(1, 4), desc = 'Repeating simulations', leave = False): #repeat simulation 3 times
            test.test_hit_counter(var)
            os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')
    
    print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

  # WRITING DATA TO CSV ########################################################################################  
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

    # PUSH DATA TO GITHUB ######################################################################################
    import subprocess
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", f"{variable_name} optimisation data files"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("Changes pushed to GitHub.")

if __name__ == '__main__':
    test = Test()
    test.test_hit_counter(1)