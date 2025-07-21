"""
Replication of the plots in the 2018 paper

Timothy Chew
21/07/25
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

class gamma_spectra:
    """Gamma spectra data set
    Args:
        mat_fname (string): filename of the gamma spectrum matlab file (filename.mat)
    
    Attributes:
        _matfile (dict): loaded matlab file containing dataset
        _gamma_energy (list): energy of gamma photon (MeV)
        _sim_Nph_A_mean, _sim_Nph_A_sigma (list): mean and standard deviation of simulated optimal beam performance
        _sim_Nph_B_mean, _sim_Nph_B_sigma (list): mean and standard deviation of simulated commissioned beam performance
        _exp_Nph_A_mean, _exp_Nph_A_sigma (list): mean and standard deviation of experimental optimal beam performance
        _exp_Nph_B_mean, _exp_Nph_B_sigma (list): mean and standard deviation of experimental commissioned beam performance
    
    Methods:
        replicate_plot: replicates plots from the literature for sanity check
        sample_pdf: returns list of sampled energies from the gamma energy probability distribution
    """
    def __init__(self, mat_fname):
        self._matfile = loadmat(mat_fname)
        self._gamma_energy = self.matfile()['GammaEnergy_MeV'][0]

        #simulated values
        self._sim_Nph_A_mean = self.matfile()['SimNph_Mean_A'][0]
        self._sim_Nph_A_sigma = self.matfile()['SimNph_Sigma_A'][0]

        self._sim_Nph_B_mean = self.matfile()['SimNph_Mean_B'][0]
        self._sim_Nph_B_sigma = self.matfile()['SimNph_Sigma_B'][0]

        #experimental values
        self._exp_Nph_A_mean = self.matfile()['ExpNph_Mean_A'][0]
        self._exp_Nph_A_sigma = self.matfile()['ExpNph_Sigma_A'][0]

        self._exp_Nph_B_mean = self.matfile()['ExpNph_Mean_B'][0]
        self._exp_Nph_B_sigma = self.matfile()['ExpNph_Sigma_B'][0]

    def replicate_plot(self):
        """Replicates the plot displayed in the paper for sanity checks
        """
        fig, ax = plt.subplots() #pylint: disable=unused-variable
        ax.set_title('Gamma Spectrum')
        ax.set_xlabel('Gamma Photon Energy (MeV)')
        ax.set_ylabel('Photons/MeV')

        ax.plot(
            self.gamma_energy(),
            self.sim_Nph_A_mean(),
            label = 'A simulation',
            color = 'blue'
        )

        ax.fill_between(
            x = self.gamma_energy(),
            y1 = self.exp_Nph_A_mean() - self.exp_Nph_A_sigma(),
            y2 = self.exp_Nph_A_mean() + self.exp_Nph_A_sigma(),
            label = 'A measurement',
            color = 'blue',
            alpha = 0.3
        )

        ax.plot(
            self.gamma_energy(),
            self.sim_Nph_B_mean(),
            label = 'B simulation',
            color = 'purple'
        )

        ax.fill_between(
            x = self.gamma_energy(),
            y1 = self.exp_Nph_B_mean() - self.exp_Nph_B_sigma(),
            y2 = self.exp_Nph_B_mean() + self.exp_Nph_B_sigma(),
            label = 'B measurement',
            color = 'purple',
            alpha = 0.3
        )
        
        ax.set_yscale('log')
        ax.set_ylim(230, 5e6)
        ax.set_xlim(0, 800)

        ax.set_axisbelow(True)
        ax.grid()
        ax.legend()

        plt.show()

    def sample_pdf(self, n_samples):
        """Samples the gamma spectral distribution

        Args:
            n_samples (int): number of samples to take from the distribution

        Returns:
            list: sampled energies
        """
        prob = self.exp_Nph_A_mean() #using optimum beam performance
        prob /= np.sum(prob) #normalisation
        samples = np.random.choice(self.gamma_energy(), size = n_samples, p = prob)

        return samples


    # ACCESS METHODS #####################################################
    def matfile(self):
        return self._matfile
    
    def gamma_energy(self):
        return self._gamma_energy
    
    def sim_Nph_A_mean(self):
        return self._sim_Nph_A_mean
    
    def sim_Nph_B_mean(self):
        return self._sim_Nph_B_mean
    
    def sim_Nph_A_sigma(self):
        return self._sim_Nph_A_sigma
    
    def sim_Nph_B_sigma(self):
        return self._sim_Nph_B_sigma
    
    def exp_Nph_A_mean(self):
        return self._exp_Nph_A_mean
    
    def exp_Nph_A_sigma(self):
        return self._exp_Nph_A_sigma
    
    def exp_Nph_B_mean(self):
        return self._exp_Nph_B_mean
    
    def exp_Nph_B_sigma(self):
        return self._exp_Nph_B_sigma

class xray_spectra:
    """Xray spectra data set
    Args:
        file_dir (string): file directory address containing the xray spectra files in pickle format (.pickle)
        resolution (float, optional): resolution of xray energy bins for averaging over data sets. Defaults to 0.5eV.
    Attributes:
        file_dir (string): file directory address containing the xray spectra files in pickle format (.pickle)
        file_list (list): list of filenames within the directory specified
        resolution (float): resolution of xray energy bins for averaging over data sets (eV)
        _Energy (list): all energy values from all datafiles
        _Nph (list): all values of number of photons/eV/J/srad from all data files, corresponds to energy value
        _Nph_err (list): standard deviation of number of photons/eV/J/srad from all data files
        _avg_Energy (list): bin energies (centre of bin values)
        _avg_Nph (list): bin-average photon count
        _avg_Nph_err (list): bin-summed photon count standard deviation

    Methods:
        get_data: Combines data from all 47 datasets
        bin_data: Bins data into energy bins and averages within the bins
        replicate_plot: Replicates the plot from literature as a sanity check
        filter_energies: Filters out energies and isolates a range of energies
        sample_pdf: Samples energergies from xray energy spectrum probability distribution
    """
    def __init__(self, file_dir, resolution=0.5):
        self._file_dir = file_dir
        self._file_list = os.listdir(file_dir)
        self._resolution = resolution

        self._Energy, self._Nph, self._Nph_err = self.get_data()

        self._avg_Energy, self._avg_NpH = self.bin_data(self.Energy(), self.Nph(), bin_width=self.resolution())
        _, self._avg_NpH_err = self.bin_data(self.Energy(), self.Nph_err(), bin_width=self.resolution(), err=True)

    
    def get_data(self):
        """Obtains and combines data from all datasets

        Returns:
            tuple: compiled data for (energy, number of photons, Nph error)
        """
        Energy = np.array([])
        Nph = np.array([])
        Nph_err = np.array([])
        for file in self.file_list():
            if file.endswith('.pickle'):
                data = np.load(self.file_dir() + file, allow_pickle=True)
                Energy = np.append(Energy, data['E'][:][0], axis = 0)
                laser_energy = data['laser_energy']

                #normalise number of photons
                Nph = np.append(Nph, data['normalised_number']/laser_energy/np.pi/2, axis = 0)
                Nph_err = np.append(Nph_err, data['normalised_number_sem']/laser_energy/np.pi/2, axis = 0)
        
        return Energy, Nph, Nph_err
            
    def bin_data(self, x, y, bin_width=0.5, err=False):
        """Bins arrays of x and y data points and averages
        y values in bins
        Generated by ChatGPT

        Args:
            x (list): list of x data points
            y (list): list of y data points
            bin_width (float, optional): resolution of bins. Defaults to 0.5.

        Returns:
            tuple (list, list): bin centre coordinates, height of bins
        """
        # Define bin edges (uniform or custom spacing)
        bins = np.arange(x.min(), x.max() + bin_width, bin_width)

        # Digitize x-values to bin indices
        bin_indices = np.digitize(x, bins)

        # Aggregate y-values into bins
        binned_y = np.zeros(len(bins) - 1)
        for i in range(1, len(bins)):
            in_bin = bin_indices == i
            if err:
                binned_y[i - 1] = np.sum(y[in_bin]) #sum uncertainties
            else:
                binned_y[i - 1] = np.mean(y[in_bin])

        # Bin centers for plotting
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        return bin_centers, binned_y
    
    def replicate_plot(self):
        """Replicates plot from the 2018 paper for sanity check
        """
        fig, ax = plt.subplots() #pylint: disable=unused-variable
        ax.set_title('Xray Spectrum')
        ax.set_xlabel('Xray Energy (eV)')
        ax.set_ylabel('Photons/eV/J/srad')
        ax.set_axisbelow(True)
        ax.grid()

        ax.fill_between(
            x = self.avg_Energy(),
            y1 = 0,
            y2 = self.avg_Nph() + self.avg_Nph_err()/47,
            color = 'black'
        )
        
        ax.fill_between(
            x = self.avg_Energy(),
            y1 = 0,
            y2 = self.avg_Nph(),
            color = 'rebeccapurple'
        )

        ax.set_xlim(1300, 1550)
        ax.set_ylim(0, 1e11)

        plt.show()

    def filter_energies(self, min_energy, max_energy):
        """Filters energies between the 1.3-1.5keV energy range

        Args:
            min_energy (float): minimum energy (eV)
            max_energy (float): maximum energy (eV)
        """
        mask = (self.avg_Energy() >= min_energy) & (self.avg_Energy() <= max_energy)
        self._avg_Energy = self.avg_Energy()[mask]
        self._avg_NpH = self.avg_Nph()[mask]
        self._avg_NpH_err = self.avg_Nph_err()[mask]

    def sample_pdf(self, min_energy, max_energy, n):
        """Generates discrete probability distribution
        and returns a sample of energies

        Args:
            min_energy (float): minimum xray energy (eV)
            max_energy (float): maximum xray energy (eV)

        Returns:
            tuple (list, list): xray energies, probability per energy
        """
        self.filter_energies(min_energy, max_energy)
        prob = self.avg_Nph()
        prob /= np.sum(prob) #normalisation

        samples = np.random.choice(self.avg_Energy(), size = n, p = prob)

        return samples

    # ACCESS METHODS #########################################################
    def file_dir(self):
        return self._file_dir

    def file_list(self):
        return self._file_list
    
    def resolution(self):
        return self._resolution
    
    def Energy(self):
        return self._Energy
    
    def Nph(self):
        return self._Nph
    
    def Nph_err(self):
        return self._Nph_err
    
    def avg_Energy(self):
        return self._avg_Energy
    
    def avg_Nph(self):
        return self._avg_NpH
    
    def avg_Nph_err(self):
        return self._avg_NpH_err


class Test:
    """Testing objects in this pyfile
    """
    def __init__(self):
        pass

    def test_gamma_replicate_plot(self):
        gamma_data = gamma_spectra('Final_simulation/data_read/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
        gamma_data.replicate_plot()
    
    def test_xray_replicate_plot(self):
        xray_data = xray_spectra('Final_simulation\\data_read\\data\\XrayBath\\XraySpectra\\', resolution=0.5)
        xray_data.replicate_plot()

    def test_xray_sampling(self):
        xray_data = xray_spectra('Final_simulation\\data_read\\data\\XrayBath\\XraySpectra\\', resolution=0.5)
        sampling = xray_data.sample_pdf(min_energy=1300, max_energy=1500, n=1000)
        plt.title('Sampled x-ray distribution')
        plt.ylabel('N')
        plt.xlabel('Xray Energy (eV)')
        plt.hist(sampling, bins=100)
        plt.show()

    def test_gammma_sampling(self):
        gamma_data = gamma_spectra('Final_simulation/data_read/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
        sampling = gamma_data.sample_pdf(1000)
        plt.title('Sampled gamma distribution')
        plt.ylabel('N')
        plt.xlabel('Gamma energy (MeV)')
        plt.hist(sampling, bins=100)
        plt.show()

if __name__ == '__main__':
    test = Test()
    test.test_xray_replicate_plot()

