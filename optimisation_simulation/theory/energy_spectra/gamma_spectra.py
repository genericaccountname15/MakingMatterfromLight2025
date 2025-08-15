"""
gamma_spectra.py

Defines the GammaSpectra class, which reads the gamma pulse's energy spectral data matlab file,
with methods to plot and sample a probability density function based on the data.
The data was taken from the 2018 paper's zenodo records repository.
The data was used to plot figures 4b and 5a in the 2018 paper.

Timothy Chew
1/8/25
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


class GammaSpectra:
    """Gamma spectra data set
    Args:
        mat_fname (str): filename of the gamma spectrum matlab file (filename.mat)
    
    Attributes:
        matfile (dict): loaded matlab file containing dataset
        gamma_energy (list[float]): energy of gamma photon (MeV)
        sim_nph_a_mean, _sim_nph_a_sigma (list[float]):
            mean and standard deviation of simulated optimal beam performance
        sim_nph_b_mean, _sim_nph_b_sigma (list[float]):
            mean and standard deviation of simulated commissioned beam performance
        exp_nph_a_mean, _exp_nph_a_sigma (list[float]):
            mean and standard deviation of experimental optimal beam performance
        exp_nph_b_mean, _exp_nph_b_sigma (list[float]):
            mean and standard deviation of experimental commissioned beam performance
    
    Methods:
        replicate_plot(): replicates plots from the literature for sanity check
        sample_pdf(n_samples: int) -> list[float]:
            returns list of sampled energies from the gamma energy probability distribution
    """
    def __init__(
            self,
            mat_fname: str
        ):
        self.matfile = loadmat(mat_fname)
        self.gamma_energy = self.get_matfile()['GammaEnergy_MeV'][0]

        #simulated values
        self.sim_nph_a_mean = self.get_matfile()['SimNph_Mean_A'][0]
        self.sim_nph_a_sigma = self.get_matfile()['SimNph_Sigma_A'][0]

        self.sim_nph_b_mean = self.get_matfile()['SimNph_Mean_B'][0]
        self.sim_nph_b_sigma = self.get_matfile()['SimNph_Sigma_B'][0]

        #experimental values
        self.exp_nph_a_mean = self.get_matfile()['ExpNph_Mean_A'][0]
        self.exp_nph_a_sigma = self.get_matfile()['ExpNph_Sigma_A'][0]

        self.exp_nph_b_mean = self.get_matfile()['ExpNph_Mean_B'][0]
        self.exp_nph_b_sigma = self.get_matfile()['ExpNph_Sigma_B'][0]

    def replicate_plot(self):
        """Replicates the plot displayed in the paper for sanity check
        """
        _, ax = plt.subplots()
        ax.set_title('Gamma Spectrum')
        ax.set_xlabel('Gamma Photon Energy (MeV)')
        ax.set_ylabel('Photons/MeV')

        ax.plot(
            self.get_gamma_energy(),
            self.get_sim_nph_a_mean(),
            label = 'A simulation',
            color = 'blue'
        )

        ax.fill_between(
            x = self.get_gamma_energy(),
            y1 = self.get_exp_nph_a_mean() - self.get_exp_nph_a_sigma(),
            y2 = self.get_exp_nph_a_mean() + self.get_exp_nph_a_sigma(),
            label = 'A measurement',
            color = 'blue',
            alpha = 0.3
        )

        ax.plot(
            self.get_gamma_energy(),
            self.get_sim_nph_b_mean(),
            label = 'B simulation',
            color = 'purple'
        )

        ax.fill_between(
            x = self.get_gamma_energy(),
            y1 = self.get_exp_nph_b_mean() - self.get_exp_nph_b_sigma(),
            y2 = self.get_exp_nph_b_mean() + self.get_exp_nph_b_sigma(),
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

    def sample_pdf(self, n_samples: int) -> list:
        """Samples the gamma spectral distribution

        Args:
            n_samples (int): number of samples to take from the distribution

        Returns:
            list[float]: sampled energies
        """
        prob = self.get_exp_nph_a_mean() #using optimum beam performance
        prob /= np.sum(prob) #normalisation
        samples = np.random.choice(self.get_gamma_energy(), size = n_samples, p = prob)

        return samples


    # ACCESS METHODS #####################################################
    def get_matfile(self) -> dict:
        """Access method for matfile

        Returns:
            dict: data from the gamma spectra which was stored in a matfile
        """
        return self.matfile
    
    def get_gamma_energy(self) -> list:
        """Access method for gamma_energy

        Returns:
            list[float]: energy of the gamma photons (MeV)
        """
        return self.gamma_energy
    
    def get_sim_nph_a_mean(self) -> list:
        """Access method for sim_nph_a_mean

        Returns:
            list[float]: mean number of photons for simulated optimal beam performance
        """
        return self.sim_nph_a_mean
    
    def get_sim_nph_b_mean(self) -> list:
        """Access method for sim_nph_b_mean

        Returns:
            list[float]: mean number of photons for simulated commissioned beam performance
        """
        return self.sim_nph_b_mean
    
    def get_sim_nph_a_sigma(self) -> list:
        """Access method for sim_nph_a_sigma

        Returns:
            list[float]: standard deviation in the number of photons
                for simulated optimal beam performance
        """
        return self.sim_nph_a_sigma
    
    def get_sim_nph_b_sigma(self) -> list:
        """Access method for sim_nph_b_sigma

        Returns:
            list[float]: standard deviation in the number of photons
                for simulated commissioned beam performance
        """
        return self.sim_nph_b_sigma
    
    def get_exp_nph_a_mean(self) -> list:
        """Access method for exp_nph_a_mean

        Returns:
            list[float]: mean number of photons for experimental optimal beam performance
        """
        return self.exp_nph_a_mean
    
    def get_exp_nph_a_sigma(self) -> list:
        """Access method for exp_nph_a_sigma

        Returns:
            list[float]: standard deviation in the number of photons
                for experimental optimal beam performance
        """
        return self.exp_nph_a_sigma
    
    def get_exp_nph_b_mean(self) -> list:
        """Access method for exp_nph_mean

        Returns:
            list[float]: mean number of photons for experimental commissioned beam performance
        """
        return self.exp_nph_b_mean
    
    def get_exp_nph_b_sigma(self) -> list:
        """Access method for exp_nph_b_sigma

        Returns:
            list[float]: standard deviation in the number of photons
                for experimental commissioned beam performance
        """
        return self.exp_nph_b_sigma

