"""
gen_electron_beam.py

Defines the ElectronSpectra class.

Reads the electron beam dataset and generates g4beamline
beams corresponding to that dataset.
Also can replicate plots from the 2018 paper.

Timothy Chew
13/8/25
"""
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

class ElectronSpectra:
    """
    The Electron Spectra class. Reading the electron beam
    spectra dataset and generation of g4beamline.beam objects.

    Dataset A corresponds to optimum beam performance
    Dataset B corresponds to commissioned beam performance

    Args:
        dataset_a_dir (str): directory for dataset A
        dataset_b_dir (str): directory for dataset B

    Attributes:
        dataset_a_dir (str): directory for dataset A
        dataset_b_dir (str): directory for dataset B
        energy_a (list[float]): mean electron energies for dataset A
        charge_a (list[float]): mean electron beam charge for dataset A
        charge_a_err (list[float]): error in electron beam charge for dataset A
        energy_b (list[float]): mean electron energies for dataset B
        charge_b (list[float]): mean electron beam charge for dataset B
        charge_b_err (list[float]): error in electron beam charge for dataset B

    Methods:
        get_data(file_dir: str) -> tuple[list[float]]: Obtains, combines and averages
                                                    electron beam datasets
        replicate_plot(): Replicates plot from the 2018 paper
        sample_pdf(n_samples: int) -> list[float]: Samples the electron energy spectrum
        get_e_momentum(energy: float) -> float: Returns the electron momentum from its energy
        get_beam_params(bins: int, n_samples: int) -> tuple[list[float]]: Generates params for
                                                g4beamline beam objects based off distribution
        gen_beam_objects(bins: int, n_samples: int): Writes the g4beamline parameters
                                as beam objects to a .g4bl file
    """
    def __init__(
            self,
            dataset_a_dir: str,
            dataset_b_dir: str
            ):
        self.dataset_a_dir = dataset_a_dir
        self.dataset_b_dir = dataset_b_dir

        self.energy_a, self.charge_a, self.charge_a_err = self.get_data(self.get_dataset_a_dir())
        self.energy_b, self.charge_b, self.charge_b_err= self.get_data(self.get_dataset_b_dir())

    # METHODS #####################################################################################
    def get_data(self, file_dir: str) -> tuple:
        """
        Obtains, combines and averages datasets.
        The electron beam datasets are stored in .mat files.

        Args:
            file_dir (str): the file directory where the electron beam dataset is stored.
        
        Returns:
            tuple[list[float]]: a tuple containing:
                - the mean electron energy (MeV)
                - the mean electron beam charge (pC)
                - the standard deviation in the electron beam charge (pC)
        """
        energy_list = []
        charge_list = []
        file_list = os.listdir(file_dir)
        for file in file_list:
            if file.endswith('.mat'):
                matfile = loadmat(file_dir + file)
                energy_list.append([matfile['Y_MEVlin'][0]])
                charge_list.append(matfile['spectrum_integrated_lin'])
        energy_mean = np.mean(energy_list, axis=1)[0]
        charge_mean = np.mean(charge_list, axis=0)[:,0]
        charge_err = np.std(charge_list, axis=0)[:,0]

        return energy_mean, charge_mean, charge_err

    def replicate_plot(self):
        """Replicates the plot from the paper (figure 3a)
        """
        _, ax = plt.subplots()
        ax.set_title('Electron Beam Spectrum')
        ax.set_xlabel('Electron Energy (MeV)')
        ax.set_ylabel('Charge (pC/MeV)')

        ax.plot(
            self.get_energy_a(),
            self.get_charge_a(),
            label = 'A (optimum)',
            color = 'blue'
        )

        ax.fill_between(
            x = self.get_energy_a(),
            y1 = self.get_charge_a() - self.get_charge_a_err(),
            y2 = self.get_charge_a() + self.get_charge_a_err(),
            color = 'blue',
            alpha = 0.3
        )

        ax.plot(
            self.get_energy_b(),
            self.get_charge_b(),
            label = 'B (commissioning)',
            color = 'purple'
        )

        ax.fill_between(
            x = self.get_energy_b(),
            y1 = self.get_charge_b() - self.get_charge_b_err(),
            y2 = self.get_charge_b() + self.get_charge_b_err(),
            color = 'purple',
            alpha = 0.3
        )

        ax.set_ylim(0, 0.2)
        ax.set_xlim(300,820)
        ax.set_axisbelow(True)
        ax.grid()
        ax.legend()

        plt.show()

    def sample_pdf(self, n_samples: int) -> list:
        """Samples the electron energy spectrum

        Args:
            n_samples (int): number of samples to take from the distribution

        Returns:
            list[float]: sampled energies
        """
        prob = self.get_charge_a()
        prob /= np.sum(prob) #normalisation
        samples = np.random.choice(self.get_energy_a(), size = n_samples, p = prob)

        return samples

    def get_e_momentum(self, energy: float):
        """
        Returns the electron momentum from its energy

        Args:
            energy (float): the electron's energy (MeV)

        Returns:
            float: magnitude of the electron's momentum (MeV/c)   
        """
        m_e = 0.511   #MeV/c^2
        return np.sqrt(energy ** 2 - m_e ** 2)

    def get_beam_params(self, bins: int, n_samples: int) -> tuple:
        """Generates params for g4beamline beam objects based off distribution

        Args:
            bins (int): the number of bins for the energy
            n_samples (int): the number of samples/g4beamline events

        Returns:
            tuple[list[int], list[float]]: a tuple containing
                - the number of events per energy
                - a list of energies
                - a list of energy sigmas
        """
        n_events, bin_edges = np.histogram(
            self.sample_pdf(n_samples),
            bins = bins
        )
        energy = [((bin_edges[i] + bin_edges[i+1]) / 2) for i in range(len(bin_edges) - 1)]
        energy_width = bin_edges[1] - bin_edges[0]

        return n_events, energy, energy_width

    def gen_beam_objects(self, bins: int, n_samples: int, gen_beams_filename='electron_beams.g4bl'):
        """Generates the g4bl containing the generated beam params
        by writing to a .g4bl file.

        Args:
            bins (int): the number of bins for the energy
            n_samples (int): the number of samples/g4beamline events
        """
        n_events_list, energy_list, energy_width = self.get_beam_params(bins, n_samples)
        p_list = self.get_e_momentum(np.array(energy_list))
        p_width = self.get_e_momentum(energy_width)

        p_list = np.round(p_list, 2)
        p_width = np.round(p_width, 2)

        with open(gen_beams_filename, 'w', encoding='utf-8') as file_handler:
            for i, n_events in enumerate(n_events_list):
                beam_string = (f"beam gaussian particle=e- nEvents={n_events} "
                               f"beamZ=0.0 weight=100 \\\n"
                            f"sigmaX=0.0 sigmaY=0.00 sigmaXp=0.0024 sigmaYp=0.0024 \\\n"
                            f"meanMomentum={p_list[i]} sigmaP={p_width} "
                            f"meanXp=0.00 meanYp=0.00 \\\n"
                            f"meanT=0.0 sigmaT=0.0 \n"
                            )
                file_handler.write(f"{beam_string}\n\n")

    # ACCESS METHODS ##############################################################################
    def get_dataset_a_dir(self) -> str:
        """Access method for dataset_a_dir

        Returns:
            str: directory of dataset A
        """
        return self.dataset_a_dir

    def get_dataset_b_dir(self) -> str:
        """Access method for dataset_b_dir

        Returns:
            str: directory of dataset B
        """
        return self.dataset_b_dir

    def get_energy_a(self) -> list:
        """Access method for energy_a

        Returns:
            list[float]: electron energies of dataset A
        """
        return self.energy_a

    def get_charge_a(self) -> list:
        """Access method for charge_a

        Returns:
            list[float]: mean electron charge of dataset A
        """
        return self.charge_a

    def get_charge_a_err(self) -> list:
        """Acesss method for charge_a_err

        Returns:
            list[float]: error in electron charge of dataset A
        """
        return self.charge_a_err

    def get_energy_b(self) -> list:
        """Access method for energy_b

        Returns:
            list[float]: electron energies of dataset B
        """
        return self.energy_b

    def get_charge_b(self) -> list:
        """Access method for charge_b

        Returns:
            list[float]: mean electron charge of dataset B
        """
        return self.charge_b

    def get_charge_b_err(self) -> list:
        """Acesss method for charge_b_err

        Returns:
            list[float]: error in electron charge of dataset B
        """
        return self.charge_b_err

if __name__ == '__main__':
    ebeams = ElectronSpectra(
        dataset_a_dir = 'noise_simulation/beam_generators/SpectraData/electron_beam_data/ElectronBeams/Dataset A/',
        dataset_b_dir = 'noise_simulation/beam_generators/SpectraData/electron_beam_data/ElectronBeams/Dataset A/'
        )
    ebeams.gen_beam_objects(100, 100000000, gen_beams_filename='electron_beam_100bins.g4bl')
