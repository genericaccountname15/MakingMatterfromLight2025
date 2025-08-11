"""
Reads the electron beam data set and replicates plots
Similar structure to Gamma_spectra

Timothy Chew
8/8/25
"""
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

class ElectronSpectra:
    def __init__(
            self,
            dataset_a_dir: str,
            dataset_b_dir: str
            ):
        self.dataset_a_dir = dataset_a_dir
        self.dataset_b_dir = dataset_b_dir

        self.energy_a, self.charge_a, self.charge_a_err = self.get_data(self.get_dataset_a_dir())
        self.energy_b, self.charge_b, self.charge_b_err= self.get_data(self.get_dataset_b_dir())

    
    def get_data(self, file_dir):
        """
        Obtains, combines and averages datasets
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
        """Samples the electron spectral distribution

        Args:
            n_samples (int): number of samples to take from the distribution

        Returns:
            list[float]: sampled energies
        """
        prob = self.get_charge_a() 
        prob /= np.sum(prob) #normalisation
        samples = np.random.choice(self.get_energy_a(), size = n_samples, p = prob)

        return samples
    
    def get_e_momentum(self, energy):
        """Returns the electron momentum from its energy

        Args:
            energy (float): Electron energy in MeV
        """
        m_e = 0.511   #MeV/c^2
        return np.sqrt(energy ** 2 - m_e ** 2)
    
    def get_beam_params(self, bins, n_samples):
        """Generates params for g4beamline beam objects based off distribution

        Args:
            bins (_type_): _description_
            n_samples (_type_): _description_
        """
        n_events, bin_edges = np.histogram(
            self.sample_pdf(n_samples),
            bins = bins
        )
        energy = [((bin_edges[i] + bin_edges[i+1]) / 2) for i in range(len(bin_edges) - 1)]
        energy_width = bin_edges[1] - bin_edges[0]

        return n_events, energy, energy_width
    
    def gen_beam_objects(self, bins, n_samples):
        n_events_list, energy_list, energy_width = self.get_beam_params(bins, n_samples)
        p_list = self.get_e_momentum(np.array(energy_list))
        p_width = self.get_e_momentum(energy_width)

        p_list = np.round(p_list, 2)
        p_width = np.round(p_width, 2)

        with open("electron_beams.g4bl", "w", encoding='utf-8') as file_handler:
            for i, n_events in enumerate(n_events_list):
                beam_string = (f"beam gaussian particle=e- nEvents={n_events} beamZ=0.0 weight=100 \\\n"
                            f"sigmaX=0.0 sigmaY=0.00 sigmaXp=0.0024 sigmaYp=0.0024 \\\n"
                            f"meanMomentum={p_list[i]} sigmaP={p_width} meanXp=0.00 meanYp=0.00 \\\n"
                            f"meanT=0.0 sigmaT=0.0 \n"
                            )
                file_handler.write(f"{beam_string}\n\n")
                        



    def get_dataset_a_dir(self):
        return self.dataset_a_dir
    
    def get_dataset_b_dir(self):
        return self.dataset_b_dir
    
    def get_energy_a(self):
        return self.energy_a
    
    def get_charge_a(self):
        return self.charge_a

    def get_charge_a_err(self):
        return self.charge_a_err
    
    def get_energy_b(self):
        return self.energy_b
    
    def get_charge_b(self):
        return self.charge_b
    
    def get_charge_b_err(self):
        return self.charge_b_err


electronspec = ElectronSpectra(
    dataset_a_dir='g4bl_stuff/electron_beam_data/ElectronBeams/Dataset A/',
    dataset_b_dir='g4bl_stuff/electron_beam_data/ElectronBeams/Dataset B/'
)
electronspec.gen_beam_objects(10, 10000)