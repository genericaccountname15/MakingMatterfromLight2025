"""
xray_spectra_Ge.py

Defines the XraySpectraGe class. Reads Ge_spectrum.csv
with methods to plot and sample a probability density function based on the data.
The data was given by Rose Steven.

Timothy Chew
6/8/25
"""

import numpy as np
import matplotlib.pyplot as plt

class XraySpectraGe():
    def __init__(
            self,
            datafile: str
            ):
        self.file_dir = datafile
        self.energy, self.flux = self.get_data()
    
    def get_data(self):
        data = np.loadtxt(self.get_datafile(), skiprows=1, delimiter=',')
        energy = data[:,0]
        flux = data[:, 1:]
        return energy, flux
    
    def replicate_plot(self):
        _, ax = plt.subplots()

        ax.set_title('Germanium energy spectra')
        ax.set_xlabel('photon energy (keV)')
        ax.set_ylabel('flux (ph/$cm^2$/s)')

        ax.semilogy(self.get_energy(), self.get_flux()[:,0], label = '0$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,1], label = '50$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,2], label = '150$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,3], label = '200$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,4], label = '250$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,5], label = '300$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,6], label = '350$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,7], label = '400$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,8], label = '450$\\mu$m')
        ax.semilogy(self.get_energy(), self.get_flux()[:,9], label = '500$\\mu$m')

        ax.set_xlim(0, 12)

        ax.legend()

        plt.show()
    
    def sample_pdf(self, n: int
                   ) -> tuple:
        prob = np.average(self.get_flux(), axis=1)
        prob /= np.sum(prob) #normalisation

        samples = np.random.choice(self.get_energy(), size = n, p = prob)

        return samples

    
    def get_datafile(self):
        return self.file_dir
    
    def get_energy(self):
        return self.energy
    
    def get_flux(self):
        return self.flux
