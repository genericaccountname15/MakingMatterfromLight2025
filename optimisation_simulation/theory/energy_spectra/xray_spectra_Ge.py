"""
Defines the XraySpectraGe class. Reads Ge_spectrum.csv
with methods to plot and sample a probability density function based on the data.
The data was given by Rose Steven.
This in an Xray spectra dataset measured without the Aluminium filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from theory import values   #pylint: disable=import-error

class XraySpectraGe():
    """Germanium Xray spectra data set

    Attributes:
        filename(str): location of the datafile
        energy(list[float]): list of Xray photon energies
        flux(list[float]): flux of the corresponding Xray photon energy
    """
    def __init__(
            self,
            datafile: str
            ):
        self.filename = datafile
        self.energy, self.flux = self.get_data()
    
    def get_data(self) -> tuple:
        """Reads the Xray spectrum datafile (.csv)

        Returns:
            tuple: Tuple containing:
                - list[float]: list of Xray photon energies
                - list[float]: flux of the corresponding Xray photon energy
        """
        data = np.loadtxt(self.get_filename(), skiprows=1, delimiter=',')
        energy = data[:,0]
        flux = data[:, 1:]
        return energy, flux
    
    def replicate_plot(self):
        """Plots the Xray spectrum
        """
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
                   ) -> list:
        """Samples a probability density function based off the Xray photon energy spectrum

        Args:
            n (int): number of samples to take

        Returns:
            list[float]: list of Xray photon energies as sampled from the distribution
        """
        prob = np.average(self.get_flux(), axis=1)
        prob /= np.sum(prob) #normalisation

        samples = np.random.choice(self.get_energy(), size = n, p = prob)

        return samples

    def normalise_nph(self):
        """Get ratio of whole spectra / spectra from 2018 experiment

        Returns:
            float: normalisation ratio
        """
        flux = np.average(self.get_flux(), axis=1)
        mask = (self.get_energy() >= values.xray_spectra_min/1000) & (self.get_energy() <= values.xray_spectra_max/1000)
        flux_filtered = flux[mask]
        normalise = sum(flux) / sum(flux_filtered)

        return normalise

    # ACCESS METHODS ##############################################################################
    def get_filename(self) -> str:
        """Access method for filename

        Returns:
            str: location of the datafile
        """
        return self.filename
    
    def get_energy(self) -> list:
        """Access method for energy

        Returns:
            list[float]: list of Xray photon energies
        """
        return self.energy
    
    def get_flux(self) -> list:
        """Access method for flux

        Returns:
            list[float]: flux of the corresponding Xray photon energy
        """
        return self.flux
