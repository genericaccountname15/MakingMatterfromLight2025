"""
plot_energy_spectra.py

Plots the energy spectra and probability density functions
from theory/energy_spectra/

Timothy Chew
1/8/25
"""
import matplotlib.pyplot as plt
from theory.energy_spectra.gamma_spectra import GammaSpectra    #pylint: disable=import-error
from theory.energy_spectra.xray_spectra import XraySpectra      #pylint: disable=import-error

def plot_gamma_spectra():
    """Plots the gamma energy spectra
    """
    gamma_data = GammaSpectra('simulation_restructure/theory/energy_spectra/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
    gamma_data.replicate_plot()

def plot_xray_spectra():
    """Plots the Xray energy spectra
    """
    xray_data = XraySpectra('simulation_restructure/theory/energy_spectra/data/XrayBath/XraySpectra/', resolution=0.5)
    xray_data.replicate_plot()

def plot_xray_pdf():
    """Plots the Xray energy probability density function
    """
    xray_data = XraySpectra('simulation_restructure/theory/energy_spectra/data/XrayBath/XraySpectra/', resolution=0.5)
    sampling = xray_data.sample_pdf(min_energy=1300, max_energy=1500, n=1000)
    plt.title('Sampled x-ray distribution')
    plt.ylabel('N')
    plt.xlabel('Xray Energy (eV)')
    plt.hist(sampling, bins=100)
    plt.show()

def plot_gamma_pdf():
    """Plots the Gamma energy probability density function
    """
    gamma_data = GammaSpectra('simulation_restructure/theory/energy_spectra/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
    sampling = gamma_data.sample_pdf(1000)
    plt.title('Sampled gamma distribution')
    plt.ylabel('N')
    plt.xlabel('Gamma energy (MeV)')
    plt.hist(sampling, bins=100)
    plt.show()