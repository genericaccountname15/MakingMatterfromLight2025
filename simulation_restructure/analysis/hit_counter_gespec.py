"""
Use a different Germanium energy spectra without
the alumnium filter
"""
import numpy as np

from analysis.hit_counter import HitCounter     #pylint: disable=import-error
from theory import values                       #pylint: disable=import-error
from theory.cross_section import c_bw           #pylint: disable=import-error
from theory.energy_spectra.gamma_spectra import GammaSpectra  #pylint: disable=import-error
from theory.energy_spectra.xray_spectra_Ge import XraySpectraGe  #pylint: disable=import-error

class HitCounterGe(HitCounter):
    def est_npairs(self, angles, samples):
        """Estimates the number of positron pairs produced
        and lands on the CsI detector

        Args:
            angles (list[float]): list of angle coordinates for each detected hit
            samples (int): number of Xray coordinates generated

        Returns:
            list[float,float]: array containing:
                [estimated number of positrons, uncertainty]
        """

        # calculate cross section of each hit and sum #####################################
        xray_data = XraySpectraGe('simulation_restructure/theory/energy_spectra/data/Ge_spectrum.csv')
        gamma_data = GammaSpectra(
            'simulation_restructure/theory/energy_spectra/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')

        xray_energy_sample = xray_data.sample_pdf(
            n = len(angles)
        )

        gamma_energy_sample = gamma_data.sample_pdf(n_samples = len(angles))

        cs_list = []
        for i, angle in enumerate(angles):
            #get cross section
            # s = 2 * ( 1 - np.cos(angle) ) * 230 * 1.38e-3
            s = 2 * ( 1 - np.cos(angle) ) * gamma_energy_sample[i] * xray_energy_sample[i]/1e3
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