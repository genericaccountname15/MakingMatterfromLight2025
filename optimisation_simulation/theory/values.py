"""
Stores values of common scientific constants in python variables.
Stores values of measurements, estimations, and experimental setup
parameters taken from the 2018 experiment.

Physical constants:
    - e: electronic charge (C)
    - c: speed of light in vacuum (m/s)
    - m: electron mass (kg)
    - re: classical electron radius

Simulation parameters from 2018:
    - gamma_length: length of gamma pulse (mm)
    - gamma_length_err: error in length of gamma pulse (mm)
    - gamma_radius: radius of gamma pulse (mm)
    - off_axial_dist: perpendicular distance from gamma pulse to Xray source when directly overhead (mm)
    - source_angle: angle of the Xray source (degrees)
    - Xray_FWHM: Full Width Half Maximum of Xray bath (mm)
    - delay_experiment: Pulse delay used (ps)
    - xray_number_density: Xray photon number density measured at 1mm from the target (m^-3)
    - xray_number_density_err: error in the Xray photon number density
    - gamma_photon_number: number of gamma photons
    - gamma_photon_number_err: error in the number of gamma photons
    - AMS_transmission: transmission efficiency of the AMS system
    - AMS_transmission_err: error in the AMS system transmission efficiency
    - xray_spectra_min: minimum Xray spectral value as measured by the Xray CCD (eV)
    - xray_spectra_max: maximum Xray spectral value as measured by the Xray CCD (ev)

Outdated variables:
    - gamma_energy: Constant gamma energy required for pair creation (MeV)
    - xray_energy: Energy of a peak in the Xray spectral distribution (MeV)
"""

# physical constants
e = 1.60e-19            # electronic charge (C)
c = 299792458           # speed of light in vacuum (m/s)
m = 9.11e-31            # electron mass (kg)
re = 2.8179403205e-15   # classical electron radius (m)

gamma_length = 45e-15 * 3e8 * 1e3  # length of gamma pulse (mm)
gamma_length_err = 5/45 * gamma_length

gamma_radius = 3.1 # radius of gamma pulse (mm)
off_axial_dist = 1 # perpendicular distance from gamma pulse to X-ray source (mm)
source_angle = 40 # angle of x-ray source (degrees)

xray_FWHM = 40e-12 * c * 1e3 # FWHM of xray bath (mm)
delay_experiment = 40 # pulse delay used in experiment (ps)

xray_number_density = 1.4e21 # xray photon number density measured at 1mm from target (m^-3) (+/-0.5e21)
xray_number_density_err = 0.5e21
gamma_photons_number = 7e6 # number of gamma (+/-1e6)
gamma_photons_number_err = 1e6

AMS_transmision = 0.25 # transmission efficiency of AMS system (+/-0.05)
AMS_transmision_err = 0.05

gamma_energy = 230 #MeV minimum energy required
xray_energy = 1.38e-3 #MeV, one of the peaks in the distribution

xray_spectra_min = 1300 #eV minimum x-ray spectra value as measured by the xray CCD 
xray_spectra_max = 1550 #eV minimum x-ray spectra value as measured by the xray CCD 