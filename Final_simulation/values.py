"""
Values of parameters from literature

Timothy Chew
18/07/25
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