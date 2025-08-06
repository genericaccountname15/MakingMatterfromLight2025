"""
main module needs no explanation sigma sigma sigma
"""

from core.gamma import Gamma    #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian     #pylint: disable=import-error
from analysis.hit_counter_gespec import HitCounterGe    #pylint: disable=import-error
from data_collection.data_params import accurate, quick #pylint: disable=import-error

# def example_hit_counter():
#     """
#     Runs hit counter on experimental values
#     """
#     xray = XrayLambertian(
#         fwhm = accurate['fwhm'],
#         rotation = accurate['rotation'],
#         n_samples_angular = quick['angle samples'],
#         n_samples = quick['samples per angle']
#     )


#     gamma = Gamma(
#         x_pos = accurate['x pos'],
#         pulse_length = accurate['pulse length'],
#         height = accurate['pulse height'],
#         off_axis_dist = accurate['off axis dist']
#     )

#     counter = HitCounterGe(
#             xray_bath = xray,
#             gamma_pulse = gamma,
#             n_samples_azimuthal = quick['azimuthal samples']
#         )

#     counter.plot_hit_count(
#         min_delay = -10,
#         max_delay = 500,
#         samples = quick['delay samples'],
#         show_exp_value = True
#     )

# example_hit_counter()
from data_collection.delay_optimise_collection import run_optimise_delay #pylint: disable=import-error


run_optimise_delay(
    dir_name = 'Germanium spectra',
    xray_type = 'lambertian',
    repeat = 5,
    spectra_type = 'Ge'
)
