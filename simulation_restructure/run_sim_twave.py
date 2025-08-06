"""
runs a travelling wave simulation
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import deep, accurate_line_5mm #pylint: disable=import-error



run_data_collection(
    variables = np.linspace(0.5, 10, 10),
    variable_name = 'travelling_wave_speed',
    variable_parameter_name = 'wave speed',
    units = 'c',
    old_value = 1,
    xray_type = 'twave',
    repeat = 3,
    additional_label = '6Aug_fwhm1',
    sim_params = {**accurate_line_5mm, 'fwhm': 1},
    sample_params = deep
    )