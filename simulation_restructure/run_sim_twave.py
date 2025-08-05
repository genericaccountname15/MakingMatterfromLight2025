"""
runs the 4mm line source simulation while changing the angle
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import deep, accurate_line_5mm #pylint: disable=import-error
from theory import values   #pylint: disable=import-error



run_data_collection(
    variables = np.linspace(values.c/2, values.c*3, 10) * np.pi / 180,
    variable_name = 'travelling_wave_speed',
    variable_parameter_name = 'wave speed',
    units = 'ms^-1',
    old_value = values.c,
    xray_type = 'twave',
    repeat = 3,
    additional_label = '5Aug',
    sim_params = accurate_line_5mm,
    sample_params = deep
    )