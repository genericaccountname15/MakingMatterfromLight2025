"""
runs the 4mm line source simulation while changing the angle
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import deep, accurate_line_5mm #pylint: disable=import-error
from theory import values   #pylint: disable=import-error



run_data_collection(
    variables = np.linspace(0, 90, 10) * np.pi / 180,
    variable_name = 'angle',
    variable_parameter_name = 'rotation',
    units = 'radians',
    old_value = values.source_angle * np.pi / 180,
    xray_type = 'line',
    repeat = 3,
    additional_label = '5mm_4Aug',
    sim_params = accurate_line_5mm,
    sample_params = deep
    )