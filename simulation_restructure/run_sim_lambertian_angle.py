"""
runs the lambertian simulation while changing the angle
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, deep #pylint: disable=import-error
from theory import values   #pylint: disable=import-error



run_data_collection(
    variables = np.linspace(0, 90, 10),
    variable_name = 'angle',
    variable_parameter_name = 'rotation',
    units = 'degrees',
    old_value = values.source_angle,
    xray_type = 'lambertian',
    repeat = 3,
    additional_label = '1Aug',
    sim_params = accurate,
    sample_params = deep
    )