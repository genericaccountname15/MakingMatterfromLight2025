"""
runs the Palladium lambertian simulation while changing the angle
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, deep #pylint: disable=import-error
from theory import values   #pylint: disable=import-error



run_data_collection(
    variables = np.array([1.3962634, 1.57079633]),
    variable_name = 'angle',
    variable_parameter_name = 'rotation',
    units = 'radians',
    old_value = values.source_angle * np.pi / 180,
    xray_type = 'lambertian_Pd',
    repeat = 3,
    additional_label = 'Pd_4Aug_part3',
    sim_params = accurate,
    sample_params = deep
    )