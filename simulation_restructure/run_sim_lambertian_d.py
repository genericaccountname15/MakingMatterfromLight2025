"""
runs the lambertian simulation while changing d
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, deep #pylint: disable=import-error
from theory import values   #pylint: disable=import-error


run_data_collection(
    variables = np.linspace(0.1, 3.0, 10),
    variable_name = 'd',
    variable_parameter_name = 'off axis dist',
    units = 'mm',
    old_value = values.off_axial_dist,
    xray_type = 'lambertian',
    repeat = 3,
    additional_label = '1Aug',
    sim_params = accurate,
    sample_params = deep
)