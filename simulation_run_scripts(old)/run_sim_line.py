"""
runs the line simulation while changing the length
"""
import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, deep_line #pylint: disable=import-error



run_data_collection(
    variables = np.linspace(0.4, 5.0, 10),
    variable_name = 'line_source_length',
    variable_parameter_name = 'line length',
    units = 'mm',
    old_value = 0.4,
    xray_type = 'line',
    repeat = 3,
    additional_label = '4Aug',
    sim_params = accurate,
    sample_params = deep_line
)