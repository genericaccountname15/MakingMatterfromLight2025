"""
runs the lambertian simulation while changing d
"""
import numpy as np
from data_collection.data_collection_multicore import run_data_collection_multicore #pylint: disable=import-error
from data_collection.data_params import accurate, deep, quick #pylint: disable=import-error
from theory import values   #pylint: disable=import-error


if __name__ == '__main__':
    run_data_collection_multicore(
        variables = np.linspace(0.1, 3.0, 10),
        variable_name = 'd',
        variable_parameter_name = 'off axis dist',
        units = 'mm',
        old_value = values.off_axial_dist,
        xray_type = 'lambertian',
        repeat = 3,
        additional_label = 'multicore_test',
        sim_params = accurate,
        sample_params = quick
    )