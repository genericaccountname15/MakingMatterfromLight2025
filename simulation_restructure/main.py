"""
main module needs no explanation sigma sigma sigma
"""

import numpy as np
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, deep #pylint: disable=import-error
from theory import values   #pylint: disable=import-error


from data_analysis.optimise_delay import write_data_csv

run_data_collection(
    variables = np.linspace(0.1, 3.0, 10),
    variable_name = 'd',
    variable_parameter_name = 'off axis dist',
    units = 'mm',
    old_value = values.off_axial_dist,
    xray_type = 'lambertian',
    repeat = 3,
    additional_label = '4Aug',
    sim_params = accurate,
    sample_params = deep
)

# write_data_csv(
#     'shite',
#     np.linspace(0,90,10) * np.pi / 180,
#     'data_test',
#     csvname='idgaf2'
# )
