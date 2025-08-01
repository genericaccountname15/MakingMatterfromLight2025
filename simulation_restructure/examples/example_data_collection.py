"""
Example data collection
Varying angles
"""
from data_collection.data_collection import run_data_collection #pylint: disable=import-error
from data_collection.data_params import accurate, quick, quick_line #pylint: disable=import-error
from theory import values   #pylint: disable=import-error

def example_data_collection_lambertian():
    """Runs an example data collection run
    on a lambertian source
    """
    run_data_collection(
        variables = [0, 40, 90],
        variable_name = 'angle',
        variable_parameter_name = 'rotation',
        units = 'degrees',
        old_value = values.source_angle,
        xray_type = 'lambertian',
        repeat = 3,
        additional_label = 'example',
        sim_params = accurate,
        sample_params = quick
    )

def example_data_collection_line():
    """Runs an example data collection run
    on a line source
    """
    run_data_collection(
        variables = [0, 40, 90],
        variable_name = 'angle',
        variable_parameter_name = 'rotation',
        units = 'degrees',
        old_value = values.source_angle,
        xray_type = 'line',
        repeat = 3,
        additional_label = 'example',
        sim_params = accurate,
        sample_params = quick_line
    )