"""
Data Collection Package
=======================

This package contains the submodules for bulk data collection.
Runs the simulation while varying specified simulation parameters
and plotting their effects.

Modules:
    - data_collection: changing a simulation parameter and measuring effect on total positron yield
    - data_params: dicts for simulation and sampling parameters
    - delay_optimise_collection: repeats a simulation and optimises the delay

Example:
    >>> import numpy as np
    >>> from data_collection.data_collection import run_data_collection 
    >>> from data_collection.data_params import accurate, quick
    >>> from theory import values
    >>> run_data_collection(
    >>>     variables = np.linspace(0,90,10),
    >>>     variable_name = 'angle',
    >>>     variable_parameter_name = 'rotation',
    >>>     units = 'degrees',
    >>>     old_value = values.source_angle,
    >>>     xray_type = 'lambertian',
    >>>     repeat = 3,
    >>>     additional_label = 'example',
    >>>     sim_params = accurate,
    >>>     sample_params = quick
    >>> )

Output:

.. image:: _static/Data_collection_example.png
    :alt: Example plot from data collection
    :width: 600px
"""
