"""
dataparams.py

Contains dict objects for ideal data collection parameters.

Timothy Chew
1/8/25
"""
import numpy as np
from theory import values       #pylint: disable=import-error

quick = {
    'angle samples': 200,
    'samples per angle': 10,
    'azimuthal samples': 5,
    'delay samples': 50
}

quick_line = {
    'angle samples': 100,
    'samples per angle': 5,
    'azimuthal samples': 5,
    'line samples': 5,
    'delay samples': 50
}

deep = {
    'angle samples': 400,
    'samples per angle': 20,
    'azimuthal samples': 50,
    'delay samples': 100
}

deep_line = {
    'angle samples': 100,
    'samples per angle': 10,
    'azimuthal samples': 50,
    'line samples': 10,
    'delay samples': 100
}

accurate = {
    'fwhm': values.xray_FWHM,
    'rotation': values.source_angle * np.pi / 180,
    'x pos': -values.delay_experiment * 1e-12 * values.c * 1e3,
    'pulse length': values.gamma_length,
    'pulse height': values.gamma_radius,
    'off axis dist': values.off_axial_dist,
    'bath vis': False,
    'line length': 1
}

visual = {
    'fwhm': 10,
    'rotation': 0,
    'x pos': -300,
    'pulse length': 100,
    'pulse height': 50,
    'off axis dist': 100,
    'bath vis': True,
    'line length': 100
}

accurate_line_4mm = {
    **accurate, **{'line length': 4}
}

accurate_line_5mm = {
    **accurate, **{'line length': 5}
}