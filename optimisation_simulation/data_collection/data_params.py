"""
Contains dict objects for ideal data collection parameters.

Attributes:
    quick(dict): Sampling parameters for fast sampling
    quick_line(dict): Sampling parameters for fast sampling for a line xray source
    deep(dict): Sampling parameters for detailed sampling
    deep_line(dict): Sampling parameters for detailed sampling for a line xray source
    accurate(dict): Simulation parameters of the 2018 experiment
    accurate_line(dict): Simulation parameters of the 2018 experiment along with line source params
    visual(dict): Simulation parameters for the visualiser, selected for visual appeal
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
    'bath vis': False
}

accurate_line = {
    'fwhm': values.xray_FWHM,
    'rotation': 0,
    'x pos': -values.delay_experiment * 1e-12 * values.c * 1e3,
    'pulse length': values.gamma_length,
    'pulse height': values.gamma_radius,
    'off axis dist': values.off_axial_dist,
    'bath vis': False,
    'line length': 1,
    'wave speed': 1
}

visual = {
    'fwhm': 10,
    'rotation': 0,
    'x pos': -300,
    'pulse length': 100,
    'pulse height': 50,
    'off axis dist': 100,
    'bath vis': True,
    'line length': 100,
    'wave speed': 0.5
}

accurate_line_1mm = {
    **accurate_line, **{'line length': 1}
}

accurate_line_2mm = {
    **accurate_line, **{'line length': 2}
}

accurate_line_3mm = {
    **accurate_line, **{'line length': 3}
}

accurate_line_4mm = {
    **accurate_line, **{'line length': 4}
}

accurate_line_5mm = {
    **accurate_line, **{'line length': 5}
}