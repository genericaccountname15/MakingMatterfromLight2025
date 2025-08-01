"""
example_hitcount.py

Runs a quick optimisation simulation over a range of delays

Timothy Chew
1/8/25
"""
from core.gamma import Gamma    #pylint: disable=import-error
from core.xray import Xray      #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian     #pylint: disable=import-error
from core.xray_line import XrayLine                 #pylint: disable=import-error
from analysis.hit_counter import HitCounter         #pylint: disable=import-error
from analysis.hit_counter_line import HitCounterLine    #pylint: disable=import-error
from examples.example_sim import accurate    #pylint: disable=import-error

# example parameters ##############################################################################
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

params = accurate

# Example hit counting ############################################################################
def example_hit_counter(xray_type: str = 'uniform'):
    """
    Runs hit counter on experimental values
    """
    if xray_type == 'uniform':
        xray = Xray(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = quick['angle samples'],
            n_samples = quick['samples per angle']
        )
    
    elif xray_type == 'lambertian':
        xray = XrayLambertian(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = quick['angle samples'],
            n_samples = quick['samples per angle']
        )
    
    elif xray_type == 'line':
        xray = XrayLine(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            line_length=params['line length'],
            n_samples_angular = quick_line['angle samples'],
            n_samples = quick_line['samples per angle'],
            n_line_samples = quick_line['line samples']
        )
    
    else:
        raise ValueError(f'Invalid xray type: {xray_type}. Expected "uniform", "lambertian", or "line".')


    gamma = Gamma(
        x_pos = params['x pos'],
        pulse_length = params['pulse length'],
        height = params['pulse height'],
        off_axis_dist = params['off axis dist']
    )

    if xray_type == 'line':
        counter = HitCounterLine(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = quick['azimuthal samples']
        )
    else:
        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = quick['azimuthal samples']
    )

    counter.plot_hit_count(
        min_delay = -10,
        max_delay = 500,
        samples = quick['delay samples'],
        show_exp_value = True
    )
