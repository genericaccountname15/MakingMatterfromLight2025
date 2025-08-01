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
from data_collection.data_params import accurate, quick, quick_line #pylint: disable=import-error

def example_hit_counter(xray_type: str = 'uniform'):
    """
    Runs hit counter on experimental values
    """
    if xray_type == 'uniform':
        xray = Xray(
            fwhm = accurate['fwhm'],
            rotation = accurate['rotation'],
            n_samples_angular = quick['angle samples'],
            n_samples = quick['samples per angle']
        )
    
    elif xray_type == 'lambertian':
        xray = XrayLambertian(
            fwhm = accurate['fwhm'],
            rotation = accurate['rotation'],
            n_samples_angular = quick['angle samples'],
            n_samples = quick['samples per angle']
        )
    
    elif xray_type == 'line':
        xray = XrayLine(
            fwhm = accurate['fwhm'],
            rotation = accurate['rotation'],
            line_length=accurate['line length'],
            n_samples_angular = quick_line['angle samples'],
            n_samples = quick_line['samples per angle'],
            n_line_samples = quick_line['line samples']
        )
    
    else:
        raise ValueError(f'Invalid xray type: {xray_type}. Expected "uniform", "lambertian", or "line".')


    gamma = Gamma(
        x_pos = accurate['x pos'],
        pulse_length = accurate['pulse length'],
        height = accurate['pulse height'],
        off_axis_dist = accurate['off axis dist']
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
