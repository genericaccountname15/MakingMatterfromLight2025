"""
example_sim.py

Runs some example simulations on the simulation module

Timothy Chew
1/8/25
"""
import numpy as np

from theory import values       #pylint: disable=import-error
from core.gamma import Gamma    #pylint: disable=import-error
from core.xray import Xray      #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian     #pylint: disable=import-error
from core.xray_line import XrayLine                 #pylint: disable=import-error
from visualisation.visualisation import Visualiser  #pylint: disable=import-error

# example parameters ##############################################################################
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

# example simulation function #####################################################################
def example_sim(xray_type: str = 'uniform',sim_type: str = 'visual'):
    """Example run of the simulation

    Args:
        xray_type (str, optional): Type of xray source: uniform, lambertian, line
        sim_type (str, optional): Type of simulation: visual or accurate. Defaults to 'visual'
    """
    if sim_type == 'visual':
        sim_dict = visual
    elif sim_type == 'accurate':
        sim_dict = accurate
    else:
        raise ValueError(f'Invalid simulation type: {sim_type}. Expected "visual" or "accurate".')
    
    if xray_type == 'uniform':
        xray = Xray(
            fwhm = sim_dict['fwhm'],
            rotation = sim_dict['rotation']
        )
    elif xray_type == 'lambertian':
        xray = XrayLambertian(
            fwhm = sim_dict['fwhm'],
            rotation = sim_dict['rotation']
        )
    
    elif xray_type == 'line':
        xray = XrayLine(
            fwhm = sim_dict['fwhm'],
            rotation = sim_dict['rotation'],
            line_length = sim_dict['line length']
        )
    
    else:
        raise ValueError(f'Invalid xray type: {xray_type}. Expected "uniform", "lambertian", or "line".')

    gamma = Gamma(
        x_pos = sim_dict['x pos'],
        pulse_length = sim_dict['pulse length'],
        height = sim_dict['pulse height'],
        off_axis_dist = sim_dict['off axis dist']
    )

    vis = Visualiser(
        xray_bath = xray,
        gamma_pulse = gamma,
        bath_vis = sim_dict['bath vis']
    )

    vis.plot()

