"""
example_sim.py

Runs some example simulations on the simulation module

Timothy Chew
1/8/25
"""
from core.gamma import Gamma    #pylint: disable=import-error
from core.xray import Xray      #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian     #pylint: disable=import-error
from core.xray_line import XrayLine                 #pylint: disable=import-error
from visualisation.visualisation import Visualiser  #pylint: disable=import-error
from data_collection.data_params import accurate, visual    #pylint: disable=import-error


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

