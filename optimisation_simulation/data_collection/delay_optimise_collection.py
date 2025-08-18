"""
Module to collect optimising delay data
"""
import os
from tqdm import tqdm

from core.xray import Xray      #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian #pylint: disable=import-error
from core.xray_lambertian_Ge import XrayLambertianGe #pylint: disable=import-error
from core.xray_lambertian_Pd import XrayLambertianPd #pylint: disable=import-error
from core.xray_line import XrayLine         #pylint: disable=import-error
from core.xray_line_twave import XrayTwave  #pylint: disable=import-error
from core.gamma import Gamma    #pylint: disable=import-error
from analysis.hit_counter import HitCounter #pylint: disable=import-error
from analysis.hit_counter_line import HitCounterLine #pylint: disable=import-error
from analysis.hit_counter_twave import HitCounterTwave  #pylint: disable=import-error
from analysis.hit_counter_gespec import HitCounterGe  #pylint: disable=import-error
from data_collection.data_params import deep, deep_line, accurate   #pylint: disable=import-error
from data_analysis.optimise_delay import plot_data, avg_data        #pylint: disable=import-error

def run_hit_counter(
        params: dict = None,
        sampling: dict = None,
        xray_type: str = 'uniform',
        spectra_type: str = 'default'
        ):
    """Runs hit counter while changing variable var

    Args:
        var (dict): variable to vary {'name': float}
        params (dict): parameters of experiment. Defaults to accurate.
        sampling (dict): sampling parameters. Defaults to deep.
        xray_type (str, optional): type of xray source use e.g. 'lambertian'. Defaults to 'uniform'.

    Raises:
        ValueError: When calling for an unsupported Xray source type
    """
    if dict is None:
        params = accurate
    if sampling is None:
        sampling = deep

    if xray_type == 'uniform':
        xray = Xray(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )

    elif xray_type == 'lambertian':
        xray = XrayLambertian(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )
    
    elif xray_type == 'lambertian_Ge':
        xray = XrayLambertianGe(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )

    elif xray_type == 'lambertian_Pd':
        xray = XrayLambertianPd(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )

    elif xray_type == 'line':
        if sampling == deep:
            sampling = deep_line
        xray = XrayLine(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            line_length = params['line length'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle'],
            n_line_samples = sampling['line samples']
        )

    elif xray_type == 'twave':
        if sampling == deep:
            sampling = deep_line
        xray = XrayTwave(
            fwhm = params['fwhm'],
            rotation = params['rotation'],
            line_length = params['line length'],
            wave_speed = params['wave speed'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle'],
            n_line_samples = sampling['line samples']
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
            n_samples_azimuthal = sampling['azimuthal samples']
        )
    elif xray_type == 'twave':
        counter = HitCounterTwave(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = sampling['azimuthal samples']
        )
    elif spectra_type == 'Ge':
        counter = HitCounterGe(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = sampling['azimuthal samples']
        )
    else:
        counter = HitCounter(
            xray_bath = xray,
            gamma_pulse = gamma,
            n_samples_azimuthal = sampling['azimuthal samples']
        )

    counter.plot_hit_count(
        min_delay = -10,
        max_delay = 500,
        samples = sampling['delay samples'],
        show_exp_value = True,
        save_data = True,
        save_params = True,
        plot_wait = 0
    )

def run_optimise_delay(
        dir_name: str,
        xray_type: str,
        spectra_type: str,
        repeat: int = 3,
        sim_params: dict = None,
        sample_params: dict = None
    ):
    """Repeats the simulation to find optimal delay
    and maximum positon yield

    Args:
        dir_name (str): directory to save the simulation data
        xray_type (str): type of xray source to use e.g.: 'lambertian'
        spectra_type (float): type of xray energy spectra to use
        repeat (int, optional): number of times to repeat the simulation. Defaults to 3.
        sim_params (dict, optional): simulation parameters. Defaults to accurate.
        sample_params (dict, optional): sampling parameters. Defaults to deep.
    """
    if dict is None:
        sim_params = accurate
    if sampling is None:
        sampling = deep

    os.makedirs(dir_name, exist_ok=True)
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)
    for i in tqdm(range(1, 1 + repeat), desc = 'Repeating simulations', leave = False):
        run_hit_counter(
            params = sim_params,
            sampling = sample_params,
            xray_type = xray_type,
            spectra_type = spectra_type
        )
        os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')
    os.rename('Simulation_parameters.csv', f'{dir_name}/Simulation_parameters.csv')

    print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

    data_sim, optimal_delay = avg_data(simdata_dir = f'{dir_name}/')
    plot_data(data_sim[:,0], data_sim[:,1], data_sim[:,2], optimal_delay)
