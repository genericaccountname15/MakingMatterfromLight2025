"""
Modified data collection script
so its more modular and easier to use

Timothy Chew
4/8/25
"""

import os
from tqdm import tqdm
import numpy as np

from core.xray import Xray      #pylint: disable=import-error
from core.xray_lambertian import XrayLambertian #pylint: disable=import-error
from core.xray_line import XrayLine         #pylint: disable=import-error
from core.gamma import Gamma    #pylint: disable=import-error
from analysis.hit_counter import HitCounter #pylint: disable=import-error
from analysis.hit_counter_line import HitCounterLine #pylint: disable=import-error
from data_collection.data_params import deep, deep_line, accurate   #pylint: disable=import-error
from data_analysis.optimise_delay import write_data_csv #pylint: disable=import-error
from data_analysis.plot_optimise_data import plot_optimised_data    #pylint: disable=import-error

def run_hit_counter_var(
        var: dict,
        params: dict = accurate,
        sampling: dict = deep,
        xray_type: str = 'uniform'
        ):
    """Runs hit counter while changing variable var

    Args:
        var (dict): variable to vary {'name': float}
        params (dict): parameters of experiment. Defaults to accurate.
        sampling (dict): sampling parameters. Defaults to deep.
        xray_type (str, optional): _description_. Defaults to 'uniform'.

    Raises:
        ValueError: When calling for an unsupported Xray source type
    """
    varied_params = {**params, **var}
    if xray_type == 'uniform':
        xray = Xray(
            fwhm = varied_params['fwhm'],
            rotation = varied_params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )

    elif xray_type == 'lambertian':
        xray = XrayLambertian(
            fwhm = varied_params['fwhm'],
            rotation = varied_params['rotation'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle']
        )

    elif xray_type == 'line':
        if sampling == deep:
            sampling = deep_line
        xray = XrayLine(
            fwhm = varied_params['fwhm'],
            rotation = varied_params['rotation'],
            line_length=varied_params['line length'],
            n_samples_angular = sampling['angle samples'],
            n_samples = sampling['samples per angle'],
            n_line_samples = sampling['line samples']
        )

    else:
        raise ValueError(f'Invalid xray type: {xray_type}. Expected "uniform", "lambertian", or "line".')


    gamma = Gamma(
        x_pos = varied_params['x pos'],
        pulse_length = varied_params['pulse length'],
        height = varied_params['pulse height'],
        off_axis_dist = varied_params['off axis dist']
    )

    if xray_type == 'line':
        counter = HitCounterLine(
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


def repeat_sim(
        dir_name: str,
        variable_parameter_name: str,
        variable_value: float,
        xray_type: str,
        repeat: int = 3,
        sim_params: dict = accurate,
        sample_params: dict = deep
    ):
    """Repeats the simulation and saves to a folder

    Args:
        dir_name (str): Directory to save simulation data to
        variable_parameter_name (str): _description_
        variable_value (float): value of the parameter to repeat
        xray_type (str): _description_
        repeat (int, optional): _description_. Defaults to 3.
        sim_params (dict, optional): _description_. Defaults to accurate.
        sample_params (dict, optional): _description_. Defaults to deep.
    """
    for i in tqdm(range(1, 1 + repeat), desc = 'Repeating simulations', leave = False):
        var_dict = {variable_parameter_name: variable_value}
        run_hit_counter_var(
            var = var_dict,
            params = sim_params,
            sampling = sample_params,
            xray_type = xray_type
        )
        os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')
    os.rename('Simulation_parameters.csv', f'{dir_name}/Simulation_parameters.csv')

def compile_data(
        data_dir: str,
        variables: list,
        variable_name: str,
        units: str,
        old_value: float,
    ):
    """Makes a csv containing all the simulation
    data and plots the relavent data

    Args:
        data_dir (str): Where the data is being kept
        variables (list): _description_
        variable_name (str): _description_
        units (str): _description_
        old_value (float): _description_
    """
    write_data_csv(
        variable_name = f'{variable_name} ({units})',
        variable_list = variables,
        datadir = data_dir,
        csvname = f'{data_dir}/optimise_{variable_name}'
    )

    plot_optimised_data(
        filename = f'{data_dir}/optimise_{variable_name}.csv',
        variable_name = variable_name,
        xlabel = f'{variable_name} ({units})',
        old_value = old_value,
        save_fig = True,
        fig_location = f'{data_dir}'
    )


def auto_data_collection(
        variables: list,
        variable_name: str,
        variable_parameter_name: str,
        units: str,
        old_value: float,
        xray_type: str,
        repeat: int = 3,
        additional_label: str = None,
        sim_params: dict = accurate,
        sample_params: dict = deep
    ):
    """Automatic data collection script
    Loops simulation to observe changes to a defined variable

    Args:
        variables (list[float]): variable values
        variable_name (str): name of the variable being varied
        variable_parameter_name (str): specific parameter name of the variable
        units (str): units of the variable
        old_value (float): value of the variable in the 2018 experiment
        xray_type (float): type of xray source used
        repeat (int, optional): number of times to repeat the simulation. Defaults to 3.
        additional_label (str, optional): additional labels on the
            directory to store the data. Defaults to None.
        sim_params (dict, optional): simulation parameters (e.g.: fwhm, rotation).
            Defaults to accurate.
        sample_params (dict, optional): sampling parameters (e.g.: n_samples).
            Defaults to deep.
    
    Raises:
        KeyError: if variable_parameter_name isn't a valid parameter name
    """
    ### Check variable parameter name is valid
    if variable_parameter_name not in accurate:
        raise KeyError(f"'{variable_parameter_name}' not found in parameter dictionary keys: {list(accurate.keys())}")

    variable_file_name = np.around(variables, 2)
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)

    datadir = f'{variable_name}_optimisation_{xray_type}'
    if additional_label is not None:
        datadir = f'{variable_name}_optimisation_{xray_type}_{additional_label}'
    os.makedirs(datadir, exist_ok=True)

    for i, var in enumerate(tqdm(variables, desc = 'Data collection progress')):
        # directory name
        dir_name = f'{datadir}/sim_datafiles_{variable_name}_{variable_file_name[i]}_{units}'
        os.makedirs(dir_name, exist_ok=True)
        repeat_sim(
            dir_name = dir_name,
            variable_parameter_name = variable_parameter_name,
            variable_value = var,
            xray_type = xray_type,
            repeat = 3,
            sim_params = sim_params,
            sample_params = sample_params
        )

    print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

    compile_data(
        data_dir = datadir,
        variables = variables,
        variable_name = variable_name,
        units = units,
        old_value = old_value
    )