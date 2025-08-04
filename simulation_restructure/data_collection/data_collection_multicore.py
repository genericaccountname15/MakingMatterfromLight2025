"""
Running simulation with multicore processing

Timothy Chew
4/8/25
"""
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

from data_collection.data_params import deep, accurate   #pylint: disable=import-error
from data_collection.data_collection import run_hit_counter_var #pylint: disable=import-error
from data_analysis.optimise_delay import write_data_csv #pylint: disable=import-error
from data_analysis.plot_optimise_data import plot_optimised_data    #pylint: disable=import-error

def cpu_task(args):
    dir_name, var_dict, xray_type, repeat, sim_params, sample_params = args
    for i in tqdm(range(1, 1 + repeat), desc = 'Repeating simulations', leave = False):
        run_hit_counter_var(
            var = var_dict,
            params = sim_params,
            sampling = sample_params,
            xray_type = xray_type
        )
        os.rename('Npos_plot_data.pickle', f'{dir_name}/Npos_plot_data{i}.pickle')
    os.rename('Simulation_parameters.csv', f'{dir_name}/Simulation_parameters.csv')


def run_data_collection_multicore(
        variables: list,
        variable_name: str,
        variable_parameter_name: str,
        units: str,
        old_value: float,
        xray_type: float,
        repeat: int = 3,
        additional_label: str = None,
        sim_params: dict = accurate,
        sample_params: dict = deep
    ):
    ### Check variable parameter name is valid
    if variable_parameter_name not in accurate:
        raise KeyError(f"'{variable_parameter_name}' not found in parameter dictionary keys: {list(accurate.keys())}")

    variable_file_name = np.around(variables, 2)
    print('-'*20 + 'BEGINNING DATA COLLECTION' + '-'*20)

    datadir = f'{variable_name}_optimisation_{xray_type}'
    if additional_label is not None:
        datadir = f'{variable_name}_optimisation_{xray_type}_{additional_label}'
    os.makedirs(datadir, exist_ok=True)

    args_list = [
        (
            f'{datadir}/sim_datafiles_{variable_name}_{variable_file_name[i]}_{units}',
            {variable_parameter_name: var},
            xray_type,
            repeat,
            sim_params,
            sample_params
         )
         for i, var in enumerate(variables)
    ]


    with Pool(processes=cpu_count()-1) as pool:
        list(tqdm(pool.imap_unordered(cpu_task, args_list), total=len(args_list)))
    
        print('-'*20 + 'DATA COLLECTION COMPLETE!' + '-'*20)

  # WRITING DATA TO CSV #####################################################################
    write_data_csv(
        variable_name = f'{variable_name} ({units})',
        variable_list = variables,
        datadir = datadir,
        csvname = f'{datadir}/optimise_{variable_name}'
    )

    plot_optimised_data(
        filename = f'{datadir}/optimise_{variable_name}.csv',
        variable_name = variable_name,
        xlabel = f'{variable_name} ({units})',
        old_value = old_value,
        save_fig = True,
        fig_location = f'{datadir}'
    )
