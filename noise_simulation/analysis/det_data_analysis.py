"""
Runs data analysis on the detector data files and the variable varying folders.
Positron hit count data analysis, assumes file contains only positron data.

Timothy Chew
14/8/25
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_npos_list(data_dir: str) -> tuple:
    """Reads the output files and plots the resulting detector data

    Args:
        data_dir (str): directory where all the data is stored

    Returns:
        tuple[list[float]]: Tuple containing:
            - the average number of Bethe Heitler positrons detected
            - the error in the number of Bethe Heitler positrons
    """
    # read simulation runs and take average
    npos_list = []
    npos_err_list = []
    data_dir_list = os.listdir(data_dir)
    for simdata_dir in data_dir_list:
        npos = []
        det_data_list = os.listdir(f'{data_dir}/{simdata_dir}')
        for det_data in det_data_list:
            npos.append(
                len(np.loadtxt(f'{data_dir}/{simdata_dir}/{det_data}', delimiter=',', comments='#'))
                )
        npos_list.append(np.mean(npos))
        npos_err_list.append(np.std(npos))
    
    return npos_list, npos_err_list

def write_csv(variable_name, variable_list, data_dir, csvname):
    npos_list, npos_err_lis = get_npos_list(data_dir)
    csv = {
        variable_name: variable_list,
        "Number of BH positrons": npos_list,
        "Error in BH positrons": npos_err_lis
    }
    df = pd.DataFrame(csv)
    df.to_csv(f'{csvname}.csv', index=False)

def plot_noise_data(
        filename: str,
        variable_name: str,
        xlabel: str,
        ylims: tuple = None,
        save_fig = False,
        fig_location: str = None
        ):
    """Plots the optimise data with option to save the figure generated

    Args:
        filename (str): _description_
        variable_name (str): _description_
        xlabel (str): _description_
        ylims (tuple, optional): _description_. Defaults to None.
        save_fig (bool, optional): _description_. Defaults to False.
        fig_location (str, optional): _description_. Defaults to None.
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    variable = data[:,0]
    npos = data[:,1]
    npos_err = data[:,2]

    _, ax = plt.subplots()
    ax.set_title(f'Positron noise count vs {variable_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Bethe-Heitler positrons produced by the Kapton')

    ax.plot(
        variable, npos,
        '-o',
        label = 'Number of positrons',
        color = 'blue'
    )

    ax.fill_between(
        x = variable,
        y1 = npos - npos_err,
        y2 = npos + npos_err,
        label = 'Uncertainty',
        color = 'blue',
        alpha = 0.3
    )

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()

    if ylims is not None:
        ax.set_ylim(ylims)

    if save_fig:
        if fig_location is None:
            plt.savefig(f'{variable_name}_noise_fig.png')
        else:
            plt.savefig(f'{fig_location}/{variable_name}_noise_fig.png')

    plt.show()


if __name__ == '__main__':
    write_csv(
        variable_name = 'axial displacement',
        variable_list = np.linspace(0.1,3.0,10),
        data_dir = 'd_noise_measure_14Aug',
        csvname = 'd_noise'
    )
    plot_noise_data(
        filename = 'd_noise.csv',
        variable_name = 'axial displacement',
        xlabel = 'axial displacement (mm)',
        save_fig = True
    )