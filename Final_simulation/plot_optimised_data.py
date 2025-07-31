"""
Plots optimisation data

Timothy Chew
30/07/25
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_optimised_data(filename, variable_name, xlabel, old_value,
                        ylims = None, save_fig = False, fig_location = None):
    """Plots the optimise data with option to save the figure generated

    Args:
        filename (_type_): _description_
        variable_name (_type_): _description_
        xlabel (_type_): _description_
        old_value (_type_): _description_
        ylims (_type_, optional): _description_. Defaults to None.
        save_fig (bool, optional): _description_. Defaults to False.
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    d = data[:,0]
    npos = data[:,1] * 2
    npos_err = data[:,2] * 2
    #due to coding error

    _, ax = plt.subplots()
    ax.set_title(f'Positron count vs {variable_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Maximum number of positrons/pC incident on CsI')

    ax.plot(
        d, npos,
        '-o',
        label = 'Positron yield',
        color = 'blue'
    )

    ax.fill_between(
        x = d,
        y1 = npos - npos_err,
        y2 = npos + npos_err,
        label = 'Uncertainty',
        color = 'blue',
        alpha = 0.3
    )

    ax.axvline(x = old_value,
        ymin = 0, ymax = 1,
        label = f'{variable_name} used in 2018', color = 'orange')

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()

    if ylims is not None:
        ax.set_ylim(ylims)

    if save_fig:
        if fig_location is None:
            plt.savefig(f'{variable_name}_optimisation_fig.png')
        else:
            plt.savefig(f'{fig_location}/{variable_name}_optimisation_fig.png')

    plt.show()

if __name__ == '__main__':
    plot_optimised_data(
        filename = 'angle_optimisation_lambert/optimise_angle.csv',
        variable_name = 'kapton tape angle',
        xlabel = 'angle (degrees)',
        old_value = 40,
        save_fig=True
    )
