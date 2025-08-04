"""
Reads the simulation data files and plots them all
To statistically find the most optimal delay
setting

Timothy Chew
21/07/25
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import values

def avg_data(simdata_dir):
    """Averages the simulation data saved in pickle files in a specified directory

    Args:
        simdata_dir (string): Directory where the datafiles are located

    Returns:
        Tuple (array ,list): Averaged dataset [delay, npos, npos_uncertainty] ,
                            [optimal delay, uncertainty]
    """
    file_list = os.listdir(simdata_dir)

    compiled_data = np.array([])
    delay_compiled = np.array([])
    npos_csi_compiled = np.array([])
    peak_delay = []
    for file in file_list:
        if file.endswith('.pickle'):
            data = np.load(simdata_dir + file, allow_pickle=True)
            delay_compiled = np.append(delay_compiled, data['delay'], axis=0)
            npos_csi_compiled = np.append(npos_csi_compiled, data['Npos_CsI'], axis=0)
            peak_delay.append(data['delay'][np.argmax(data['Npos_CsI'])])

    compiled_data = np.column_stack((delay_compiled, npos_csi_compiled))

    #sort data
    compiled_data = compiled_data[compiled_data[:, 0].argsort()]

    # average over data
    data_avg = []
    for i in range(len(compiled_data) // len(file_list)):
        #get index bounds
        index_min = i * len(file_list)
        index_max = (i + 1) * len(file_list)

        delay = compiled_data[index_min, 0]
        npos_data = compiled_data[index_min:index_max, 1]
        npos_avg = np.mean(npos_data)
        npos_sigma = np.std(npos_data)

        data_avg.append([delay, npos_avg, npos_sigma])

    # find peak delay
    peak_delay_mean = np.mean(peak_delay)
    peak_delay_err = np.std(peak_delay)

    return np.array(data_avg), [peak_delay_mean, peak_delay_err]

def plot_data(delay, npos, npos_err, peak_delay):
    """Plots the averaged simulation data

    Args:
        delay (list): Delay between pulse and xray ignition (ps)
        npos (list): Number of positrons/pC incident on the CsI detector
        npos_err (list): Standard deviation in the number of positrons
        peak_delay (list): [Delay value which gives greatest positron yield, uncertainty]
    """

    _, ax = plt.subplots()
    ax.set_title('Simulation averaged positron count')
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Number of positrons/pC incident on CsI')

    ax.plot(
        delay, npos,
        label = 'Positrons',
        color = 'blue'
    )

    ax.fill_between(
        x = delay,
        y1 = npos - npos_err,
        y2 = npos + npos_err,
        label = 'Uncertainty',
        color = 'blue',
        alpha = 0.3
    )

    ax.axvline(x = values.delay_experiment,
        ymin = 0, ymax = 1,
        label = 'Delay used in 2018', color = 'orange')

    # peak delay plotting
    ax.set_ylim(*ax.get_ylim()) #force matplotlib to not readjust the axis
    ax.fill_betweenx(
        y = [*ax.get_ylim()],
        x1 = [peak_delay[0] - peak_delay[1], peak_delay[0] - peak_delay[1]],
        x2 = [peak_delay[0] + peak_delay[1], peak_delay[0] + peak_delay[1]],
        label = 'optimal delay',
        color = 'red',
        alpha = 0.5
    )

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()

    print(f'Optimum delay is {peak_delay[0]} +/- {peak_delay[1]} ps')

    yield_gain, yield_gain_err = find_yield_gain(delay, npos, peak_delay[0], peak_delay[1])
    print(f'Expect a yield gain of {yield_gain} +/- {yield_gain_err} % when using optimal delay')

    yield_npos, yield_npos_err = find_yield(delay, npos, peak_delay[0], peak_delay[1])
    print(
        f'Expect a yield of {yield_npos} +/- {yield_npos_err} positrons/pC when using optimal delay'
        )

    plt.show()

def find_yield_gain(delay, npos, peak_delay, peak_delay_err):
    """returns the yield gain (%) compared to the 2018 experiment

    Args:
        delay (list): pulse delay values (ps)
        npos (list): number of positrons incident on CsI/pC
        peak_delay (float): mean value of the optimal delay (ps)
        peak_delay_err (float): standard deviation in the optimal delay (ps)

    Returns:
        tuple: yield gain, yield gain error (%)
    """
    pos_exp = npos[np.argmin(abs(delay - 40))]
    pos_max = npos[np.argmin(abs(delay - peak_delay))]
    pos_max_sigma = max([npos[np.argmin(abs(delay - peak_delay + peak_delay_err))],
                      npos[np.argmin(abs(delay - peak_delay - peak_delay_err))]])

    return pos_max/ pos_exp * 100, abs(pos_max - pos_max_sigma) / pos_exp * 100

def find_yield(delay, npos, peak_delay, peak_delay_err):
    """returns the maximal positron yield

    Args:
        delay (list): pulse delay values (ps)
        npos (list): number of positrons incident on CsI/pC
        peak_delay (float): mean value of the optimal delay (ps)
        peak_delay_err (float): standard deviation in the optimal delay (ps)

    Returns:
        tuple: maximal positron yield, maximal positron yield error
    """
    pos_max = npos[np.argmin(abs(delay - peak_delay))]
    pos_max_sigma = max([npos[np.argmin(abs(delay - peak_delay + peak_delay_err))],
                      npos[np.argmin(abs(delay - peak_delay - peak_delay_err))]])

    return pos_max, abs(pos_max - pos_max_sigma)

def write_data_csv(variable_name, variable_list, datadir, csvname):
    """Auto scans data and generates a csv for it
    Gets the positron yield dependence on the data

    Args:
        variable_name (string): Name of the variable (csv column)
        variable_list (list of variables): in the variable name column
        datadir (string): name of directory containing all simulation runs data
        csvname (string): name of csv to be saved to
    """
    npos_yield_arr = []
    npos_err_yield_arr = []
    data_dir_list = os.listdir(datadir)
    for simdata_dir in data_dir_list:
        data_sim, optimal_delay = avg_data(f"{datadir}/{simdata_dir}/")
        yield_npos, yield_npos_err = find_yield(data_sim[:,0], data_sim[:,1],
                                                optimal_delay[0], optimal_delay[1])
        npos_yield_arr.append(yield_npos)
        npos_err_yield_arr.append(yield_npos_err)

    data = {
        variable_name: variable_list,
        "positron yield / pC": npos_yield_arr,
        "positron yield error / pC": npos_err_yield_arr
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{csvname}.csv', index=False)


if __name__ == '__main__':
    # data_sim, optimal_delay = avg_data(simdata_dir = 'sim_datafiles_angle_1.57_radians/')
    # plot_data(data_sim[:,0], data_sim[:,1], data_sim[:,2], optimal_delay)
    write_data_csv(
        variable_name = 'line length (mm)',
        variable_list = np.linspace(0.4, 5.0, 10),
        datadir = 'line_source_length_optimisation_line_4Aug',
        csvname = 'optimise_line_source_length'
    )
