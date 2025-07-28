"""
Reads the simulation data files and plots them all
To statistically find the most optimal delay
setting

Timothy Chew
21/07/25
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import Final_simulation.values as values

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
    Npos_CsI_compiled = np.array([])
    peak_delay = []
    for file in file_list:
        if file.endswith('.pickle'):
            data = np.load(simdata_dir + file, allow_pickle=True)
            delay_compiled = np.append(delay_compiled, data['delay'], axis=0)
            Npos_CsI_compiled = np.append(Npos_CsI_compiled, data['Npos_CsI'], axis=0)
            peak_delay.append(data['delay'][np.argmax(data['Npos_CsI'])])
    
    compiled_data = np.column_stack((delay_compiled, Npos_CsI_compiled))
    
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

def plot_data(delay, Npos, Npos_err, peak_delay):
    """Plots the averaged simulation data

    Args:
        delay (list): Delay between pulse and xray ignition (ps)
        Npos (list): Number of positrons/pC incident on the CsI detector
        Npos_err (list): Standard deviation in the number of positrons
        peak_delay (list): [Delay value which gives greatest positron yield, uncertainty]
    """
    
    fig, ax = plt.subplots() #pylint: disable=unused-variable
    ax.set_title('Simulation averaged positron count')
    ax.set_xlabel('Delay (ps)')
    ax.set_ylabel('Number of positrons/pC incident on CsI')

    ax.plot(
        delay, Npos,
        label = 'Positrons',
        color = 'blue'
    )

    ax.fill_between(
        x = delay,
        y1 = Npos - Npos_err,
        y2 = Npos + Npos_err,
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

    yield_gain, yield_gain_err = find_yield_gain(delay, Npos, peak_delay[0], peak_delay[1])
    print(f'Expect a yield gain of {yield_gain} +/- {yield_gain_err} % when using optimal delay')
    
    yield_npos, yield_npos_err = find_yield(delay, Npos, peak_delay[0], peak_delay[1])
    print(f'Expect a yield of {yield_npos} +/- {yield_npos_err} positrons/pC when using optimal delay')

    plt.show()

def find_yield_gain(delay, npos, peak_delay, peak_delay_err):
    pos_exp = npos[np.argmin(abs(delay - 40))]
    pos_max = npos[np.argmin(abs(delay - peak_delay))]
    pos_max_sigma = max([npos[np.argmin(abs(delay - peak_delay + peak_delay_err))],
                      npos[np.argmin(abs(delay - peak_delay - peak_delay_err))]])

    return pos_max/ pos_exp * 100, abs(pos_max - pos_max_sigma) / pos_exp * 100

def find_yield(delay, npos, peak_delay, peak_delay_err):
    pos_max = npos[np.argmin(abs(delay - peak_delay))]
    pos_max_sigma = max([npos[np.argmin(abs(delay - peak_delay + peak_delay_err))],
                      npos[np.argmin(abs(delay - peak_delay - peak_delay_err))]])

    return pos_max, abs(pos_max - pos_max_sigma)

if __name__ == '__main__':
    data_sim, optimal_delay = avg_data(simdata_dir = 'sim_datafiles_l100\\')
    plot_data(data_sim[:,0], data_sim[:,1], data_sim[:,2], optimal_delay)
    