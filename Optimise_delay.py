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

def plot_data(simdata_dir):
    file_list = os.listdir(simdata_dir)
    
    compiled_data = np.array([])
    delay_compiled = np.array([])
    Npos_CsI_compiled = np.array([])
    for file in file_list:
        data = np.load(simdata_dir + file, allow_pickle=True)
        delay_compiled = np.append(delay_compiled, data['delay'], axis=0)
        Npos_CsI_compiled = np.append(Npos_CsI_compiled, data['Npos_CsI'], axis=0)
    
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
    
    return np.array(data_avg)

if __name__ == '__main__':
    data_avg = plot_data(simdata_dir = 'sim_datafiles\\')
    print(data_avg[:, 0])