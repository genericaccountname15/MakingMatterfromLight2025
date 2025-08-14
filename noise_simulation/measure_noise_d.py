"""
Script to measure the noise profile from varying d
"""
import numpy as np
from analysis.setup_analysis import Analysis
from analysis.file_locations import files_HPC

if __name__ == '__main__':
    analysis = Analysis(file_dict = files_HPC, save_filename = 'noise_d_gamma_Det.txt')
    analysis.run_analysis(
        g4bl_filename = 'noise_simulation/g4beamlinefiles/whole_setup_gammaspectra.g4bl',
        save_filedir = 'd_noise_measure_14Aug',
        variables = np.linspace(0.1, 3.0, 10),
        variable_name = 'd',
        save_all = False,
        repeat = 3
    )
