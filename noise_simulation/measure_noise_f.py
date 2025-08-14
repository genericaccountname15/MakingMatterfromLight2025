"""
Script to measure the noise profile from varying half block displacement
"""
import numpy as np
from analysis.setup_analysis import Analysis
from analysis.file_locations import files_HPC

if __name__ == '__main__':
    analysis = Analysis(file_dict = files_HPC, save_filename = 'noise_f_Det.txt')
    analysis.run_analysis(
        g4bl_filename = 'noise_simulation/g4beamlinefiles/whole_setup_electronspectra.g4bl',
        save_filedir = 'f_noise_measure_14Aug',
        variables = np.linspace(-1.5, 1.5, 10),
        variable_name = 'f',
        save_all = False,
        repeat = 3
    )
