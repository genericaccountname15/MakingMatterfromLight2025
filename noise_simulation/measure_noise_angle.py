"""
Script to measure the noise profile from varying angle
"""
import numpy as np
from analysis.setup_analysis import Analysis
from analysis.file_locations import files_local

if __name__ == '__main__':
    analysis = Analysis(file_dict = files_local, save_filename = 'noise_angle_Det.txt')
    analysis.run_analysis(
        g4bl_filename = 'noise_simulation/g4beamlinefiles/whole_setup_electronspectra.g4bl',
        save_filedir = 'angle_noise_measure_13Aug',
        variables = np.linspace(0, 90, 10),
        variable_name = 'angle',
        save_all = False,
        repeat = 3
    )
