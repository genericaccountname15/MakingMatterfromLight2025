"""
Runs g4beamlines and changes setup
with each iteration, and prints the analysed
detector data.

Timothy Chew
11/8/25
"""

import os
import subprocess
import numpy as np
import pandas as pd

# DICTS #######################################################
files = {
    'g4bl path': 'C:/Program Files/Muons, Inc/G4beamline/bin/',
    'workspace dir': 'C:/Users/Timothy Chew/Desktop/UROP2025/MakingMatterfromLight2025/',
    'output fname': 'noise_measure_Det.txt'
}


g4bl_variables = {
    'off axial dist': 'd'
}

# FUNCTIONS ######################################################
def run_d_analysis(
        g4bl_file: str,
        save_dir: str,
        repeat = 3,
        save_all=True):
    """Analyse changing values of d on detector output
    save_all is for memory considerations

    Args:
        g4bl_file (str): _description_
        save_dir (str): _description_
        repeat (int, optional): _description_. Defaults to 3.
        save_all (bool, optional): True to save all data, False to only save positron data. Defaults to True.
    """
    d_list = np.linspace(0.1, 3.0, 10)
    os.makedirs(save_dir, exist_ok=True)

    for d in d_list:
        os.makedirs(f'{save_dir}/g4bl_sim_d_{np.round(d, 2)}')
        for i in range(repeat):
            run_g4blsim(g4bl_file=g4bl_file, d=d)

            if save_all:
                os.rename(files['output fname'],
                        f'{save_dir}/g4bl_sim_d_{np.round(d, 2)}/Det{i+1}.txt')
                
            else:
                df = pd.read_csv(files['output fname'], sep="\s+", header=1)
                filtered_df = df[df['PDGid'] == -11]
                filtered_df.to_csv(f'{save_dir}/g4bl_sim_d_{np.round(d, 2)}/Det{i+1}.txt', index=False)
    
    if not save_all:
        os.remove(files['output fname'])
            


def run_g4blsim(g4bl_file: str, d=1, theta=40):
    """Runs the g4beamlines simulation

    Args:
        g4bl_file (str): _description_
        d (int, optional): _description_. Defaults to 1.
        theta (int, optional): _description_. Defaults to 40.
    """
    # command string
    theta = theta * np.pi / 180
    command = [
        files['g4bl path'] + 'g4bl',
        files['workspace dir'] + g4bl_file,
        f'd={d}',
        f'sin={np.sin(theta)}',
        f'cos={np.cos(theta)}'
    ]

    subprocess.run(command, check=True)


if __name__ == '__main__':
    run_d_analysis('g4bl_stuff/g4beamlinesfiles/setup_redo.g4bl',
                   'test', save_all=False)