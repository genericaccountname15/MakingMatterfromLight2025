"""
Runs g4beamlines and changes setup with each iteration.
Saves and handles large quantities of data from the simulation output.

Timothy Chew
13/8/25
"""
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

class Analysis:
    """
    Class for the analysis of the g4beamline setup.

    Args:
        file_dict (dict): dictionary containing all relavent file paths.
    
    Attributes:
        file_dict (dict): dictionary containing all relavent file paths.
        pdgid_dict (dict): dictionary containing the g4beamline PDGid values.
    
    Methods:
        run_g4blsim(self, g4bl_filename: str, d=1, angle=40):
                runs the g4beamline simulation in the command line using subprocess.
        filter_particle(input_filename: str, output_filename: str, chunsize, target_particle):
               Filters the dataset with chunksizing to decrease memory requirements.
        run_analysis(g4bl_filename: str, save_filedir: str, variables: list, variable_name: str,
                    repeat = 3, save_all = True): Repeats the g4beamline simulation while
                    varying a setup parameter and saving the data.
    """
    def __init__(self, file_dict: dict, save_filename: str = None):
        self.file_dict = file_dict
        self.pdgid_dict = {
            'positron': -11,
            'electron': 11,
            'gamma': 22
        }
        if save_filename is not None:
            self.file_dict = {
                **self.get_file_dict(),
                'output fname': save_filename
                }

    def run_g4blsim(self, g4bl_filename: str, d=1, angle=40):
        """Runs the g4beamlines simulation

        Args:
            g4bl_file (str): g4beamline filename (with extension)
            d (float, optional): off axial displacement of the kapton tape (mm). Defaults to 1.
            angle (float, optional): angle of the kapton tape (degrees). Defaults to 40.
        """
        # convert to radians
        angle = angle * np.pi / 180

        # command string
        command = [
            self.get_file_dict()['g4bl path'] + 'g4bl',
            self.get_file_dict()['workspace dir'] + g4bl_filename,
            "format='ascii'",
            f'd={d}',
            f'angle={angle}',
            f'sin={np.sin(angle)}',
            f'cos={np.cos(angle)}',
            f"filename={self.get_file_dict()['output fname']}"
        ]

        status = subprocess.run(command, check=True, cwd=self.get_file_dict()['workspace dir'])
        if status.returncode != 0:
            print(f"Command failed with status {status.returncode}")

    def filter_particle(
            self,
            input_filename: str,
            output_filename: str,
            chunksize = 1000000,
            target_particle = 'positron'
            ):
        """Filters the dataset to only get the target, desired particle.
        Uses chunksizing to decrease memory requirements

        Args:
            input_filename (str): name of the file to filter
            output_filename (str): name of the file to write to
            chunksize (int, optional): number of rows to look at a time. Defaults to 1000000.
            target_particle (str, optional): target particle name. Defaults to 'positron'.
        """
        # get column names
        with open(input_filename, encoding='utf-8') as file_manager:
            preamble = file_manager.readline()
            header = file_manager.readline().strip().split()
        
        filtered_data = []
        for chunk in tqdm(pd.read_csv(
                input_filename,
                sep = r'\s+',
                names = header,
                comment = '#',
                header = None,
                skiprows = 1,
                chunksize = chunksize,
                dtype = {col: 'float32' for col in header if col != 'PDGid'} | {'PDGid': 'int32'},
            ),
            desc = 'Reading file',
            unit = 'chunk'
        ):
            filtered_chunk = chunk[chunk['PDGid'] == self.get_pdgid_dict()[target_particle]]
            filtered_data.append(filtered_chunk)

        #Combine the filtered chunks and write
        with open(output_filename, 'w', encoding='utf-8') as file_manager:
            file_manager.write(preamble + '\n')
        pd.concat(filtered_data).to_csv(output_filename, index=False, mode='a')
        

    def run_analysis(
            self,
            g4bl_filename: str,
            save_filedir: str,
            variables: list,
            variable_name: str,
            repeat = 3,
            save_all = True
        ):
        """Analyse changing values of a setup parameter on detector output.
        save_all for memory considerations.

        Args:
            g4bl_filename (str): filename of the g4beamline file to analyse
            save_filedir (str): directory of the saved files
            variables (list[float]): list of parameter values to check
            variable_name (str): name of the variable.
                    Currently only supports 'd' and 'angle'
            repeat (int, optional): Number of times to repeat the simulation. Defaults to 3.
            save_all (bool, optional): 
                - True to save the detector datafile.
                - False to filter and save only the positron data.
                Defaults to True.

        Raises:
            ValueError: If variable name is not 'd' or 'angle'
        """
        os.makedirs(save_filedir, exist_ok=True)

        for var in variables:
            os.makedirs(f'{save_filedir}/g4bl_sim_{np.round(var, 1)}', exist_ok=True)

            if os.path.exists(f'{save_filedir}/g4bl_sim_{np.round(var, 1)}'):
                print(
                    'Existing directory detected!'
                    'This operation will overwrite all files in the specified directory: '
                    f'{save_filedir}/g4bl_sim_{np.round(var, 1)}'
                    )

            for i in range(repeat):
                if variable_name == 'd':
                    self.run_g4blsim(
                        g4bl_filename = g4bl_filename,
                        d = var
                        )
                elif variable_name == 'angle':
                    self.run_g4blsim(
                        g4bl_filename = g4bl_filename,
                        angle = var
                        )
                else:
                    os.remove(f'{save_filedir}/g4bl_sim_{np.round(var, 1)}')
                    os.remove(f'{save_filedir}/g4bl_sim_{np.round(var, 1)}')
                    raise ValueError("Invalid variable name. Expected: 'd' or 'angle'")

                # check if saving all data or filter first
                if save_all:
                    os.rename(self.get_file_dict()['output fname'],
                              f'{save_filedir}/g4bl_sim_{variable_name}_{np.round(var, 1)}'
                              f'/sim_run{i+1}.txt')
                else:
                    self.filter_particle(
                        input_filename = self.get_file_dict()['output fname'],
                        output_filename = (f'{save_filedir}/g4bl_sim_'
                                           f'{variable_name}_{np.round(var, 1)}'
                                           f'/sim_run{i+1}.txt')
                    )
                    os.remove(self.get_file_dict()['output fname'])


    def get_file_dict(self) -> dict:
        """Access method for file_dict

        Returns:
            dict: dictionary containing all relavent file paths
        """
        return self.file_dict

    def get_pdgid_dict(self) -> dict:
        """Access method for pdgid_dict

        Returns:
            dict: dictionary containing the g4beamline PDGid values
        """
        return self.pdgid_dict
