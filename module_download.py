"""
Automatically downloads required modules for the simulation

Timothy Chew
30/07/25
"""

import subprocess

def install_modules():
    """installs python modules
    Note: HPC uses pip3
    """
    subprocess.run(["pip3", "install", "--user", "numpy"], check=True)

    subprocess.run(["pip3", "install", "--user", "matplotlib"], check=True)

    subprocess.run(["pip3", "install", "--user", "tqdm"], check=True)

    subprocess.run(["pip3", "install", "--user", "pandas"], check=True)

    subprocess.run(["pip3", "install", "--user", "scipy"], check = True)

    subprocess.run(["pip3", "install", "--user", "uproot"], check = True)

if __name__=='__main__':
    install_modules()