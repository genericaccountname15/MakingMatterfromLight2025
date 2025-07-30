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

def git_config(email: str, name: str):
    """Setup git config for pushing

    Args:
        email (string): email address of pusher
        name (name): name of pusher
    """
    subprocess.run(["git", "config", "--global", "user.email", email], check=True)
    subprocess.run(["git", "config", "--global", "user.name", name], check=True)

if __name__=='__main__':
    install_modules()
    git_config(email = 'timothy.chew22@imperial.ac.uk', 
               name = 'Timothy')