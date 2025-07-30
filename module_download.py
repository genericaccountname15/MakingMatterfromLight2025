"""
Automatically downloads required modules for the simulation

Timothy Chew
30/07/25
"""

import subprocess

def install_modules():
    """installs python modules
    """
    subprocess.run('pip install numpy', check=True)

    subprocess.run('pip install matplotlib', check=True)

    subprocess.run('pip install tqdm', check=True)

    subprocess.run('pip install pandas', check=True)

def git_config(email, name):
    """Setup git config for pushing

    Args:
        email (string): email address of pusher
        name (name): name of pusher
    """
    subprocess.run(f'git config --global user.email {email}', check=True)

    subprocess.run(f'git config --global user.name {name}', check=True)

if __name__=='__main__':
    install_modules()

    email = 'timothy.chew22@imperial.ac.uk'
    name = 'Timothy'
    git_config(email, name)