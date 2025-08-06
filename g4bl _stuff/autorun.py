"""
Runs g4beamline simulations
Using command line and cygwin environment
Draft python script

You will need to install the cygwin environment first:
https://www.cygwin.com/

Timothy Chew
13/11/24
"""

import subprocess
import os

# Path to the Cygwin Bash executable
cygwin_bash_path = "C:/cygwin64/bin/bash.exe"

#filepath to g4beamlines setup
g4beamline_setup_path = "C:/Program Files/Muons, Inc/G4beamline/bin/g4bl-setup.sh" #filepath to g4beamlines setup file (may be different for your computer)

#filepath to g4beamlines simulation input.g4bl file (shouldn't need to change anything here)
sim_path = os.path.abspath("g4beamlinesfiles/test.g4bl")

# Command to run inside Cygwin (e.g., `ls` to list directory contents)
cygwin_command = f"source '{g4beamline_setup_path}'"
run_sim_command = f"G4bl '{sim_path}' viewer=best"

full_command = f"{cygwin_command} && {run_sim_command}"

# Use subprocess to run the command in Cygwin
try:
    result = subprocess.run(
        [cygwin_bash_path, "-l", "-c", full_command],
        capture_output=True, text=True, check=True
    )
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:", e.stderr)
