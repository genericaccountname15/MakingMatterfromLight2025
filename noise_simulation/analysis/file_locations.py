"""
File locations for g4beamlines and the workspace directory
along with the default output filename for the virtual detector

Timothy Chew
13/8/25
"""
#for home directory in HPC (follow plasmawiki instructions for g4beamline installation)
files_HPC = {
    'g4bl path': '/rds/general/user/ttc22/home/G4beamline-3.08/bin/',
    'workspace dir': '/rds/general/user/ttc22/home/MakingMatterfromLight2025/',
    'output fname': 'noise_measure_Det.txt'
}

#locally run files
files_local = {
    **files_HPC,
    'g4bl path': 'C:/Program Files/Muons, Inc/G4beamline/bin/',
    'workspace dir': 'C:/Users/Timothy Chew/Desktop/UROP2025/MakingMatterfromLight2025/',
}
