# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:46:52 2025

@author: archi
"""

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os

Nphotons_list_Ge = []
Nphotons_list_Pd = []
thickness_list = np.arange(0.1,6.3,0.3)
for thickness in thickness_list:
    # Variables
    targetThickness = thickness
    targetMaterial = 'Au'
    pathtog4bl = "C:/Program Files/Muons, Inc/G4beamline/bin/" # don't change 
    runfile = r"C:\Users\Timothy Chew\Desktop\UROP2025\MakingMatterfromLight2025\g4bl_stuff\archie_files\Target_investigation.g4bl"

    # Build the command string
    command = [
    pathtog4bl + "g4bl",
    runfile,
    f"targetThickness={targetThickness}",
    f"targetMaterial={targetMaterial}",
    ]

    # Run the command
    status = subprocess.run(command)
    if status.returncode != 0:
        print(f"Command failed with status {status.returncode}")

    # Load data from Det.txt
    det = np.loadtxt("vd2_coll.txt")
    
    # Column 7 in MATLAB is index 6 in Python (0-based)
    particle_ids = det[:, 7]
    print("Unique particle IDs in data:", np.unique(particle_ids))


    # Find indices for each particle type
    gamma_i = np.where(particle_ids == 22)[0]
    electron_i = np.where(particle_ids == 11)[0]
    positron_i = np.where(particle_ids == -11)[0]

    # Create arrays for each particle type
    gammas = det[gamma_i]
    electrons = det[electron_i]
    positrons = det[positron_i]

    #Define gamma momentums/energies and add to array
    gamma_Px = gammas[:,3]
    gamma_Py = gammas[:,4]
    gamma_Pz = gammas[:,5]
    gamma_P = np.sqrt(gamma_Px**2 + gamma_Py**2 + gamma_Pz**2)
    gamma_P = gamma_P.reshape(-1,1)
    gamma_w_P = np.hstack((gammas,gamma_P))

    #Functions for obtaining number of acceptable photons for each X-ray source
    def acceptable_Ge_photons(x, column_index=12, target_value=373):
        x = np.asarray(x)
        mask = (x[:,column_index] >= target_value)
        x = x[mask]
        return len(x[:,column_index])
    
    def acceptable_Pd_photons(x, column_index=12, target_value=187):
        x = np.asarray(x)
        mask = (x[:,column_index] >= target_value)
        x = x[mask]
        return len(x[:,column_index])
    
    # Map PDG IDs to names and colors (add more as needed)
    particle_info = {
    22: ("Gamma", "g"),
    11: ("Electron", "r"),
    -11: ("Positron", "b"),
    -13: ("Muon+", "m"),
    13: ("Muon-", "c"),
    -14: ("Muon anti-neutrino", "y"),
    12: ("Electron neutrino", "k"),
    # Add more if you want
    }

    unique_ids = np.unique(particle_ids)

    for pid in unique_ids:
    # Extract particles of this type
        indices = np.where(particle_ids == pid)[0]
        particles = det[indices]

    name, color = particle_info.get(
    pid, (f"PID {pid}", "k")
    ) # default to black if unknown

    #Find fraction of acceptable photons and append
    frac_Ge = acceptable_Ge_photons(gamma_w_P)
    frac_Pd = acceptable_Pd_photons(gamma_w_P)
    Nphotons_list_Ge.append(frac_Ge)
    Nphotons_list_Pd.append(frac_Pd)

#Ensure arrays and save
Nphotons_list_Ge=np.array(Nphotons_list_Ge)
Nphotons_list_Pd=np.array(Nphotons_list_Pd)
thickness_list=np.array(thickness_list)
np.savetxt('Brehm_Sim/Nphotons_Au_acceptable_Ge.txt', Nphotons_list_Ge)
np.savetxt('Brehm_Sim/Nphotons_Au_acceptable_Pd.txt', Nphotons_list_Pd)
np.savetxt('Brehm_Sim/thickness_list.txt', thickness_list)

#Plot
plt.title('Signal as a function of Target Thickness')
opt_thickness_Ge = thickness_list[np.argmax(Nphotons_list_Ge)]
opt_thickness_Pd = thickness_list[np.argmax(Nphotons_list_Pd)]
opt_thickness_Ge = f"{opt_thickness_Ge:.3g}"
opt_thickness_Pd = f"{opt_thickness_Pd:.3g}"
max_photons_Ge = Nphotons_list_Ge[np.argmax(Nphotons_list_Ge)]
max_photons_Pd = Nphotons_list_Pd[np.argmax(Nphotons_list_Pd)]
plt.scatter(thickness_list, Nphotons_list_Ge, color='yellow',label=f'Ge: {opt_thickness_Ge}mm')
plt.scatter(thickness_list, Nphotons_list_Pd, color='orange',label=f'Pd: {opt_thickness_Pd}mm')
plt.xlabel(r'Au Thickness (mm)')
plt.ylabel('Acceptable Photons (A.U.)')
plt.legend()
plt.grid()
plt.show()
