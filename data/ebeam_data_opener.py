import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat6 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s006_Espec1')
mat7 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s007_Espec1')
mat8 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s008_Espec1')
mat9 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s009_Espec1')
mat10 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s010_Espec1')
mat11= scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s011_Espec1')
mat12 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s012_Espec1')
mat13 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s013_Espec1')
mat14 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s014_Espec1')
mat15 = scipy.io.loadmat('Brehm Sim/electron beam data/ElectronBeams/Dataset A/20180406r001s015_Espec1')
mats = [mat6,mat7,mat8,mat9,mat10,mat11,mat12,mat13,mat14,mat15]
spectrum_list=[]
for i in mats:
    spectrum_list.append(i['spectrum_integrated_lin'])
mean_spectrum_list = sum(spectrum_list)/15
mean_spectrum_list = 5*mean_spectrum_list #scaling
mean_spectrum = np.array(mean_spectrum_list)
mean_spectrum = mean_spectrum.ravel()
Y_MEVlin = np.array(mat6['Y_MEVlin'][0])
#plt.plot(Y_MEVlin, mean_spectrum)
#plt.ylabel('Charge pC/MeV')
#plt.xlabel('Energy MeV')
#plt.grid()
#plt.xlim(300,800)
#plt.show()

np.savetxt('Brehm Sim/beam_charge_data.txt', mean_spectrum)
np.savetxt('Brehm Sim/Electron_energy_data.txt',Y_MEVlin)


