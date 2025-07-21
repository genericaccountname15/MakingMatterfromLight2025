from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

filelist = os.listdir('data_read\\data\\XrayBath\\XraySpectra')

Energy = np.array([])
Nph = np.array([])
for file in filelist:
    if file.endswith('.pickle'):
        data = np.load(f'data_read\\data\\XrayBath\\XraySpectra\\{file}', allow_pickle=True)
        Energy = np.append(Energy, data['E'][:][0], axis = 0)
        laser_energy = data['laser_energy']
        Nph = np.append(Nph, data['normalised_number']/laser_energy/np.pi/2, axis = 0)

# data = np.load('data_read\data\XrayBath\XraySpectra\\20180409r001s133.pickle', allow_pickle=True)
# Energy = data['E'][:][0]

# Nph = data['normalised_number']
# laser_energy = data['laser_energy']

# plt.plot(Energy, Nph/np.pi/2, 'o')
# plt.xlim(1300, 1550)
# plt.ylim(0, 1e11)
# plt.show()


# Define bin edges (uniform or custom spacing)
bin_width = 0.5
bins = np.arange(Energy.min(), Energy.max() + bin_width, bin_width)

# Digitize x-values to bin indices
bin_indices = np.digitize(Energy, bins)

# Aggregate y-values into bins
binned_y = np.zeros(len(bins) - 1)
for i in range(1, len(bins)):
    in_bin = bin_indices == i
    binned_y[i - 1] = np.mean(Nph[in_bin]) # or np.mean(y[in_bin]) for averaging

# Bin centers for plotting
bin_centers = 0.5 * (bins[:-1] + bins[1:])

plt.bar(bin_centers, binned_y, width=bin_width, align='center', color='blue')
plt.xlim(1300, 1550)
plt.ylim(0, 1e11)
plt.show()