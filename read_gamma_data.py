"""
Reads the detector output for the gamma data that comes
out of the collimator and replicates plots for sanity checks

Timothy Chew
11/8/25
"""
import numpy as np
import matplotlib.pyplot as plt

class GammaDetRead:
    def __init__(self, detdatafile: str):
        self.data = np.loadtxt(detdatafile, skiprows=1)
        self.x = self.get_data()[:, 0]
        self.y = self.get_data()[:, 1]
        self.z = self.get_data()[0, 2]
        self.p = self.get_data()[:, 3:6]
        self.energy = self.get_y_energy(self.get_p())

    def filter_gammas(self):
        mask = (self.get_data()[:,7] == 22)
        self.data = self.data[mask]
    
    def get_y_energy(self, momentum):
        m_e = 0.511   #MeV/c^2
        return np.sqrt(np.linalg.norm(momentum, axis=1) ** 2 + m_e ** 2)
    
    def plot_hitmap(self):
        fig, ax = plt.subplots()
        ax.set_title('hit map of gamma beam at the interaction point')
        ax.set_xlabel('x/mrad')
        ax.set_ylabel('y/mrad')

        self.z = 865
        
        # beam divergence (mrad)
        div_x = np.arctan2(self.get_x(), self.get_z()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_z()) * 1000

        _, _, _ , mappable = ax.hist2d(div_x, div_y, bins=500, cmap='gnuplot')
        fig.colorbar(mappable, ax=ax, label='Counts')

        ax.set_ylim(-6, 4)
        ax.set_xlim(-5, 5)
        
        plt.show()
    
    def plot_spectra(self):
        _, ax = plt.subplots()
        ax.set_title('energy spectra of gamma beam at interaction point')
        ax.set_xlabel('Photon Energy (MeV)')
        ax.set_ylabel('Photons/MeV')

        counts, bin_edges = np.histogram(self.get_energy(), bins=500)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.plot(bin_centers, counts, color='blue')

        ax.set_yscale('log')
        ax.set_xlim(0, 800)

        plt.show()

    def plot_positron_spectra(self):
        _, ax= plt.subplots()
        ax.set_title('positron energy spectra')
        ax.set_xlabel('positron energy (MeV)')
        ax.set_ylabel('positrons/MeV')
        
        mask = (self.get_data()[:,7] == -11)
        
        ax.hist(self.get_energy()[mask])
        
        plt.show()

    def sample_pdf(self, n_samples: int):
        self.z = 865
        div_x = np.arctan2(self.get_x(), self.get_z()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_z()) * 1000

        prob, bin_edges_x, bin_edges_y = np.histogram2d(div_x, div_y, bins=500)
        prob /= np.sum(prob)
        bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

    def bin_statistics(self, bins):
        self.z = 865
        div_x = np.arctan2(self.get_x(), self.get_z()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_z()) * 1000

        prob, bin_edges_x, bin_edges_y = np.histogram2d(div_x, div_y, bins=bins)
        energy_list = np.zeros(prob.shape)
        energy_err_list = np.zeros(prob.shape)

        for i in range(bins - 1):
            for j in range(bins - 1):
                mask = (bin_edges_x[i] <= div_x) & (div_x <= bin_edges_x[i+1]) & \
                (bin_edges_y[j] <= div_y) & (div_y <= bin_edges_y[j+1])

                if np.any(mask):
                    energy_list[i,j] = np.mean(self.get_energy()[mask])
                    energy_err_list[i,j] = np.std(self.get_energy()[mask])
        
        plt.imshow(energy_list)
        plt.show()



    
    def get_data(self):
        return self.data
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_z(self):
        return self.z
    
    def get_p(self):
        return self.p
    
    def get_energy(self):
        return self.energy


if __name__ == '__main__':
    cat = GammaDetRead('Gamma_profile_Det_LWFA_100mil.txt')
    cat.bin_statistics(500)