"""
Reads the detector output for the gamma data that comes
out of the collimator and replicates plots for sanity checks

Timothy Chew
11/8/25
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

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
        return np.linalg.norm(momentum, axis=1)
    
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

    
    def bin_det_data(self, nbins, n_samples, binning_range: list=None):
        """Bins all detector data

        Args:
            nbins (int): number of bins in each direction (x,y)
            binning_range (list[list[float]]):
                range of coordinates to bin over [[xmin, xmax], [ymin, ymax]]
                (units: mrad)

        Returns:
            tuple[list[float]]: A tuple which contains:
                - the divergence of the beam [divx, divy] (mrad)
                - the number of events
                - the x momentum [mean, std] (MeV/c)
                - the y momentum [mean, std] (MeV/c)
                - the z momentum [mean, std] (MeV/c)
        """
        self.z = 865
        div_x = np.arctan2(self.get_x(), self.get_z()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_z()) * 1000

        #count data
        n_events, _, _ = np.histogram2d(div_x, div_y, bins=nbins, range=binning_range)
        n_events = np.round(n_events * (n_samples / np.sum(n_events)), 0).astype(int)

        #bin data
        mean_p, div_x_edges, div_y_edges, _ = binned_statistic_2d(
            div_x, div_y, self.get_energy(),
            statistic = 'mean',
            bins = [nbins, nbins],
            range=binning_range
        )
        std_p, _, _, _ = binned_statistic_2d(
            div_x, div_y, self.get_energy(),
            statistic = 'std',
            bins = [nbins, nbins],
            range=binning_range
        )
        mean_px, _, _, _ = binned_statistic_2d(
            div_x, div_y, self.get_p()[:,0],
            statistic = 'mean',
            bins = [nbins, nbins],
            range=binning_range
        )
        std_px, _, _, _ = binned_statistic_2d(
            div_x, div_y, self.get_p()[:,0],
            statistic = 'std',
            bins = [nbins, nbins],
            range=binning_range
        )
        mean_py, _, _, _ = binned_statistic_2d(
            div_x, div_y, self.get_p()[:,1],
            statistic = 'mean',
            bins = [nbins, nbins],
            range=binning_range
        )
        std_py, _, _, _ = binned_statistic_2d(
            div_x, div_y, self.get_p()[:,1],
            statistic = 'std',
            bins = [nbins, nbins],
            range=binning_range
        )

        div_x_centre = [(div_x_edges[i] + div_x_edges[i+1]) / 2 for i in range(len(div_x_edges) - 1)]
        div_y_centre = [(div_y_edges[i] + div_y_edges[i+1]) / 2 for i in range(len(div_y_edges) - 1)]
        div_x_width = div_x_edges[1] - div_x_edges[0]
        div_y_width = div_y_edges[1] - div_y_edges[0]

        return (
            [div_x_centre, div_y_centre],
            [div_x_width, div_y_width],
            n_events,
            [np.nan_to_num(mean_p, nan=0.0), np.nan_to_num(std_p, nan=0.0)],
            [np.nan_to_num(mean_px, nan=0.0), np.nan_to_num(std_px, nan=0.0)],
            [np.nan_to_num(mean_py, nan=0.0), np.nan_to_num(std_py, nan=0.0)]
            )

    def gen_beam_objects(self, nbins, n_samples, z, col_pos, binning_range=None):
        div, div_err, n_events, p, px, py = self.bin_det_data(nbins, n_samples, binning_range=binning_range)

        with open("gamma_beams.g4bl", "w", encoding='utf-8') as file_handler:
            for i, div_x in enumerate(div[0]):
                for j, div_y in enumerate(div[1]):
                    #check for momentum
                    beam_string = (f"beam gaussian particle=gamma nEvents={n_events[i][j]} beamZ={col_pos} "
                                f"beamX={z * np.tan( div_x / 1000 )} beamY={z * np.tan( div_y / 1000 )} weight=100 \\\n"
                                f"sigmaX={z * np.tan(div_err[0] / 1000)} sigmaY={z * np.tan(div_err[1] / 1000)} "
                                f"sigmaXp={px[1][i][j]} sigmaYp={py[1][i][j]} \\\n"
                                f"meanMomentum={p[0][i][j]} sigmaP={p[1][i][j]} meanXp={px[0][i][j]} meanYp={py[0][i][j]} \\\n"
                                f"meanT=0.0 sigmaT=0.0 \n"
                                )
                    file_handler.write(f"{beam_string}\n\n")


    
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
    cat = GammaDetRead('profie_test_Det_norange.txt')
    # print(len(cat.get_data()))
    # cat.plot_hitmap()
    # cat.plot_spectra()
    cat.gen_beam_objects(nbins=100, n_samples=len(cat.get_data()), z=280, col_pos=475)