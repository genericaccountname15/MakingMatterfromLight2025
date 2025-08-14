"""
gen_gamma_beam.py

Defines the GammaDetRead class.

Reads the g4beamline virtual detector output for the gamma beam generated
from the LWFA electrons and generates a set of g4beamline gamma beam objects
to replicate the spectra.
To be placed after the collimator to ensure higher simulation efficiency.
Also replicates plots from the 2018 paper.

Timothy Chew
13/8/25
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

class GammaDetRead:
    """
    The GammaDetRead class. Reads the g4beamline virtual detector
    data and generation of g4beamline.beam objects.
    NOTE: PLEASE do NOT use this script for large sets of detector data unless
        you have enough memory to load the entire data file.

    Args:
        detdatafile (str):
        det_pos (float): distance between the virtual detector and LWFA bremsstrahlung converter
    """
    def __init__(self, detdatafile: str, det_pos: float):
        self.data = np.loadtxt(detdatafile, skiprows=2)
        self.x = self.get_data()[:, 0]
        self.y = self.get_data()[:, 1]
        self.det_pos = det_pos
        self.p = self.get_data()[:, 3:6]
        self.energy = np.linalg.norm(self.p, axis=1)

    # METHODS #####################################################################################
    def filter_gammas(self):
        """
        Filters out all particles except gamma photons from the dataset and updates
        the self.data attribute.
        """
        mask = (self.get_data()[:,7] == 22)
        self.data = self.data[mask]

    def plot_hitmap(self):
        """Plots the hitmap on the virtual detector, scaling for beam divergence (mrad)
        """
        fig, ax = plt.subplots()
        ax.set_title('hit map of gamma beam at the interaction point')
        ax.set_xlabel('x/mrad')
        ax.set_ylabel('y/mrad')

        # beam divergence (mrad)
        div_x = np.arctan2(self.get_x(), self.get_det_pos()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_det_pos()) * 1000

        _, _, _ , mappable = ax.hist2d(div_x, div_y, bins=500, cmap='gnuplot')
        fig.colorbar(mappable, ax=ax, label='Counts')

        ax.set_ylim(-6, 4)
        ax.set_xlim(-5, 5)

        plt.show()

    def plot_spectra(self):
        """Plots the energy spectra of the gamma beam with a logarithmic yscale
        """
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
        """Plots the resulting positron spectra
        """
        _, ax= plt.subplots()
        ax.set_title('positron energy spectra')
        ax.set_xlabel('positron energy (MeV)')
        ax.set_ylabel('positrons/MeV')

        mask = (self.get_data()[:,7] == -11)

        ax.hist(self.get_energy()[mask], bins=100)

        plt.show()

    def bin_det_data(self, nbins: int, n_samples: int, binning_range: list=None) -> tuple:
        """Bins all detector data and takes relevant statistics within each bin

        Args:
            nbins (int): number of bins in each direction (x,y)
            n_samples(int): number of total events to generate
            binning_range (list[list[float]]):
                range of coordinates to bin over [[xmin, xmax], [ymin, ymax]]
                (units: mrad)

        Returns:
            tuple[list[float]]: A tuple which contains:
                - the divergence of the beam [divx, divy] (mrad)
                - width in the divergence bins [divx err, divy err] (mrad)
                - the number of events
                - the total momentum/energy [mean, std] (MeV/c)
        """
        div_x = np.arctan2(self.get_x(), self.get_det_pos()) * 1000
        div_y = np.arctan2(self.get_y(), self.get_det_pos()) * 1000

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

        div_x_centre = [(div_x_edges[i] + div_x_edges[i+1]) / 2
                        for i in range(len(div_x_edges) - 1)]
        div_y_centre = [(div_y_edges[i] + div_y_edges[i+1]) / 2
                        for i in range(len(div_y_edges) - 1)]
        div_x_width = div_x_edges[1] - div_x_edges[0]
        div_y_width = div_y_edges[1] - div_y_edges[0]

        return (
            [div_x_centre, div_y_centre],
            [div_x_width, div_y_width],
            n_events,
            [np.nan_to_num(mean_p, nan=0.0), np.nan_to_num(std_p, nan=0.0)],
            )

    def gen_beam_objects(
            self,
            nbins: int,
            n_samples: int,
            dist_source: float,
            binning_range: list=None,
            gen_beams_filename = 'gamma_beams.g4bl'
            ):
        """Generates a set of g4beamline gamma gaussian beam objects
        Uses energy spectrum of a gaussian pulse generated by the LWFA conversion
        Write the beam objects to a .g4bl file.

        Args:
            nbins (int): number of bins to bin the data
            n_samples (_type_): number of total events
            dist_source (_type_): distance from the bismuth source to the gamma source we're placing
            beam_pos (_type_): distance from the origin to the gamma source we're placing
            binning_range (list[list[float]], optional): Range of divergences to bin over (mrad).
                Of the form [[xmin, xmax], [ymin, ymax]]. Defaults to None.
        """
        div, div_err, n_events, p = self.bin_det_data(nbins, n_samples, binning_range=binning_range)
        beam_pos = dist_source + self.get_det_pos()

        with open(gen_beams_filename, "w", encoding='utf-8') as file_handler:
            for i, div_x in enumerate(div[0]):
                for j, div_y in enumerate(div[1]):
                    #check n_events
                    if n_events[i][j] != 0:
                        beam_string = (f"beam gaussian particle=gamma nEvents={n_events[i][j]} "
                                    f"beamZ={beam_pos} "
                                    f"beamX={dist_source * np.tan( div_x / 1000 )} "
                                    f"beamY={dist_source * np.tan( div_y / 1000 )} weight=100 \\\n"
                                    f"sigmaX={dist_source * np.tan(div_err[0] / 1000)} "
                                    f"sigmaY={dist_source * np.tan(div_err[1] / 1000)} "
                                    f"sigmaXp={np.tan(div_err[0] / 1000)} sigmaYp={np.tan(div_err[1] / 1000)} \\\n"
                                    f"meanMomentum={p[0][i][j]} sigmaP={p[1][i][j]} "
                                    f"meanXp={np.tan(div_x / 1000)} meanYp={np.tan(div_y / 1000)} \\\n"
                                    f"meanT=0.0 sigmaT=0.0 \n"
                                    )
                        file_handler.write(f"{beam_string}\n\n")

    # ACCESS METHODS ##############################################################################
    def get_data(self) -> list:
        """Access method for data

        Returns:
            np.ndarray[float]: array of the detector data
        """
        return self.data

    def get_x(self) -> list:
        """Access method for x

        Returns:
            list[float]: x positions of the hits
        """
        return self.x

    def get_y(self) -> list:
        """Access method for y

        Returns:
            list[float]: y positions of the hits
        """
        return self.y

    def get_det_pos(self) -> float:
        """Access method for det_pos

        Returns:
            float: distance of virtual detector from the collimator
        """
        return self.det_pos

    def get_p(self) -> list:
        """Access method for p

        Returns:
            list[list[float]]: three-momenta of the hits
        """
        return self.p

    def get_energy(self) -> list:
        """Access method for energy

        Returns:
            list[float]: energies of the hits
        """
        return self.energy

if __name__ == '__main__':
    gbeams = GammaDetRead(
        detdatafile = 'noise_simulation/g4beamlinefiles/gamma_profile_test.txt',
        det_pos = 1060-195
    )
    gbeams.plot_hitmap()
    gbeams.plot_spectra()

    # gbeams = GammaDetRead(
    #     detdatafile = 'noise_simulation/g4beamlinefiles/gamma_spec_LWFA_100mil.txt',
    #     det_pos = 1060-195
    # )
    # gbeams.gen_beam_objects(
    #     nbins=100,
    #     n_samples=len(gbeams.get_data()),
    #     dist_source=195,
    #     binning_range=None,
    #     gen_beams_filename='gamma_beams_new.g4bl'
    #     )
