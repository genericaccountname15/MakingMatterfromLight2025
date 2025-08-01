"""
simulation.py

Defines the class simulation, which simulates the interaction
between the Xray bath and Gamma pulse using a Monte Carlo method.
Performs basic simulation tasks such as predicting hits,
detecting hits, and plotting.
Designed as a subclass to more in-depth simulation tasks
such as visualisation and estimating positron pairs produced

Timothy Chew
1/8/25
"""
import numpy as np

from core.xray import Xray      #pylint: disable=import-error
from core.gamma import Gamma    #pylint: disable=import-error

class Simulation:
    """
    Simulation of X-ray bath and gamma pulse interactions (hits)

    Attributes:
        xray_bath (Xray): Xray object instance
        gamma_pulse (Gamma): Gamma object instance
    
    Methods:
        get_overlap_coords(coords: list, beam_bounds: list) -> int, numpy.ndarray:
            finds coordinates of Xrays inside the gamma pulse beam bounds
        find_hits(self, eff_height = None, eff_d = None) -> int, list[list[float]]
    """
    def __init__(
            self,
            xray_bath: Xray,
            gamma_pulse: Gamma
        ):
        self.xray_bath = xray_bath
        self.gamma_pulse = gamma_pulse

        self.n_samples_angular = xray_bath.get_n_samples_angular()
        self.n_samples = xray_bath.get_n_samples()

    ############ METHODS ###########################################################################
    def get_overlap_coords(self, coords: list, beam_bounds: list):
        """Get coordinates of overlapping points of x-ray and gamma pulses

        Args:
            coords (list): Coordinates of x-ray pulse points
            beam_bounds (list): Boundaries of gamma pulse [xmin, xmax, ymin, ymax]

        Returns:
            numpy.ndarray: Array of overlapping coordinates
        """
        overlap_coords = []
        for r in coords:
            if beam_bounds[0] <= r[0] <= beam_bounds[1]:
                if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                    overlap_coords.append([r[0], r[1]])

        return np.array(overlap_coords)

    def find_hits(self, eff_height: float = None, eff_d: float = None):
        """Calculate hits using X-ray seed coordinates
        Check future position and see if registers a hit

        Args:
            eff_height (float, optional): effective height if on an off-azimuthal position.
                Defaults to None
            eff_d (float, optional): effective off-axis displacement. Defaults to None
        
        Returns:
            tuple (float, list[list[float]]): Tuple containing:
                - number of ovelaps
                - coordinates of overlaps in the form [time, x, y, angle]
        """
        seed_coords = self.get_xray_bath().get_xray_coords()
        bounds = self.get_gamma_pulse().get_bounds()

        if eff_d is not None:
            bounds[2] = eff_d

        if eff_height is not None:
            bounds[3] = bounds[2] + eff_height

        #unit conversion back to metres (m)
        beam_bounds = np.array(bounds) * 1e-3
        coords = seed_coords * np.array([1e-3, 1e-3, 1])


        n = 0 #hit counter
        overlap_coords = [] #coordinates of hits [time, x, y, theta]

        for r in coords:
            if r[1] > beam_bounds[3]: #check if already above beam
                pass #skip calculations

            elif r[2] == 0: #check if axial, ignore if so(will have to change if moving beam around)
                pass

            #check if already in beam
            elif beam_bounds[0] <= r[0] <= beam_bounds[1]:
                if beam_bounds[2] <= r[1] <= beam_bounds[3]:
                    n += 1
                    overlap_coords.append([0, r[0], r[1], r[2]])

            else:
                #calculate time to hit max and min boundaries
                t_ymin = ( beam_bounds[2] - r[1]) / ( 3e8 * np.sin( r[2] ) )
                t_ymax = ( beam_bounds[3] - r[1]) / ( 3e8 * np.sin( r[2] ) )

                # future pulse positions
                gamma_xmin_tmin, gamma_xmax_tmin = t_ymin * 3e8 + (beam_bounds[0], beam_bounds[1])
                gamma_xmax_tmax = t_ymax * 3e8 + beam_bounds[1]

                # future x-ray point position
                rx_tmin = t_ymin * 3e8 * np.cos( r[2] ) + r[0]
                rx_tmax = t_ymax * 3e8 * np.cos( r[2] ) + r[0]

                # check inside boundary at t=tmin
                if gamma_xmin_tmin <= rx_tmin <= gamma_xmax_tmin:
                    n += 1
                    overlap_coords.append([t_ymin, rx_tmin, beam_bounds[3], r[2]])

                # if past beam, check if enters beam
                elif rx_tmin > gamma_xmax_tmin:
                    if rx_tmax < gamma_xmax_tmax:
                        n += 1
                        x = gamma_xmax_tmax - (gamma_xmax_tmax - rx_tmax)
                        t = t_ymax - (gamma_xmax_tmax - rx_tmax) / 3e8
                        y = beam_bounds[2] + 3e8 * np.sin(r[2]) * (t_ymax - t_ymin)
                        overlap_coords.append([t, x, y, r[2]])

        if len(overlap_coords) == 0:
            return 0, np.array([0,0,0,0])
        else:
            #unit conversion
            overlap_coords = np.array(overlap_coords)
            overlap_coords *= np.array([1e12, 1e3, 1e3, 1])

        return n, overlap_coords

    ############ ACCESS METHODS ####################################################################
    def get_gamma_pulse(self):
        """Access method for the gamma pulse

        Returns:
            Gamma: Gamma pulse instance
        """
        return self.gamma_pulse

    def get_xray_bath(self):
        """Access method for the x ray path

        Returns:
            Xray: Xray bath instance
        """
        return self.xray_bath

    def get_n_samples_angular(self):
        """Access method for the number of angular samples

        Returns:
            int: number of angles to sample
        """
        return self.n_samples_angular

    def get_n_samples(self):
        """Access method for number of samples

        Returns:
            int: number of points to sample per angle
        """
        return self.n_samples