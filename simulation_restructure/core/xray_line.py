"""
xrayline.py

Defines the XrayLine class which inherits from the XrayLambertian class.
Represents a line source's Xray emissions, done so by placing multiple
point-Lambertian sources along its length.

Timothy Chew
1/8/25
"""
import numpy as np

from core.xray_lambertian import XrayLambertian     #pylint: disable=import-error

class XrayLine(XrayLambertian):
    """Generates a line source by setting up an
    array of lambertian point sources

    This class extends 'XrayLambertian'
    Attributes:
        line_length (float): Total physical length of the X-ray line source (mm).
        n_line_samples (int): Number of discrete Lambertian emitters along the line.
        xray_coords (np.ndarray): 2D array of sampled ray coordinates (overridden).
        n_samples_total (int): Total number of Xray photons generated (overridden).
    
    Overridden Methods:
        - resample(azimuthal_angle: float): Resamples Xrays based on azimuthal orientation.
        - get_n_samples_total() -> int: Returns total number of Xrays photons generated.

    New Methods:
        - gen_xray_seed_line(azimuthal_angle: float) -> np.ndarray, int:
            Initializes the Xray coordinates by seeding each point source along the line.
    """
    def __init__(
            self,
            fwhm: float,
            line_length: float,
            rotation=0,
            n_samples_angular=400,
            n_samples=10,
            n_line_samples=10
        ):
        super().__init__(fwhm, rotation, n_samples_angular, n_samples)
        self.n_line_samples = n_line_samples
        self.line_length = line_length

        self.xray_coords, self.n_samples_total = self.gen_xray_seed_line( get_total_samples = True )

    def gen_xray_seed_line(self, phi = 0, get_total_samples=False):
        """Generates xray coordinates for a line source

        Returns:
            np.ndarray: array of line source generated xray coordinates
        """
        coords = []
        n_samples_total = 0

        if phi == 0:
            n_samples = self.get_n_samples()
        else:
            n_samples = round( self.get_n_samples() * np.cos( phi ) )


        for i in range(self.get_n_line_samples()):
            # generate point source coordinates
            gen_coords, n_samples_lambert = self.gen_xray_seed(
                mean = -self.get_fwhm(),
                variance = self.get_variance(),
                rotation=self.get_rotation(),
                n_samples_angular = self.get_n_samples_angular(),
                n_samples = n_samples,
                get_n_lambert = True
            )

            #shift point source coordinates
            shift = self.get_line_length() / self.get_n_line_samples() * i
            gen_coords[:, 0] -= shift * np.cos(self.get_rotation())
            gen_coords[:, 1] -= shift * np.sin(self.get_rotation())

            # append to coords
            if len(coords) == 0:
                coords = np.array(gen_coords)
            else:
                coords = np.append(coords, gen_coords, axis=0)

            n_samples_total += n_samples_lambert

        if get_total_samples:
            return coords, n_samples_total
        else:
            return coords

    def resample(self, phi=None):
        """Resamples x-ray distribution
        """
        if phi is None:
            self.xray_coords = self.gen_xray_seed_line(phi=0)
        else:
            self.xray_coords = self.gen_xray_seed_line(phi)

    def get_n_samples_total(self):
        return self.n_samples_total


    def get_n_line_samples(self):
        """Access method for number of line samples

        Returns:
            int: number of line samples
        """
        return self.n_line_samples

    def get_line_length(self):
        """Access method for line_length

        Returns:
            float: length of line source
        """
        return self.line_length