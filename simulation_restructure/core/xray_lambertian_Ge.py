"""
xray_lambertian.py

Defines a Lambertian Xray source Characteristic of a
Germanium source.
cos0.4 x (Cary Colgan PhD)

Timothy Chew
4/8/25
"""
import numpy as np

from core.xray import Xray      #pylint: disable=import-error

class XrayLambertianGe(Xray):
    """Coordinates and distribution of the X-ray bath
    Inherits from the Xray bath class
    Uses a Germanium lambertian distribution for points instead

    Overridden methods:
        gen_xray_seed(mean: float, variance: float, **kwargs: float) -> numpy.ndarray:
            Generates Xray coordinates sampled from a lambertian, gaussian profile.
        get_n_samples_total() -> int: Returns the total number of Xray coordinates generated
            appropriate for a lambertian distribution.
    """
    def __init__(
            self,
            fwhm: float,
            rotation=0,
            n_samples_angular=400,
            n_samples=10
        ):
        super().__init__(fwhm, rotation, n_samples_angular, n_samples)

        self.xray_coords, self.n_samples_lambert = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples,
            get_n_lambert = True
        )

    def gen_xray_seed(
            self,
            mean: float,
            variance: float,
            **kwargs
        ):
        """Generates a lambertian distribution of X ray pulse in 2D

        Args:
            mean (float): mean of distribution, radial position (m)
            variance (float): variance of x-ray distribution (mm^2)
            **kwargs: optional
                rotation (float, optional): rotation of the Xray source
                    or the kapton tape angle (rad). Defaults to 0
                n_samples_angular (int, optional): Number of angles to sample. Defaults to 400
                n_samples (int, optional): Number of samples per angle. Defaults to 10.
                get_n_lambert (bool, optional): If True, returns the number of generated
                    samples along with the coordinates. Defaults to False.

        Returns:
            np.ndarray[list[float]]: list of coordinates for distribution points
            int (conditional): number of samples taken
        """
        # kwargs ##################################################################################
        rotation = kwargs.get('rotation', 0)
        n_samples_angular = kwargs.get('n_samples_angular', 400)
        n_samples = kwargs.get('n_samples', 10)
        get_n_lambert = kwargs.get('get_n_lambert', False)


        coords = []
        n_lambert = 0
        #rotate distribution 180 degrees
        angles = np.linspace(0 + rotation, np.pi + rotation, n_samples_angular)
        for theta in angles:
            # estimated lambert distribution
            lambert_samples = round(n_samples * np.cos( np.pi/2 - theta + rotation ) ** 0.59)
            # random distribution centred at 0
            ndist = np.random.normal(mean, variance, lambert_samples)

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                coords.append([rotated_coords[0], rotated_coords[1], theta])
            n_lambert += lambert_samples

        if get_n_lambert:
            return np.array(coords), n_lambert

        return np.array(coords)

    def resample(self, phi=0):
        """Resamples xray distribution depending
        on azimuthal angle

        Args:
            phi (float): _description_
        """
        if phi == 0:
            n_samples = self.get_n_samples()
        else:
            n_samples = round( self.get_n_samples() * np.cos( phi ) ** 0.59 )

        self.xray_coords, self.n_samples_lambert = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation = self.get_rotation(),
            n_samples_angular = self.get_n_samples_angular(),
            n_samples = n_samples,
            get_n_lambert = True
        )

    def get_n_samples_total(self) -> int:
        """Access method for n_samples_total

        Returns:
            int: total number of xray coordinates generated
        """
        return self.n_samples_lambert
