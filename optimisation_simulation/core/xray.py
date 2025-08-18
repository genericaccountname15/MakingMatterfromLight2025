"""
Defines the class Xray, representing the Xray bath in the simulation.
Does so by sampling a radially-uniform, gaussian travelling profile.
This class provides attributes and methods for sampling, generating,
and moving the sampled 'Xray photons'.
Designed to be subclassed by other distribution types for Xray modelling
such as 'XrayLambertian' and 'XrayLine'
"""
import numpy as np

class Xray:
    """
    Coordinates and distribution of the X-ray bath

    Attributes:
        fwhm (float): Full-Width-Half-Maximum of the X-ray distribution (mm)
        Variance (float): Variance of the X-ray distribution (mm^2)
    """
    def __init__(
            self,
            fwhm: float,
            rotation=0,
            n_samples_angular=400,
            n_samples=10
        ):
        self.fwhm = fwhm
        self.variance = ( fwhm / 2.3548 ) ** 2 #convert from fwhm to variance
        self.rotation = rotation
        self.n_samples_angular = n_samples_angular
        self.n_samples = n_samples

        self.xray_coords = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation=rotation,
            n_samples_angular = n_samples_angular,
            n_samples = n_samples
        )

    ############ METHODS ##########################################################################
    def gen_xray_seed(self, mean, variance, **kwargs) -> list:
        """Generates distribution of X ray pulse in 2D

        Args:
            mean (float): mean of distribution, radial position (m)
            variance (float): variance of x-ray distribution (mm^2)
            **kwargs: optional keyword arguments.
                rotation (float, optional): rotation of the Xray source
                or the kapton tape angle (rad). Defaults to 0.
                n_samples_angular (int, optional): Number of angles to sample. Defaults to 400.
                n_samples (int, optional): Number of samples per angle. Defaults to 10.

        Returns:
            numpy.ndarray[list[float]]: numpy array containing:
                - coordinates for distribution points in the form [x, y, angle].
                The angle is to the x-axis in the plane of the gamma pulse's motion
        """
        # kwargs ##################################################################################
        rotation = kwargs.get('rotation', 0)
        n_samples_angular = kwargs.get('n_samples_angular', 400)
        n_samples = kwargs.get('n_samples', 10)


        coords = []
        #rotate distribution 180 degrees
        angles = np.linspace(0 + rotation, np.pi + rotation, n_samples_angular)
        for theta in angles:
            ndist = np.random.normal(mean, variance, n_samples) # random distribution centred at 0

            #rotation matrix
            rot_matrix = np.array(
                [ [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)] ])

            #append coords
            for k in ndist:
                rotated_coords = np.matmul(rot_matrix, [k, 0])
                coords.append([rotated_coords[0], rotated_coords[1], theta])

        return np.array(coords)

    def move_xrays(self, t):
        """Moves xray points

        Args:
            t (float): Time step (ps)

        Returns:
            numpy.ndarray: moved xray point coordinates
        """
        t *= 1e-9 #mm and picosecond, unit conversion
        moved_coords = []
        for r in self.get_xray_coords():
            #calculate distance to move
            dx = 3e8 * t * np.cos(r[2])
            dy = 3e8 * t * np.sin(r[2])

            moved_coords.append([r[0] + dx, r[1] + dy, r[2]])

        return np.array(moved_coords)

    def resample(self, phi): #pylint: disable=unused-argument
        """Resamples x-ray distribution
        """
        self.xray_coords = self.gen_xray_seed(
            mean = -self.get_fwhm(),
            variance = self.get_variance(),
            rotation = self.get_rotation(),
            n_samples_angular = self.get_n_samples_angular(),
            n_samples = self.get_n_samples()
        )

    def get_n_samples_total(self) -> int:
        """Get total number of samples

        Returns:
            int: total number of distribution samples
        """
        return self.get_n_samples() * self.get_n_samples_angular()

    ############ ACCESS METHODS ####################################################################
    def get_fwhm(self) -> float:
        """Access method for fwhm

        Returns:
            float: Full-Width-Half-Maximum of the X-ray distribution (mm)
        """
        return self.fwhm

    def get_variance(self) -> float:
        """Access method for variance

        Returns:
            float: Variance of the X-ray distribution (mm^2)
        """
        return self.variance

    def get_xray_coords(self) -> list:
        """Access method for Xray coordinates

        Returns:
            numpy.ndarray: Array of xray coordinates
        """
        return self.xray_coords

    def get_n_samples_angular(self) -> int:
        """Access method for n_samples_angular

        Returns:
            int: number of angles sampled
        """
        return self.n_samples_angular

    def get_n_samples(self) -> int:
        """Access method for n_samples

        Returns:
            int: number of samples per angle
        """
        return self.n_samples

    def get_rotation(self) -> float:
        """Access method for rotation

        Returns:
            float: rotation of kapton tape
        """
        return self.rotation
