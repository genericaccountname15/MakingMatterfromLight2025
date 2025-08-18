"""
Defines the XrayTwave class which inherits from the XrayLine class.
Represents a line source's Xray emissions, done so by placing multiple
point-Lambertian sources along its length.
And the sources are ignited by a travelling wave.
"""
import numpy as np

from core.xray_line import XrayLine     #pylint: disable=import-error

class XrayTwave(XrayLine):
    """
    Generates a line source by setting up an array of lambertian point sources which are
    ignited by a travelling wave. This class extends 'Xrayline'

    Attributes:
        wave_speed (float, optional): speed of the travelling wave igniting the xray source
                (natural units). Defaults to 1.

    Overridden Methods:
        gen_xray_seed_line(azimuthal_angle: float) -> tuple[list, int(conditional)]:
            Initializes the Xray coordinates by seeding each point source along the line
            and displacing their mean to for a travelling wave with a specified velocity.
    """
    def __init__(self, fwhm, line_length, rotation=0, n_samples_angular=400, n_samples=10, n_line_samples=10, wave_speed = 1):
        self.wave_speed = wave_speed
        super().__init__(fwhm, line_length, rotation, n_samples_angular, n_samples, n_line_samples)

    def gen_xray_seed_line(self, phi = 0, get_total_samples=False) -> tuple:
        """Generates xray coordinates for a line source generated
        by a travelling wave source

        Returns:
            tuple[list, int]: A tuple containing
                - list: array of line-source-generated xray coordinates
                - int(conditional): the total number of xray points
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
                mean = -self.get_fwhm() - (
                    self.get_line_length() * (self.get_n_line_samples() - i) / self.get_n_line_samples() * 1 / self.get_wave_speed() ),
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
        return coords
    
    def get_wave_speed(self):
        """Access method for wave_speed

        Returns:
            float: travelling wave velocity (c)
        """
        return self.wave_speed