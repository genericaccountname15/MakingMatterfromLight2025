"""
hit_counter_line.py

Defines the HitCounterTwave class which inherits from the HitCounter class.
Modifies the class to add the travelling wave speed to the params file.

Timothy Chew
1/8/25
"""
from analysis.hit_counter import HitCounter     #pylint: disable=import-error

class HitCounterTwave(HitCounter):
    """Counts the number of collisions between the X-ray bath
    and Gamma pulse for a line source

    Changed methods: get_params
    """
    def get_params(self):
        params = {
            'FWHM (mm)': self.get_xray_bath().get_fwhm(),
            'Rotation (rad)': self.get_xray_bath().get_rotation(),
            'Length of Xray source (mm)': self.get_xray_bath().get_line_length(),
            'Travelling wave speed (ms^-1)': self.get_xray_bath().get_wave_speed(),
            'Number of samples on line': self.get_xray_bath().get_n_line_samples(),
            'Number of sampled angles': self.get_xray_bath().get_n_samples_angular(),
            'Number of samples per angle': self.get_n_samples(),
            'Height of gamma pulse (mm)': self.get_gamma_pulse().get_height(),
            'Length of gamma pulse (mm)': self.get_gamma_pulse().get_pulse_length(),
            'Off axial distance (mm)': self.get_gamma_pulse().get_off_axis_dist()
        }

        return params