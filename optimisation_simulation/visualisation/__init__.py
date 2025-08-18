"""
Visualisation Package
=====================

This package contains the matplotlib renderer for
visualisation of the simulation.

Example:
    >>> from core.xray import Xray
    >>> from core.gamma import Gamma
    >>> from visualisation.visualisation import Visualiser
    >>> xray = Xray(fwhm = 10,
    >>>     rotation = 0, 
    >>>     n_samples = 10,
    >>>     n_samples_angular = 400
    >>> )
    >>> gamma = Gamma(x_pos = -300,
    >>>     pulse_length = 100,
    >>>     height = 50,
    >>>     off_axis_dist = 100
    >>> )
    >>> vis = Visualiser(
    >>>     xray_bath = xray,
    >>>     gamma_pulse = gamma,
    >>>     bath_vis = True
    >>> )

Output:

.. image:: _static/Visualier_example.png
    :alt: Example of the visualiser
    :width: 600px
"""
