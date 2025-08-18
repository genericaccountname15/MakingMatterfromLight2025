"""
Energy Spectra Subpackage
=========================

This package contains the energy spectra for the Xray and Gamma photon distribution.
Contains the datafiles and methods to plot and process data.
Includes methods to sample a probability distribution from those energy spectra.

Modules:
    - gamma_spectra: gamma spectra after the collimator from the 2018 experiment
    - xray_spectra: measured Xray spectra with aluminium filter
    - xray_spectra: measured Xray spectra without aluminium filter, so uncallibrated

Example:
    >>> import matplotlib.pyplot as plt
    >>> from theory.energy_spectra.gamma_spectra import GammaSpectra
    >>> from theory.energy_spectra.xray_spectra import XraySpectra
    >>> 
    >>> gamma_data = GammaSpectra('optimisation_simulation/theory/energy_spectra/data/GammaSpectra/Fig4b_GammaSpecLineouts.mat')
    >>> xray_data = XraySpectra('optimisation_simulation/theory/energy_spectra/data/XrayBath/XraySpectra/', resolution=0.5)
    >>> gamma_data.replicate_plot()
    >>> xray_data.replicate_plot()

Output:

.. image:: _static/Gamma_spectrum_example.png
    :alt: Example plot of gamma spectrum
    :width: 600px

.. image:: _static/Xray_spectrum_example.png
    :alt: Example plot of the Xray spectrum
    :width: 600px
"""