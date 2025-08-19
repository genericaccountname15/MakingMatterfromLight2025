"""
Analysis Package
================

This package contains the submodules for delay optimisation analysis.
Runs the Simulation over pulse delay values and measures the number of 
hits and estimates the number of positron pairs generated.
The simulation data saves to pickled dict files (.pickle).

Modules:
    - hit_counter: The default hit counter
    - hit_counter_line: Default hit counter with additional parameters added to csv
    - hit_counter_twave: Default hit counter with additional parameters added to csv
    - hit_counter_gespec: Hit counter using a different Xray spectra for number of pairs estimation

Example:
    >>> from core.gamma import Gamma
    >>> from core.xray import Xray
    >>> from analysis.hit_counter import hit_counter
    >>> xray = Xray(
    >>>     fwhm = 12,
    >>>     rotation = 0.698,
    >>>     n_samples_angular = 10,
    >>>     n_samples = 400
    >>> )
    >>> gamma = Gamma(
    >>>     x_pos = -12,
    >>>     pulse_length = 0.0135,
    >>>     height = 3.1,
    >>>     off_axis_dist = 1
    >>> )
    >>> counter = HitCounter(
    >>>     xray_bath = xray,
    >>>     gamma_pulse = gamma,
    >>>     n_samples_azimuthal = 5
    >>> )
    >>> counter.plot_hit_count(
    >>>     min_delay = -10,
    >>>     max_delay = 500,
    >>>     samples = 50,
    >>>     show_exp_value = True
    >>> )

Output:

.. image:: _static/Analysis_example.png
    :alt: Example of the hit counter plot
    :width: 600px
"""
