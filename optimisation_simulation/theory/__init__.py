"""
Theory Package
==============

This package contains the values of the 2018 experiment, measurements
of the setup, energy spectra of the Xray and Gamma pulses, and the Breit-Wheeler
cross-section formula as a Python function.

Modules:
    - cross_section: Breit-Wheeler cross section function
    - values: values from the 2018 experiment
    - energy_spectra: Xray and Gamma energy spectra and datasets
    
Example:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from theory.cross_section import c_bw, dc_bw
    >>>  
    >>> # setup figure
    >>> fig = plt.figure(figsize=plt.figaspect(0.5))
    >>> 
    >>> ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    >>> ax1.view_init(elev=15, azim=60, roll=0)
    >>> ax1.set_title('Differential Breit-Wheeler scattering cross-section')
    >>> ax1.set_xlabel('$\\theta$(rad)')
    >>> ax1.set_ylabel('$\\sqrt{s}$(MeV)')
    >>> ax1.set_zlabel('$d\\sigma_{\\gamma\\gamma}/d\\Omega (b)$')
    >>> 
    >>> ax2 = fig.add_subplot(1, 2, 2)
    >>> ax2.set_title('Total Breit-Wheeler scattering cross-section')
    >>> ax2.set_xlabel('$\\sqrt{s}$(MeV)')
    >>> ax2.set_ylabel('$\\sigma_{\\gamma\\gamma}(b)$')
    >>> ax2.set_xlim(1e-1, 1e3)
    >>>
    >>> # generate values
    >>> theta = np.linspace(0, np.pi, 100)
    >>> root_s_differential = np.linspace(1, 5, 100)
    >>> theta, root_s_differential = np.meshgrid(theta, root_s_differential) #for array dimensions
    >>> 
    >>> # differential cross section
    >>> diff_cross_sec = dc_bw(root_s_differential, theta)
    >>> 
    >>> # total cross section
    >>> root_s = np.logspace(0, 3, 1000)
    >>> cross_sec = c_bw(root_s)
    >>> 
    >>> # plot values
    >>> ax1.plot_surface(theta, root_s_differential, diff_cross_sec, cmap = 'gnuplot')
    >>> ax2.loglog(root_s, cross_sec, color='red')

Output:

.. image:: _static/cross_section_example.png
    :alt: Example plot of the Breit-Wheeler cross section formulae
    :width: 600px
"""
