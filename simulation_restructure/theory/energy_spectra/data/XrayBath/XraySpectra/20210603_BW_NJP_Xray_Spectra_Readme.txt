X-ray spectra pickle file description

Pickled object is a python dict

key:                                value:
-------------------------------------------------------------------------------
gsn                                 Gemini shot number from ecat2
laser_energy                        pre-compressor laser energy from ecat2
date_info                           (date, run, shot number)
col_id                              array of column numbers for valid pixels across image
E                                   array of central X-ray energy of column, and array of uncertainty on this value
bin_widths                          gradient in central X-ray energies

normalised_number                   array of number of X-ray photons per sphere per eV
normalised_number_sem               uncertainty on normalised_number

normalised_number_rearside          array of normalised_number assuming cold kapton transmission from CXRO
normalised_number_sem_rearside      uncertainty on normalised_number_rearside