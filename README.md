# MakingMatterfromLight2025

## Description
Simulates Gamma pulse and Xray bath collision geometry for the pair generation section of the Breit Wheeler laser-plasma platform experiment in [1]
using a Monte Carlo method.

## Installation
**Requirements: python3, pip3**
```bash
git clone https://github.com/genericaccountname15/MakingMatterfromLight2025.git
cd MakingMatterfromLight2025
python3 module_download.py
```

## Brief Module Descriptions
| Module name | Description |
| --- | ---------- |
| module_download | Script which downloads all required python modules using pip3 |
| spectral_data | Samples a probability density function based on the gamma and xray emmision spectra given in [1] (figures 3 and 4).|
| cross_section | Calculates Breit-Wheeler cross section using formula given by equation 3 of section 8 of [1] and checked with [2]. |
| values | Experimental parameters from experiment [1]. Data sourced from [3][4]. Also contains values of some scientific constants. |
| simulation | Defines Xray, Gamma, Simulation, and Visualiser objects |
| hit_counter | Defines the HitCounter object |
| simulation_lambertian | Defines the XrayLambertian object. Simulates an Xray distribution which follows a Lambertian distribution [5]. |
| simulation_linesource | Defines the XrayLine object. Simulates an Xray distribution originating from a line source |
| optimise_delay | Reads output .pickle files from multiple simulation runs and compiles relavent data into a .csv file |
| plot_optimised_data | Plots the .csv file |
| analysis4 | Attempt at an analytical model for the simulation |

## Directory and File descriptions
| Directory/file name | Description |
| --- | ----------|
| Final_simulation | Contains all files required for the simulation |
| .pbs | Job scripts |
| data_read | Contains the gamma and xray energy spectra |
| other folders | data from simulation runs |

## Simulation Outputs
When a simulation is run it will output the following:
- The raw simulation data, saved to a pickled python dict
- The simulation parameters (pulse dimensions, position etc.), saved to a .csv file

When a simulation is run using the run_data_collection function it will additionally output:
- How the chosen independent variable affects the positron yield, saved to a .csv file
- A plot of the results, saved to a .png file

### Simulation Raw Data
Simulation pickle file description \
Pickled object is a python dict
| key | value |
| --- | ----- |
| delay | pulse timing delay values used (ps) |
| hit_count | number of hits |
| Npos_CsI | estimated number of positrons incident on the CsI detector |
| Npos_CsI_err | error in estimated positrons |
| n_samples | number of 'Xray photons' sampled |


## Class Diagram
```mermaid
classDiagram
    Xray <|-- XrayLambertian
    Xray <|-- XrayLambertianGe
    Xray <|-- XrayLambertianPd
    XrayLambertian <|-- XrayLine
    Xray : +float FWHM
    Xray : +float rotation
    Xray : +int n_samples
    Xray : +int n_samples_angular
    Xray : +array xray_coords
    Xray : +gen_xray_seed()
    Xray : +move_xrays()
    Xray : +resample()
    Xray : +get_n_samples_total()
    class XrayLambertian {
        +gen_xray_seed()
        +resample(azimuthal angle)
        +get_n_samples_total()
    }
    class XrayLambertianGe {
        +gen_xray_seed()
        +resample(azimuthal angle)
        +get_n_samples_total()
    }
    class XrayLambertianPd {
        +gen_xray_seed()
        +resample(azimuthal angle)
        +get_n_samples_total()
    }
    class XrayLine {
        +int n_line_samples
        +int line_length
        +gen_xray_seed_line(azimuthal angle)
        +resample(azimuthal angle)
        +get_n_samples_total()
    }
```

```mermaid
classDiagram
    Simulation <|-- Visualiser
    Simulation <|-- HitCounter
    HitCounter <|-- HitCounterLine
    Simulation : +Xray xray_bath
    Simulation : +Gamma gamma_pulse
    Simulation : +get_overlap_coords(xray and gamma coordinates)
    Simulation : +find_hits(effective height and off axial distance)
    class Visualiser {
        +bool bath_vis
        +plot
    }
    class HitCounter{
        +int n_samples_azimuthal
        +count_hits(delay)
        +calc_effective_height(azimuthal angle)
        +calc_effective_d(azimuthal angle)
        +est_npairs(angle, number of samples)
        +plot_hit_count(min and max delay)
        +get_params()
        +plot_ang_dist(delays)
    }
    class HitCounterLine{
        +get_params()
    }
```

```mermaid
classDiagram
    Gamma : +float x_pos
    Gamma : +float pulse_length
    Gamma : +float height
    Gamma : +float off_axis_dist
    Gamma : +matplotlib.patches gamma_axes_obj
    Gamma : +get_bounds()
    Gamma : +move()
    Gamma : +set_x_pos()
    Gamma : +set_height()
```


## References
1) B. Kettle , D. Hollatz, et al. 2021 A laser-plasma platform for photon-photon physics URL https://arxiv.org/pdf/2106.15170
2) Robbie Watt. Monte Carlo Modelling of QED Interactions in Laser-Plasma Experiments. PhD. ; 2021
3) Dominik Hollatz. Detection of Positrons from Breit-Wheeler Pair Formation. PhD. ; 2021
4) Cary Colgan. Laser-Plasma Interactions as Tools for Studying Processes in Quantum Electrodynamics. PhD ; 2022
5) Wikipedia. Lambert's Cosine Law. URL https://en.wikipedia.org/wiki/Lambert%27s_cosine_law


