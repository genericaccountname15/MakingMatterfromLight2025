# MakingMatterfromLight2025

## Description
Simulates Gamma pulse and Xray bath collision geometry for the pair generation section of the Breit Wheeler laser-plasma platform experiment developed by B.Kettle et al. using a Monte Carlo method. Also simulates the setup in g4beamline for Bethe-Heitler positron noise investigation.

## Installation and instruction for use
**Requirements: python3, pip3**
```bash
git clone https://github.com/genericaccountname15/MakingMatterfromLight2025.git
cd MakingMatterfromLight2025
python3 module_download.py
```
Write scripts in main.py located in noise_simulation and optimisation_simulation

## Brief Module Descriptions
| Module name | Description |
| --- | ---------- |
| module_download | Script which downloads all required python modules using pip3 |
| analytical_modelling | Attempt at an analytical model for the optimisation_simulation |
| optimisation_simulation | Monte Carlo simulation for pulse timing optimisations. Docs available [here]({https://html-preview.github.io/?url=https://raw.githubusercontent.com/genericaccountname15/MakingMatterfromLight2025/main/optimisation_simulation/docs/_build/html/index.html) |
| noise_simulation | g4beamline simulation to investigate Bethe-Heitler noise |

