EBM documentation
=================

EBM is a one-dimensional, seasonal energy balance model in Python for studying
the climates of rocky planets around F, G, K and M stars. It handles eccentric
orbits and planets cold enough for CO2 ice to form on the surface.

The model follows North and Coakley (1979). It tracks separate land and ocean
temperatures at each latitude, moves heat between latitudes by diffusion, and
includes a simple sea ice model. It is a Python version of the MATLAB model
written by Cecilia Bitz, with the stellar type dependent albedos and the CO2
ice treatment of `Venkatesan et al. (2025) <https://doi.org/10.1089/ast.2023.0103>`_.

The whole model is one file, ``EBM_one_file.py``, available at
https://github.com/astrovidee/EBM.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   settings
   co2
   validation
   api

How to cite
-----------

If you use this code, please cite the paper and the software.

Venkatesan, V., Shields, A. L., Deitrick, R., Wolf, E. T., & Rushby, A. (2025).
A One-Dimensional Energy Balance Model Parameterization for the Formation of
CO2 Ice on the Surfaces of Eccentric Extrasolar Planets. *Astrobiology*, 25(1),
42–59. https://doi.org/10.1089/ast.2023.0103

Venkatesan, V. EBM. Zenodo. https://doi.org/10.5281/zenodo.16813585
