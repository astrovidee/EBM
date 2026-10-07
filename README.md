# EBM: a seasonal energy balance model for exoplanet climates

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://astrovidee.github.io/EBM/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16813585.svg)](https://doi.org/10.5281/zenodo.16813585)

A one-dimensional, seasonal energy balance model (EBM) in Python for studying
the climates of rocky planets around F, G, K and M stars, including planets on
eccentric orbits. It is a Python version of the EBM originally written in
MATLAB by Cecilia Bitz.

![](EBM_new.png)

## Requirements

Python 3 with NumPy, SciPy and Matplotlib.

## Installation

```bash
git clone https://github.com/astrovidee/EBM.git
cd EBM
pip install -r requirements.txt
```

## Quick start

A two-year run from Python, which takes a few seconds:

```python
import EBM_one_file as ebm

cfg = ebm.DEFAULTS.copy()
cfg["runlength"] = 2    # model years
cfg["star"] = "G"       # host star type: "F", "G", "K" or "M"
cfg["ecc"] = 0.0        # orbital eccentricity
cfg["obl"] = 0.0        # obliquity in degrees
cfg["scaleQ"] = 1.0     # stellar flux relative to the default

results = ebm.seasonal_run(cfg)
print(ebm.mean_iceline(results))   # mean ice-line latitude in degrees
```

To run the built-in example, a warm-start sweep over stellar flux that takes a
few minutes:

```bash
python EBM_one_file.py
```

It writes `scaleQ_vs_iceline.txt` and `scaleQ_vs_iceline.png`.

## What is in this repository

- `EBM_one_file.py`: the whole model in one file. Start here.
- `seasonal.py`, `seasonal_setup.py`, `seasonal_solar.py`, `albedo_seasonal.py`,
  `icebalance.py`, `get_broadband_albedo.py`, `defaults.py`, `warmstart.py`:
  the same model split into modules.
- `docs/`: the documentation site, also at https://astrovidee.github.io/EBM/

## How to cite

If you use this code, please cite the paper and the software.

Venkatesan, V., Shields, A. L., Deitrick, R., Wolf, E. T., & Rushby, A. (2025).
A One-Dimensional Energy Balance Model Parameterization for the Formation of
CO2 Ice on the Surfaces of Eccentric Extrasolar Planets. *Astrobiology*, 25(1),
42–59. https://doi.org/10.1089/ast.2023.0103 (arXiv:2501.11667)

Venkatesan, V. (2025). EBM (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.16813585

```bibtex
@article{Venkatesan2025,
  author  = {Venkatesan, Vidya and Shields, Aomawa L. and Deitrick, Russell and
             Wolf, Eric T. and Rushby, Andrew},
  title   = {A One-Dimensional Energy Balance Model Parameterization for the
             Formation of {CO$_2$} Ice on the Surfaces of Eccentric Extrasolar Planets},
  journal = {Astrobiology},
  year    = {2025},
  volume  = {25},
  number  = {1},
  pages   = {42--59},
  doi     = {10.1089/ast.2023.0103}
}
```

## License and contact

MIT License. Questions and bug reports are welcome through GitHub issues.




