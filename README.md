# EBM: a seasonal energy balance model for exoplanet climates

[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://astrovidee.github.io/EBM/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16813585.svg)](https://doi.org/10.5281/zenodo.16813585)

A one-dimensional, seasonal energy balance model (EBM) in Python for studying
the climates of rocky planets around F, G, K and M stars, including planets on
eccentric orbits and planets cold enough for CO2 ice to form on the surface.
It is a Python version of the EBM originally written in MATLAB by Cecilia Bitz.

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
print(ebm.mean_iceline(results))   # mean ice-line latitude in degrees: 52.26
```

To run the built-in example, a warm-start sweep over stellar flux that takes
about two minutes:

```bash
python EBM_one_file.py
```

It writes `scaleQ_vs_iceline.txt` and `scaleQ_vs_iceline.png`.

## CO2 ice

The model can include the CO2 ice albedo of Venkatesan et al. (2025). Use
`co2_run` in place of `seasonal_run`, with the Spiegel outgoing radiation law:

```python
import EBM_one_file as ebm

cfg = ebm.DEFAULTS.copy()
cfg["star"] = "G"            # host star type: "F", "G", "K" or "M"
cfg["scaleQ"] = 0.05         # stellar flux relative to the default
cfg["co2_grain"] = 200       # CO2 ice grain size in microns
cfg["olr"] = "spiegel"       # needed for CO2 ice
cfg["jmx"] = 60              # latitude cells

results, has_co2 = ebm.co2_run(cfg, start="warm")   # or start="cold"
means = ebm.final_year_annual_means(results)
print("CO2 ice present:", has_co2)                               # True
print("Global mean temperature [K]:", round(means["Tglob"], 1))  # 106.3
```

This takes about 15 to 30 seconds. CO2 is taken to condense when the global
annual mean surface temperature is at or below `co2_Tcond` (131.06 K by
default, for 400 ppmv of CO2 at 1 bar). The CO2 ice albedo then replaces the
water ice albedo on every frozen surface. `start="warm"` begins from an
ice-free planet and `start="cold"` from a frozen one.

`olr="spiegel"` is required because the default linear law, A + B·T, cannot
cool below about 176 K. With the linear law `co2_run` warns and finds no CO2 ice.

## Main settings

Copy `ebm.DEFAULTS` and change what you need. The most used settings:

| Setting | Meaning | Default |
|---|---|---|
| `star` | Host star type: `"F"`, `"G"`, `"K"` or `"M"` | `"G"` |
| `scaleQ` | Stellar flux relative to the default (1353 W/m² at the planet) | 1.0 |
| `ecc` | Orbital eccentricity | 0.0 |
| `obl` | Obliquity in degrees | 0.0 |
| `land` | Land configuration, such as `"modern"` or `"Aquaplanet"` | `"modern"` |
| `coldstart` | 1 starts from a frozen planet, 0 from a warm one | 0 |
| `olr` | Outgoing radiation law: `"linear"` or `"spiegel"` | `"linear"` |
| `co2_grain` | CO2 ice grain size in microns: 1, 2, 5, 20, 100, 200 or 2000 | 200 |
| `jmx` | Number of latitude cells | 120 |
| `runlength` | Length of the run in model years | 100 |

The [documentation](https://astrovidee.github.io/EBM/) lists every setting and
compares the model with the results of Venkatesan et al. (2025).

## What is in this repository

- `EBM_one_file.py`: the whole model in one file.
- `source/`: the source of the documentation site.
- `docs/`: the built documentation site, served at https://astrovidee.github.io/EBM/

To rebuild the documentation after a change:

```bash
pip install sphinx sphinx_rtd_theme
sphinx-build -b html source docs
```

## How to cite

If you use this code, please cite the paper and the software.

Venkatesan, V., Shields, A. L., Deitrick, R., Wolf, E. T., & Rushby, A. (2025).
A One-Dimensional Energy Balance Model Parameterization for the Formation of
CO2 Ice on the Surfaces of Eccentric Extrasolar Planets. *Astrobiology*, 25(1),
42–59. https://doi.org/10.1089/ast.2023.0103 (arXiv:2501.11667)

Venkatesan, V. EBM. Zenodo. https://doi.org/10.5281/zenodo.16813585

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
