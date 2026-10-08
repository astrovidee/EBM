# EBM: a seasonal energy balance model for exoplanet climates

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16813585.svg)](https://doi.org/10.5281/zenodo.16813585)

A one-dimensional, seasonal energy balance model in Python for the climates of
rocky planets around F, G, K and M stars, including planets on eccentric orbits
and planets cold enough for CO2 ice to form. It is a Python version of the
MATLAB model written by Cecilia Bitz, and a modified version was used in
Venkatesan et al. (2025). In the
[FILLET intercomparison](https://doi.org/10.3847/PSJ/ae1c3c) it is called
Shields-Bitz. It solves

```
C ∂T/∂t = S(x,t)·(1 − α(T)) − OLR(T) + ∂/∂x[D·(1 − x²)·∂T/∂x]
```

for land and ocean separately at each latitude, with heat exchange between the
two, where `x = sin(lat)`. The whole model is one file, `EBM_one_file.py`.

![](EBM_new.png)

## Requirements

Python 3 with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`).

## Quick start

```python
import EBM_one_file as ebm

cfg = ebm.DEFAULTS.copy()
cfg["runlength"] = 2    # model years
cfg["star"] = "G"       # host star type: "F", "G", "K" or "M"
cfg["scaleQ"] = 1.0     # stellar flux relative to the default

results = ebm.seasonal_run(cfg)
print(ebm.mean_iceline(results))                         # ice-line latitude: 52.26
print(ebm.final_year_annual_means(results)["Tglob"])     # global mean temperature (K)
```

`python EBM_one_file.py` runs a built-in example, a sweep over stellar flux
(about two minutes).

Every setting is in `DEFAULTS` at the top of the file. The most used ones:

| Setting | Meaning | Default |
|---|---|---|
| `star` | Host star type: `"F"`, `"G"`, `"K"` or `"M"` | `"G"` |
| `scaleQ` | Stellar flux relative to `solar_constant` | 1.0 |
| `solar_constant` | Stellar flux at the planet for `scaleQ = 1` (W m⁻²) | about 1353 |
| `ecc`, `obl` | Eccentricity; obliquity in degrees | 0, 0 |
| `land` | Land configuration, such as `"modern"`, `"Fillet"` or `"Aquaplanet"` | `"modern"` |
| `coldstart` | 1 starts from a frozen planet | 0 |
| `olr` | Outgoing radiation law: `"linear"` or `"spiegel"` | `"linear"` |
| `albedo_land`, `albedo_ocean`, `albedo_ice` | Constant surface albedos; `None` uses the model's own | `None` |
| `jmx`, `runlength` | Latitude cells; model years | 120, 100 |

## Physics

- **Insolation.** Daily means for circular or eccentric orbits (Berger 1978),
  with 360 time steps per orbit.
- **Albedo.** Broadband albedos of ocean, land and ice weighted by the host
  star's spectrum (Shields et al. 2013), with a zenith-angle term on ice-free
  surfaces. A surface takes the ice albedo at or below −2 °C.
- **Outgoing radiation.** A + B·T with A = 203.3 W m⁻² and B = 2.09 W m⁻² K⁻¹
  (T in °C), or the law of Spiegel et al. (2008) with `olr="spiegel"`.
- **Heat transport.** Diffusion with D = 0.44 W m⁻² K⁻¹, stronger in the
  tropics to mimic the Hadley circulation.
- **Sea ice.** Thermodynamic sea ice with a thickness at each latitude.
- **CO2 ice.** `ebm.co2_run(cfg, start="warm")` (or `"cold"`) applies the CO2
  ice albedo of Venkatesan et al. (2025) once the global mean temperature is
  at or below 131.06 K. Set `olr="spiegel"` for this; `co2_grain` sets the
  grain size.

With `olr="spiegel"`, `co2_run` reproduces the CO2 ice thresholds of Venkatesan
et al. (2025), which were computed with the MATLAB model, for F, G, K and M
stars at eccentricities 0 and 0.5. The exception is the 2000 micron grain case
at eccentricity 0.5. Eccentricity 0.9 has not been compared.

## FILLET benchmarks and experiments

```bash
cd fillet
python run_fillet.py     # Benchmarks 1-3 and Experiments 1-4; about 25 minutes on 2 cores
python plot_fillet.py    # fillet_shields_bitz.png
```

Results are in `fillet/Results/shields_bitz/` in the layout of the
[FILLET repository](https://github.com/projectcuisines/fillet), and packed in
`fillet/shields_bitz_fillet_results.tar.gz`. Every file header states the
configuration, the ice-line definition and the convergence.

| | Benchmark 1 | Benchmarks 2-3, Experiments 1-4 |
|---|---|---|
| Land | Earth-like | 25% at every latitude |
| Surface albedo, land / ocean / ice | the model's own | 0.3 / 0.2 / 0.6 |
| Diffusion | Hadley profile, 0.44 W m⁻² K⁻¹ | constant 0.5 W m⁻² K⁻¹ |
| Heat capacity, land / ocean | model values | 1×10⁷ / 4×10⁸ J m⁻² K⁻¹ |
| Longwave constant A | tuned to 200.3 W m⁻² | 203.3 W m⁻² |

| Case | Result |
|---|---|
| Benchmark 1 (tuned) | 288.0 K; sea ice poleward of 54.8° |
| Benchmark 2 (ε = 23.5°) | 302.0 K; ice free |
| Benchmark 3 (ε = 60°) | 302.1 K; ice free |
| Experiment 1 (ice free / caps / belt / snowball) | 139 / 16 / 0 / 35 |
| Experiment 2 (ice free / caps / belt / snowball) | 149 / 0 / 5 / 36 |
| Experiment 3 | snowball up to 0.9000 S⊕; thaws from 1.1875 S⊕ |
| Experiment 4 | snowball up to 5.2 ppm; thaws from 39,100 ppm |

Departures from the protocol:

- Experiments 3 and 4 use the Benchmark 2 configuration.
- The longwave law has no CO2 term. Experiment 4 shifts A by the change in the
  Williams & Kasting (1997) fit between 280 ppm and each CO2 value.
- Sea ice has no heat capacity.
- Experiments 1a and 2a are not run, because the year is fixed at 360 steps.

The five ice belts in Experiment 2 persist in runs of 1000 orbits, and their
filed global mean temperatures are within about 0.4 K of the 1000-orbit values.

`python run_fillet.py --albedo native` repeats the runs with the model's own
albedos, into `fillet/Results_native_albedo/`. That set gives 285.0 K with sea
ice poleward of 51.6° for Benchmark 2.

## Changes since version 1.0

- The stellar flux scaling `scaleQ` was applied twice; it is now applied once.
- `mean_iceline` read the wrong latitude grid; fixed. The new `ice_edges` gives
  the extent of sea ice and of ice-covered land.
- New settings: `solar_constant`, the three surface albedos, `olr="spiegel"`,
  and CO2 ice through `co2_run`.
- The albedo's zenith-angle term follows the star's declination, as in the
  MATLAB model.
- The multi-file version is removed. The solver is about four times faster.

## Known limitations

- No atmosphere, so the surface albedo is also the planetary albedo.
- The sea ice step does not conserve energy exactly (up to 0.44 W m⁻² when sea
  ice is present).
- With the model's own albedos for a G star, snow-free land is brighter than
  ice-covered land poleward of about 59°.
- Compared with Venkatesan et al. (2025), ice lines just after a planet thaws
  are 1° to 11° lower, and an Earth-like case is at 286 K against the paper's
  293 K.

## Citation

Venkatesan, V., Shields, A. L., Deitrick, R., Wolf, E. T., & Rushby, A. (2025).
A One-Dimensional Energy Balance Model Parameterization for the Formation of
CO2 Ice on the Surfaces of Eccentric Extrasolar Planets. *Astrobiology*, 25(1),
42–59. https://doi.org/10.1089/ast.2023.0103

Software: https://doi.org/10.5281/zenodo.16813585. Citation metadata is in
`citation.cff`.

## Authors

Vidya Venkatesan (Python version). The original MATLAB model is by Cecilia
Bitz. MIT License. Questions and bug reports are welcome through GitHub issues.

## Use of AI

Version 1.1 was prepared with the help of Claude (Anthropic). A review of the
FILLET model codes, run by the FILLET project lead with Claude, found the bugs
that are fixed in this version.
