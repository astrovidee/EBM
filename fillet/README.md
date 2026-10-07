# FILLET runs with the Shields-Bitz EBM

This folder holds the scripts and results for the FILLET intercomparison of
energy balance models (Protocol v1.0, Deitrick et al. 2023; Protocol v1.1,
Barnes et al. 2025). In FILLET this model is called Shields-Bitz.

- `run_fillet.py` runs the benchmarks and experiments and writes
  `Results/shields_bitz/` in the layout of the FILLET repository.
- `plot_fillet.py` reads those files and draws `fillet_shields_bitz.png`.
- `shields_bitz_fillet_results.tar.gz` is the `Results/shields_bitz/` folder,
  packed for filing.

```bash
python run_fillet.py          # everything; about 25 minutes on 2 cores
python run_fillet.py ben exp3 # only the named parts: ben, exp1, exp2, exp3, exp4
python plot_fillet.py
```


## Configuration

Every setting is written out in `run_fillet.py`; nothing relies on the model's
defaults.

| Quantity | Benchmark 1 | Benchmarks 2 and 3, Experiments 1 to 4 |
|---|---|---|
| Instellation unit | 1361 W m⁻² | 1361 W m⁻² |
| Land | Earth-like, varying smoothly with latitude (`smooth`) | 25% at every latitude (`Fillet`) |
| Diffusion | the model's Hadley profile, 0.44 W m⁻² K⁻¹ | constant 0.5 W m⁻² K⁻¹ |
| Heat capacity, land / ocean | 1.4×10⁷ / 3.1×10⁸ J m⁻² K⁻¹ (model values) | 1×10⁷ / 4×10⁸ J m⁻² K⁻¹ |
| Longwave | A + B·T, A tuned to 200.284, B = 2.09 | A = 203.3, B = 2.09 |
| Grid and run length | 120 equal-area cells, 100 orbits of 360 steps | the same |

Benchmark 1 is tuned with one parameter, the longwave constant A, to reach
288 K.

## Results

| Case | Result |
|---|---|
| Benchmark 1 | 288.00 K; sea ice poleward of 54.8° in both hemispheres |
| Benchmark 2 | 285.00 K; sea ice poleward of 51.6° in both hemispheres |
| Benchmark 3 | 288.78 K; ice free |
| Experiment 1 (ice free / caps / belt / snowball) | 105 / 37 / 3 / 45 |
| Experiment 2 (ice free / caps / belt / snowball) | 173 / 13 / 0 / 4 |
| Experiment 3 | warm start is a snowball up to 0.9125 S⊕; cold start thaws from 1.0625 S⊕; bistable width 0.15 S⊕ |
| Experiment 4 | warm start is a snowball up to 6.6 ppm; cold start thaws from 5960 ppm |

Climate states are classified from the northern sea ice edges.

## Departures from the protocol

- **Surface albedos.** The model uses stellar-weighted albedos for a G star
  (ocean 0.319, land 0.415, ice 0.514) in place of the prescribed 0.2 / 0.3 /
  0.6. They cannot be set, and they stand in for a planetary albedo because the
  model has no atmosphere. The contrast between ice and ocean is about half the
  prescribed one.
- **Zenith-angle albedo term.** The albedo of ice-free land and ocean follows
  the star's declination through the year (`zenithflag = 1`), as in the
  original MATLAB model.
- **CO2.** The longwave law has no CO2 term. In Experiment 4, A is shifted by
  the change in the Williams & Kasting (1997) OLR fit between 280 ppm and each
  CO2 value, at 288 K and 1 bar. B is unchanged, so at 280 ppm the model is the
  one used in Experiments 1 to 3. The fit is extrapolated below 10 ppm.
- **Sea ice heat capacity.** Sea ice has no heat capacity in this model. The
  protocol lists 1×10⁷ J m⁻² K⁻¹.
- **Experiments 1a and 2a are not run.** The year is fixed at 360 time steps,
  so the orbital period cannot be varied yet.

## Definitions

- **Ice edges.** The eight Protocol v1.1 values come from the model's own ice.
  A cell has sea ice when its sea ice thickness is above zero for at least half
  of the final orbit. It has ice-covered land when the land temperature is at
  or below −2 °C, where the land albedo switches to ice, for at least half of
  the final orbit. Edges lie on the boundaries between cells. An ice-free
  hemisphere is written as Max = Min = 90 (north) or −90 (south).
- **Starts.** Every case starts from the same prescribed state. A warm start is
  7.5 + 20·(1 − 2 sin²φ) °C. A cold start is 40 °C colder with 2 m of sea ice.
  No case continues from another.
- **Albedo columns.** Asurf and ATOA are the same and are unweighted time means
  over the final orbit.

## Convergence

A case whose global mean still changed by more than 0.01 K over its final
orbit was rerun for 400 orbits. That applied to 11 of 190 cases in Experiment 1
and 1 of 50 in the warm branch of Experiment 4. The largest remaining change
over a final orbit is 0.017 K (Experiment 1); in every other file it is below
0.008 K. Each file's header records these numbers and the largest difference
between hemispheres. Benchmark 1 differs between hemispheres by design, because
its land is Earth-like.
