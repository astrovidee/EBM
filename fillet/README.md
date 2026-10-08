# FILLET runs with the Shields-Bitz EBM

This folder holds the scripts and results for the FILLET intercomparison of
energy balance models (Protocol v1.0, Deitrick et al. 2023; Protocol v1.1,
Barnes et al. 2025). In FILLET this model is called Shields-Bitz.

- `run_fillet.py` runs the benchmarks and experiments and writes
  `Results/shields_bitz/` in the layout of the FILLET repository.
- `plot_fillet.py` reads those files and draws `fillet_shields_bitz.png`.
- `shields_bitz_fillet_results.tar.gz` is the `Results/shields_bitz/` folder,
  packed for filing.
- `Results_native_albedo/` and `fillet_shields_bitz_native_albedo.png` are the
  same runs with the model's own albedos (see below).

```bash
python run_fillet.py          # everything; about 25 minutes on 2 cores
python run_fillet.py ben exp3 # only the named parts: ben, exp1, exp2, exp3, exp4
python plot_fillet.py

python run_fillet.py --albedo native    # the variant with the model's own albedos
python plot_fillet.py --albedo native
```


## Configuration

Every setting is written out in `run_fillet.py`; nothing relies on the model's
defaults.

| Quantity | Benchmark 1 | Benchmarks 2 and 3, Experiments 1 to 4 |
|---|---|---|
| Instellation unit | 1361 W m⁻² | 1361 W m⁻² |
| Land | Earth-like, varying smoothly with latitude (`smooth`) | 25% at every latitude (`Fillet`) |
| Surface albedo, land / ocean / ice | the model's own values for a G star, with its zenith-angle term | 0.3 / 0.2 / 0.6, constant (protocol Table 4) |
| Diffusion | the model's Hadley profile, 0.44 W m⁻² K⁻¹ | constant 0.5 W m⁻² K⁻¹ |
| Heat capacity, land / ocean | 1.4×10⁷ / 3.1×10⁸ J m⁻² K⁻¹ (model values) | 1×10⁷ / 4×10⁸ J m⁻² K⁻¹ |
| Longwave | A + B·T, A tuned to 200.284, B = 2.09 | A = 203.3, B = 2.09 |
| Grid and run length | 120 equal-area cells, 100 orbits of 360 steps | the same |

Benchmark 1 is tuned with one parameter, the longwave constant A, to reach
288 K. The protocol leaves its configuration open, so it keeps the model's own
albedos.

## Results

| Case | Protocol albedos (`Results/`) | Model's own albedos (`Results_native_albedo/`) |
|---|---|---|
| Benchmark 1 | 288.00 K; sea ice poleward of 54.8° | the same run |
| Benchmark 2 | 302.04 K; ice free | 285.00 K; sea ice poleward of 51.6° |
| Benchmark 3 | 302.05 K; ice free | 288.78 K; ice free |
| Experiment 1 (ice free / caps / belt / snowball) | 139 / 16 / 0 / 35 | 105 / 37 / 3 / 45 |
| Experiment 2 (ice free / caps / belt / snowball) | 149 / 0 / 5 / 36 | 173 / 13 / 0 / 4 |
| Experiment 3: warm start is a snowball up to | 0.9000 S⊕ | 0.9125 S⊕ |
| Experiment 3: cold start thaws from | 1.1875 S⊕ | 1.0625 S⊕ |
| Experiment 3: bistable width | 0.29 S⊕ | 0.15 S⊕ |
| Experiment 4: warm start is a snowball up to | 5.2 ppm | 6.6 ppm |
| Experiment 4: cold start thaws from | 39,100 ppm | 5,960 ppm |

Climate states are classified from the northern sea ice edges.

The model has no atmosphere, so its surface albedo is also its planetary
albedo. With the protocol's values an ice-free planet reflects 22.5% of the
light (0.25 × 0.3 + 0.75 × 0.2), against about 29% for Earth, which is why
Benchmarks 2 and 3 are warm and ice free. In Benchmark 2 the albedo rises to
0.26 near the poles because the land there is snow covered for a little under
half of the year. The protocol paper reports about 301 K with no year-round
ice for VPLanet/POISE, which descends from the same model.

## The variant with the model's own albedos

`Results_native_albedo/` holds the same benchmarks and experiments with the
model's stellar-weighted albedos for a G star (ocean 0.319, land 0.415, ice
0.514) and its zenith-angle term, which follows the star's declination through
the year (`zenithflag = 1`). These stand in for a planetary albedo, and the
contrast between ice and ocean is about half the protocol's. This is the
configuration of Venkatesan et al. (2025). It is not the set to file.

## Departures from the protocol

- **Experiments 3 and 4 use the Benchmark 2 configuration.** Protocol v1.0
  words them as starting from Benchmark 1. Using the untuned configuration
  keeps Experiments 1 to 4 on one model.
- **CO2.** The longwave law has no CO2 term. In Experiment 4, A is shifted by
  the change in the Williams & Kasting (1997) OLR fit between 280 ppm and each
  CO2 value, at 288 K and 1 bar. B is held at the model's value; in that fit
  it changes by less than 0.04 W m⁻² K⁻¹ between 10 and 100,000 ppm. At
  280 ppm the model is the one used in Experiments 1 to 3. The fit is
  extrapolated below 10 ppm.
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
orbit was rerun for 400 orbits. Each file's header records how many cases that
applied to, the largest change left over a final orbit, and the largest
difference between hemispheres. Benchmark 1 differs between hemispheres by
design, because its land is Earth-like.
