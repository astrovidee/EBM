Comparison with Venkatesan et al. (2025)
========================================

The results in Venkatesan et al. (2025) were produced with the original
MATLAB model. This page compares the Python model with numbers reported in
that paper.

All runs here use ``olr="spiegel"`` with every other setting at its default,
60 latitude cells and 100 model years. CO2 cases use ``co2_run`` with 200
micron grains unless stated. Fluxes are percentages of the default stellar
flux, orbit-averaged, and are stepped by 5% of ``scaleQ`` as in the paper.

Circular orbits
---------------

.. list-table::
   :header-rows: 1
   :widths: 52 24 24

   * - Result
     - Paper
     - This model
   * - Warm start, G star: ice line at 90% flux
     - 40.4°
     - 40.6°
   * - Warm start, G star: flux at which the planet freezes over
     - 85%
     - 85%
   * - Cold start: flux at which the water ice thaws, F / G / K / M
     - 115 / 110 / 105 / 80%
     - 115 / 110 / 105 / 80%
   * - Warm start: CO2 ice forms at 5% flux, F / G / K / M
     - yes for all
     - yes for all
   * - Warm start: CO2 ice at 10% flux, G and M
     - no
     - no
   * - Cold start: flux at which the CO2 ice is gone, F / G / K / M
     - 75 / 65 / 55 / 30%
     - 75 / 65 / 55 / 30%

Eccentricity 0.5
----------------

.. list-table::
   :header-rows: 1
   :widths: 52 24 24

   * - Result
     - Paper
     - This model
   * - Cold start: flux at which the CO2 ice is gone, F / G / K / M
     - 80.8 / 69.3 / 57.7 / 28.9%
     - 80.8 / 69.3 / 57.7 / 28.9%
   * - Same, G star with 2 micron grains
     - 340.6%
     - 340.6%
   * - Same, G star with 20 micron grains
     - 150.1%
     - 150.1%
   * - Same, G star with 2000 micron grains
     - 46.2%
     - 40.4% or lower
   * - Warm start at 5.8% flux: CO2 ice on an M star planet
     - no
     - no
   * - Warm start, G star: flux at which the planet freezes over
     - 40%
     - 40.4%
   * - Cold start, G star at 69.3% flux: mean ice line
     - 10.9°
     - 12.0°

Known differences
-----------------

- **Ice lines after thawing.** For circular orbits the flux at which each
  planet thaws matches, but the ice line just after thawing is lower than in
  the paper: 66° against 77° for F, 64° against 71° for G, 61° against 67° for
  K, and 46° against 47° for M.
- **2000 micron grains.** At eccentricity 0.5 the model loses its CO2 ice at
  least one flux step earlier than the paper.
- **Earth and Mars checks.** The paper quotes 293 K and an ice line of 64° for
  Earth, and 211 K for Mars at 43% flux. With the Spiegel law this model gives
  286 K and 64° for an Earth-like case (obliquity 23.44°, eccentricity 0.0167)
  and 200 K at 43% flux. With the linear law it gives 210 K at 43% flux.
- **Narrow margins.** Several CO2 cases sit within 0.1 K of the 131.06 K
  threshold, so small changes to the settings can move them by one flux step.
- **Not compared.** Eccentricity 0.9, the runaway greenhouse limits, and the
  increase in flux needed to thaw a planet when CO2 ice is included.

How this model relates to the MATLAB model
------------------------------------------

- **CO2 ice albedo.** ``co2_run`` puts the CO2 ice albedo on every surface at
  or below -2 deg C. This is the rule in the MATLAB CO2 albedo routine.
- **Spiegel law.** The MATLAB model linearizes the Spiegel law about the
  global mean temperature at every time step. This model applies the law at
  each latitude. In a Python test of the two approaches, global mean
  temperatures agreed to within 0.02 K for the CO2 cases. For water ice, the
  two approaches can differ by one flux step near the freezing and thawing
  thresholds.
- **Sea ice constant.** The CO2 ice runs use 9.8·246/50, as in the MATLAB CO2
  setup.
- **CO2 condensation test.** The test of the global mean temperature against
  ``co2_Tcond`` was reconstructed from the paper and from the results above.
  It has not been compared with the MATLAB script that drove the published
  runs.
