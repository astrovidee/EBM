Using the model for FILLET
==========================

This model takes part in the FILLET intercomparison of energy balance models
(Deitrick et al. 2023; Barnes et al. 2025), where it is called Shields-Bitz.
This page lists the settings that matter for the protocol, what the model
cannot do, and the status of the findings from the FILLET code audit of
September 2026.

The scripts that run the FILLET benchmarks and experiments, their results and
a figure are in the ``fillet`` folder of the repository, with a README that
lists the configuration and the departures from the protocol.

Settings for the protocol
-------------------------

Take the prescribed values from the protocol itself. These are the settings
they map onto:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Protocol quantity
     - Setting
   * - Instellation
     - ``solar_constant = 1361.0`` makes ``scaleQ`` the instellation in units
       of S⊕ as the protocol defines it. The default is about 1353.
   * - Land fraction of 25%
     - ``land = "Fillet"``
   * - Surface albedos of land, ocean and ice (0.3, 0.2, 0.6)
     - ``albedo_land``, ``albedo_ocean`` and ``albedo_ice``. Each becomes a
       constant, with no zenith-angle term.
   * - Constant diffusion
     - ``hadleyflag = 0`` and ``Dmag`` set to the prescribed value
   * - Obliquity
     - ``obl``
   * - Heat capacities
     - ``Cl`` and ``Cw``, in W yr m⁻² K⁻¹
   * - Warm or cold start
     - ``coldstart = 0`` or ``1``

The model has no atmosphere, so its surface albedo is also its planetary
albedo. With the prescribed values an ice-free planet reflects 23% of the
light, against about 29% for Earth, and Benchmark 2 comes out near 302 K and
ice free. The protocol paper reports about 301 K, with no year-round ice, for
VPLanet/POISE, which descends from the same model.

Left at ``None``, the three albedo settings give the model's own
stellar-weighted albedos. In that case ``zenithflag`` matters: with 1, the
default, the albedo of ice-free land and ocean follows the star's declination
through the year, as in the original MATLAB model, and with 0 it depends on
latitude only, as in version 1.0 of this code. The two differ by about 2 K in
the global mean at 60° obliquity.

A benchmark run and its outputs:

.. code-block:: python

   import EBM_one_file as ebm

   cfg = ebm.DEFAULTS.copy()
   cfg["solar_constant"] = 1361.0
   cfg["land"] = "Fillet"
   cfg["albedo_land"], cfg["albedo_ocean"], cfg["albedo_ice"] = 0.3, 0.2, 0.6
   cfg["obl"] = 23.5

   results = ebm.seasonal_run(cfg)
   means = ebm.final_year_annual_means(results)   # Tglob and latitude profiles
   edges = ebm.ice_edges(results)                 # the eight ice-edge columns

``ice_edges`` returns the eight ice-edge values of Protocol v1.1 from the
model's own ice: sea ice thickness for the sea edges and the land albedo
switch for the land edges. Do not use ``mean_iceline`` for FILLET. It is a
temperature crossing that lies poleward of the sea ice.

What the model cannot do
------------------------

- **A CO2 term in the longwave.** The linear law has none. Experiment 4 needs
  a pair of ``A`` and ``B`` for each CO2 value.
- **Experiments 1a and 2a.** The year is fixed at 360 time steps, so the
  orbital period cannot be varied without a code change.

Findings of the code audit
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Finding
     - Status
   * - Insolation scaled by ``scaleQ`` twice
     - Fixed in version 1.1
   * - ``mean_iceline`` read evenly spaced latitudes on a grid that is evenly
       spaced in sin(latitude)
     - Fixed in version 1.1
   * - ``mean_iceline`` threshold of -2.013 °C misses sea ice at its melting
       point
     - ``mean_iceline`` keeps the definition of Venkatesan et al. (2025).
       ``ice_edges`` gives the extent of the ice itself.
   * - Solar constant fixed at about 1353 W m⁻²
     - Now the setting ``solar_constant``
   * - Sea ice step does not conserve energy exactly: up to 0.44 W m⁻² at
       equilibrium when sea ice is present
     - Open. Inherited from the original model.
   * - Snow-free land is brighter than ice-covered land poleward of about 59°
       for a G star
     - Open. It follows from the albedo values of the original model.
   * - Defects in the multi-file version
     - The multi-file version has been removed.

Check against the audit's runs
------------------------------

The audit ran this model for Benchmarks 2 and 3 with ``land = "Fillet"``, the
model's own albedos and the other settings at their defaults. With
``zenithflag = 0`` the current code reproduces those runs:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Case
     - Audit
     - ``zenithflag = 0``
     - ``zenithflag = 1``
   * - Benchmark 2 (23.5° obliquity), global mean
     - 284.17 K
     - 284.17 K
     - 284.32 K
   * - Benchmark 3 (60° obliquity), global mean
     - 286.00 K
     - 286.00 K
     - 288.11 K

In Benchmark 2 the sea ice reaches 51.6° in both hemispheres, where the audit
found 52° to 53°.

The audit also reports that substituting the prescribed albedos into this
model gives an ice-free planet at 302 K. The FILLET run of Benchmark 2 with
the prescribed albedos gives 302.04 K, ice free.
