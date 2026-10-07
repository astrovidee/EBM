CO2 ice
=======

On a cold enough planet the CO2 in the atmosphere condenses onto the surface.
CO2 ice is far more reflective than water ice, so it makes a frozen planet
harder to thaw. The model includes this through the albedo, following
Venkatesan et al. (2025).

Running a case
--------------

Use ``co2_run`` in place of ``seasonal_run``, with the Spiegel outgoing
radiation law:

.. code-block:: python

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

``co2_run`` returns the model results and whether the planet ends with CO2
ice. A case takes 15 to 30 seconds with 60 latitude cells.

How it works
------------

CO2 is taken to condense when the global annual mean surface temperature is
at or below ``co2_Tcond``. The default, 131.06 K, is the condensation
temperature of CO2 at 400 ppmv in a 1 bar atmosphere. Once CO2 has condensed,
the CO2 ice albedo replaces the water ice albedo on every frozen surface.

``start="warm"``
   The model first runs from an ice-free start with water ice only. If the
   global mean temperature is at or below ``co2_Tcond``, the case is run again
   with the CO2 ice albedo.

``start="cold"``
   The model first runs from a frozen start with the CO2 ice albedo. If the
   global mean temperature is above ``co2_Tcond``, the CO2 ice is taken to have
   sublimated and the case is run again with water ice only.

The CO2 ice albedo depends on the host star and on the grain size. Larger
grains are darker. The values are those of Table 2 in Venkatesan et al.
(2025), from the CO2 ice spectra of Hansen (1997), and are returned by
``ebm.get_co2_albedo(star, grain)``.

The CO2 ice runs use the sea ice constant ``co2_Lfice`` and the water ice runs
use ``Lfice``, as in the original MATLAB model.

Why the Spiegel law is needed
-----------------------------

With the default linear law, the outgoing radiation A + B·T falls to zero at
T = -A/B, about 176 K. No part of the planet can settle below that
temperature, so the 131.06 K threshold is never reached. With
``olr="linear"``, ``co2_run`` prints a warning and finds no CO2 ice. The
Spiegel law stays positive at all temperatures and has no such floor.

The latitude by latitude option
-------------------------------

Setting ``cfg["co2_ice"] = 1`` and calling ``seasonal_run`` applies the CO2
ice albedo at each latitude and time step where the temperature is at or
below ``co2_Tcond``. This is useful for experiments, but it is not the
procedure that reproduces the paper. In particular it does not hold a cold
start: the 2 m of sea ice that a frozen start begins with conducts enough heat to
lift the surface above the threshold in the first steps.
