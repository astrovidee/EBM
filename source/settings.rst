Settings
========

Every setting lives in the dictionary ``ebm.DEFAULTS``. Copy it, change the
entries you need, and pass it to ``seasonal_run`` or ``co2_run``.

Planet and star
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Setting
     - Meaning
     - Default
   * - ``star``
     - Host star type: ``"F"``, ``"G"``, ``"K"`` or ``"M"``. Sets the albedos
       of ocean, land, water ice and CO2 ice.
     - ``"G"``
   * - ``scaleQ``
     - Stellar flux relative to ``solar_constant``. For an eccentric orbit the
       orbit-averaged flux is larger by a factor 1/sqrt(1 - ecc²).
     - 1.0
   * - ``solar_constant``
     - Stellar flux at the planet's semi-major axis for ``scaleQ = 1``, in
       W/m²
     - about 1353
   * - ``ecc``
     - Orbital eccentricity
     - 0.0
   * - ``obl``
     - Obliquity in degrees
     - 0.0
   * - ``per``
     - Longitude of periastron in degrees
     - 102.07
   * - ``land``
     - Land configuration; see below
     - ``"modern"``

Land configurations
-------------------

Only the first letters of the name are checked.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Land fraction
   * - ``"modern"``
     - Modern Earth continents in latitude bands, as in Venkatesan et al. (2025)
   * - ``"smooth"``
     - Modern Earth continents, interpolated smoothly between the bands
   * - ``"Fillet"``
     - 25% land at every latitude
   * - ``"Aquaplanet"``
     - 1% land
   * - ``"Landplanet"``
     - 99% land
   * - ``"Symmetric"``
     - 34% land at every latitude
   * - ``"Precambrian"``, ``"Ordovician"``
     - Paleo-continents

Climate physics
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Setting
     - Meaning
     - Default
   * - ``olr``
     - Outgoing longwave radiation law. ``"linear"`` is A + B·T with T in
       deg C (North and Coakley 1979). ``"spiegel"`` is
       σT⁴ / (1 + 0.75 τ) with τ = 0.79 (T / 273 K)³ (Spiegel et al. 2008).
     - ``"linear"``
   * - ``A``, ``B``
     - Constants of the linear law, in W/m² and W/m²/K. With the Spiegel law
       they do not affect the climate; ``B`` only helps the solver converge.
     - 203.3, 2.09
   * - ``Dmag``
     - Heat diffusion coefficient in W/m²/K
     - 0.44
   * - ``hadleyflag``
     - 1 strengthens heat transport in the tropics to represent the Hadley
       circulation
     - 1
   * - ``Cl``, ``Cw``
     - Heat capacity of land and of the ocean mixed layer, in W yr/m²/K
     - 0.45, 9.8
   * - ``nu``
     - Heat exchange between land and ocean in W/m²/K
     - 3.0
   * - ``ice_model``
     - 1 turns on the sea ice model
     - 1
   * - ``Lfice``
     - Latent heat constant of sea ice
     - 9.8·83.5/50
   * - ``zenithflag``
     - 1 makes the albedo of ice-free land and ocean follow the star's
       declination through the year. 0 makes it depend on latitude only.
     - 1
   * - ``rghflag``
     - 1 fixes the outgoing radiation at 300 W/m² above 46.2 deg C, a simple
       runaway greenhouse limit
     - 0
   * - ``albedoflag``
     - 1 uses a prescribed albedo climatology from a file ``temperatures.mat``,
       if present, in place of the albedo feedback
     - 0

CO2 ice
-------

See :doc:`co2` for how these are used.

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Setting
     - Meaning
     - Default
   * - ``co2_grain``
     - CO2 ice grain size in microns: 1, 2, 5, 20, 100, 200 or 2000
     - 200
   * - ``co2_Tcond``
     - CO2 condensation temperature in deg C. The default is 131.06 K, for
       400 ppmv of CO2 at 1 bar.
     - -142.09
   * - ``co2_Lfice``
     - Value of ``Lfice`` that ``co2_run`` uses for its CO2 ice runs
     - 9.8·246/50
   * - ``co2_ice``
     - 1 applies the CO2 ice albedo latitude by latitude inside
       ``seasonal_run``. ``co2_run`` sets this itself.
     - 0
   * - ``co2_coldstart``
     - With ``co2_ice``, 1 starts every latitude 10 deg C below ``co2_Tcond``
     - 0

Run control
-----------

.. list-table::
   :header-rows: 1
   :widths: 18 62 20

   * - Setting
     - Meaning
     - Default
   * - ``jmx``
     - Number of latitude cells, evenly spaced in sin(latitude)
     - 120
   * - ``runlength``
     - Length of the run in model years
     - 100
   * - ``coldstart``
     - 1 starts 40 deg C colder, from a frozen planet
     - 0
   * - ``casename``
     - A label for the run. It does not affect the results.
     - ``"Control"``
   * - ``Toffset``
     - Not used
     - -30
