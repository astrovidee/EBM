Getting started
===============

Installation
------------

The model needs Python 3 with NumPy, SciPy and Matplotlib.

.. code-block:: bash

   git clone https://github.com/astrovidee/EBM.git
   cd EBM
   pip install -r requirements.txt

A first run
-----------

Copy the default settings, change what you need, and run. This two-year run
takes a few seconds.

.. code-block:: python

   import EBM_one_file as ebm

   cfg = ebm.DEFAULTS.copy()
   cfg["runlength"] = 2    # model years
   cfg["star"] = "G"       # host star type: "F", "G", "K" or "M"
   cfg["ecc"] = 0.0        # orbital eccentricity
   cfg["obl"] = 0.0        # obliquity in degrees
   cfg["scaleQ"] = 1.0     # stellar flux relative to the default

   results = ebm.seasonal_run(cfg)
   print(ebm.mean_iceline(results))   # mean ice-line latitude in degrees: 52.26

Reading the results
-------------------

``seasonal_run`` returns a dictionary. Temperatures are in degrees Celsius and
there are 360 time steps per year.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry
     - Meaning
   * - ``Lann``, ``Wann``
     - Land and ocean temperature in the final year, shape (latitudes, 360)
   * - ``h_ann``
     - Sea ice thickness in metres in the final year
   * - ``alb_l_ann``, ``alb_w_ann``
     - Land and ocean albedo in the final year
   * - ``L_out``, ``W_out``, ``h_out``
     - The same fields for the whole run
   * - ``setup``
     - The grid and inputs: ``setup["phi"]`` is the latitude of each cell in
       degrees, ``setup["fl"]`` and ``setup["fw"]`` are the land and ocean
       fractions, and ``setup["insol"]`` is the insolation in W/m²

Three functions summarize a run:

.. code-block:: python

   means = ebm.final_year_annual_means(results)
   means["Tglob"]     # global mean temperature in kelvin
   means["T_avg"]     # annual mean temperature at each latitude, deg C
   means["A_avg"]     # annual mean albedo at each latitude
   means["lat"]       # latitude of each cell in degrees

   ebm.mean_iceline(results)   # mean ice-line latitude in degrees
   ebm.ice_edges(results)      # latitude limits of sea ice and ice-covered land

The ice line is 0 when the hemisphere is fully frozen and 90 when it is ice
free. It is the latitude where the ocean crosses -2.013 deg C, as in
Venkatesan et al. (2025), and usually lies poleward of the sea ice.
``ice_edges`` gives the extent of the ice itself, from the sea ice thickness.

Warm and cold starts
--------------------

Climates like these can have two stable states at the same stellar flux, one
frozen and one not. Set ``cfg["coldstart"] = 1`` to start from a frozen planet
and leave it at 0 to start from a warm one. Running both over a range of
``scaleQ`` traces the hysteresis loop.

The built-in example
--------------------

.. code-block:: bash

   python EBM_one_file.py

This runs a warm-start sweep over eight stellar fluxes in about two minutes
and writes ``scaleQ_vs_iceline.txt`` and ``scaleQ_vs_iceline.png``. The same
sweep is available from Python as ``ebm.warmstart_sweep``.
