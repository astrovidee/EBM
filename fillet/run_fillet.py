"""
Run the FILLET benchmarks and experiments with the Shields-Bitz EBM and write
the results in the layout of the FILLET repository (projectcuisines/fillet).

    python run_fillet.py              # everything, about 25 minutes on 2 cores
    python run_fillet.py ben exp3     # only the named parts
    python run_fillet.py --workers 4
    python run_fillet.py --albedo native   # the same runs with the model's own albedos

Parts: ben (Benchmarks 1-3), exp1, exp2, exp3, exp4.

Benchmarks 2 and 3 and the experiments use the surface albedos of protocol
Table 4 (land 0.3, ocean 0.2, ice 0.6) and write to Results/shields_bitz/ next
to this script. With --albedo native they use the model's own stellar-weighted
albedos and write to Results_native_albedo/shields_bitz/.

Every setting the protocol prescribes is set here explicitly. Nothing relies
on the model's defaults. See the README at the top of the repository for what
is declared as a departure from the protocol.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per case; cases run in parallel
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import EBM_one_file as ebm   # noqa: E402

OUT = os.path.join(HERE, "Results", "shields_bitz")
SECONDS_PER_YEAR = 365.25 * 86400.0

# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

# Protocol v1.0 Table 4 and v1.1: used for Benchmarks 2 and 3 and all experiments.
PROTOCOL = dict(
    star="G",
    solar_constant=1361.0,              # 1 S_Earth as the protocol defines it
    scaleQ=1.0,
    ecc=0.0,
    obl=23.5,
    land="Fillet",                      # 25% land at every latitude
    hadleyflag=0.0, Dmag=0.5,           # constant diffusion, W m^-2 K^-1
    Cl=1.0e7 / SECONDS_PER_YEAR,        # land heat capacity 1e7 J m^-2 K^-1
    Cw=4.0e8 / SECONDS_PER_YEAR,        # ocean heat capacity 4e8 J m^-2 K^-1
    A=203.3, B=2.09, olr="linear",      # the model's published longwave law
    albedo_land=0.3, albedo_ocean=0.2, albedo_ice=0.6,   # protocol Table 4: constants, no zenith-angle term
    zenithflag=1.0,                     # only matters with --albedo native: albedo follows the star's declination
    coldstart=0.0,
    jmx=120, runlength=100,
)
XCO2_DEFAULT = 280.0

# --albedo native: the model's own stellar-weighted albedos, with its zenith-angle term
NATIVE_ALBEDO = dict(albedo_land=None, albedo_ocean=None, albedo_ice=None)

# Benchmark 1 (pre-industrial Earth, tuning allowed): the model's own Earth
# set-up, with the longwave constant A tuned to give 288 K.
EARTH = dict(
    star="G", solar_constant=1361.0, scaleQ=1.0, ecc=0.0, obl=23.5,
    land="smooth", hadleyflag=1.0, Dmag=0.44, Cl=0.45, Cw=9.8,
    A=203.3, B=2.09, olr="linear", zenithflag=1.0, coldstart=0.0,
    albedo_land=None, albedo_ocean=None, albedo_ice=None,   # the model's own albedos
    jmx=120, runlength=100,
)
BENCHMARK1_TARGET_K = 288.0

DRIFT_LIMIT = 1.0e-2      # K per orbit; cases above it are rerun for longer
LONG_RUN_FACTOR = 4


# --------------------------------------------------------------------------
# CO2 for Experiment 4
# --------------------------------------------------------------------------

def olr_wk97(pCO2, T):
    """OLR fit of Williams & Kasting (1997), as in fillet/src/polynomials.py. pCO2 in bar, T in K."""
    phi = np.log(pCO2 / 3.3e-4)
    return (9.468980 - 7.714727e-5 * phi - 2.794778 * T - 3.244753e-3 * phi * T - 3.547406e-4 * phi**2.
            + 2.212108e-2 * T**2 + 2.229142e-3 * phi**2 * T + 3.088497e-5 * phi * T**2 - 2.789815e-5 * (phi * T)**2
            - 3.442973e-3 * phi**3 - 3.361939e-5 * T**3 + 9.173169e-3 * phi**3 * T - 7.775195e-5 * phi**3 * T**2
            - 1.679112e-7 * phi * T**3 + 6.590999e-8 * phi**2 * T**3 + 1.528125e-7 * phi**3 * T**3
            - 3.367567e-2 * phi**4 - 1.631909e-4 * phi**4 * T + 3.663871e-6 * phi**4 * T**2
            - 9.255646e-9 * phi**4 * T**3)


def longwave_constant_for_co2(xco2_ppm, A0=203.3, T_ref=288.0):
    """
    The model has no CO2 term. For Experiment 4, A is shifted by the change in
    the Williams & Kasting (1997) OLR between 280 ppm and the requested CO2, at
    288 K in a 1 bar atmosphere. B is unchanged, so at 280 ppm the model is
    exactly the one used in the other experiments.
    """
    return A0 + float(olr_wk97(xco2_ppm * 1e-6, T_ref) - olr_wk97(XCO2_DEFAULT * 1e-6, T_ref))


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------

def run_case(job):
    """Run one case and reduce it to the FILLET outputs."""
    cfg = dict(job["cfg"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, drift = _run_and_drift(cfg)
        years = cfg["runlength"]
        if drift > DRIFT_LIMIT:
            cfg["runlength"] = cfg["runlength"] * LONG_RUN_FACTOR
            res, drift = _run_and_drift(cfg)
            years = cfg["runlength"]

    setup = res["setup"]
    means = ebm.final_year_annual_means(res)
    T_lat = means["T_avg"]                       # deg C, land and ocean weighted
    olr_lat = setup["A"] + setup["B"] * T_lat    # linear law, so the annual mean commutes
    edges = ebm.ice_edges(res)
    half = setup["jmx"] // 2
    asym = float(np.max(np.abs(T_lat[half:] - T_lat[:half][::-1])))
    out = dict(
        key=job["key"], inst=cfg["scaleQ"], obl=cfg["obl"], xco2=job.get("xco2", XCO2_DEFAULT),
        Tglob=float(means["Tglob"]), OLRglob=float(np.mean(olr_lat)), edges=edges,
        diff=cfg["Dmag"], drift=drift, asym=asym, years=years, A=cfg["A"],
    )
    if job.get("keep_lat"):
        out["lat"] = dict(lat=means["lat"], T=T_lat + 273.15, alb=means["A_avg"], olr=olr_lat)
    return out


def _run_and_drift(cfg):
    res = ebm.seasonal_run(cfg)
    setup = res["setup"]
    n = setup["nstepinyear"]
    fl, fw = setup["fl"][:, None], setup["fw"][:, None]
    last = np.mean(fl * res["L_out"][:, -n:] + fw * res["W_out"][:, -n:])
    prev = np.mean(fl * res["L_out"][:, -2 * n:-n] + fw * res["W_out"][:, -2 * n:-n])
    return res, float(abs(last - prev))


def _earth_tglob(A):
    return run_case(dict(key=0, cfg=dict(EARTH, A=A)))["Tglob"]


def tune_benchmark1(pool):
    """Find the longwave constant A that gives 288 K for the Earth set-up (secant method)."""
    A0, A1 = 203.3, 197.0
    T0, T1 = pool.map(_earth_tglob, [A0, A1])
    for _ in range(8):
        if abs(T1 - BENCHMARK1_TARGET_K) < 0.005:
            break
        A2 = A1 + (BENCHMARK1_TARGET_K - T1) * (A1 - A0) / (T1 - T0)
        A0, T0, A1 = A1, T1, A2
        T1 = pool.map(_earth_tglob, [A1])[0]
    return A1


# --------------------------------------------------------------------------
# Writing files
# --------------------------------------------------------------------------

ICE_KEYS = ("NMaxLand", "NMinLand", "NMaxSea", "NMinSea", "SMaxLand", "SMinLand", "SMaxSea", "SMinSea")
ICE_FREE = dict(NMaxLand=90.0, NMinLand=90.0, NMaxSea=90.0, NMinSea=90.0,
                SMaxLand=-90.0, SMinLand=-90.0, SMaxSea=-90.0, SMinSea=-90.0)

MODEL_NOTES = """\
# Code: Shields-Bitz EBM (EBM_one_file.py, https://github.com/astrovidee/EBM), version {version}
# Grid: {jmx} cells of equal area (evenly spaced in sin latitude). Global means are plain means over cells.
# Surfaces: separate land and ocean temperatures at each latitude; thermodynamic sea ice.
{albedo_note}# Longwave: A + B*T with T in deg C. No CO2 term (see the Experiment 4 header for how CO2 is handled there).
# Sea ice has no heat capacity in this model (the protocol lists 1e7 J m^-2 K^-1).
# Every case starts from the prescribed warm or cold state. No case continues from another.
#   Warm start: 7.5 + 20*(1 - 2*sin(lat)^2) deg C. Cold start: 40 deg C colder, with 2 m of sea ice.
# Run length: {years} orbits of 360 steps; a case whose global mean still changed by more than
#   {drift_limit:g} K over its final orbit was rerun for {long_years} orbits.
"""

ALBEDO_PRESCRIBED = """\
# Albedo: constant surface albedos, land {land:g}, ocean {ocean:g}, ice {ice:g} (protocol Table 4), with no
#   zenith-angle term. Land and ocean take the ice value at or below -2 C. The model has no atmosphere, so
#   Asurf and ATOA are the same. Both are unweighted time means over the final orbit.
"""

ALBEDO_NATIVE = """\
# Albedo: the model's own stellar-weighted values for a G star (ocean 0.319, land 0.415, ice 0.514), not the
#   protocol's 0.2/0.3/0.6, with a zenith-angle term on ice-free surfaces that follows the star's
#   declination (zenithflag = 1). The model has no atmosphere, so Asurf and ATOA are the same. Both are
#   unweighted time means over the final orbit.
"""


def albedo_note(base):
    if base.get("albedo_land") is None:
        return ALBEDO_NATIVE
    return ALBEDO_PRESCRIBED.format(land=base["albedo_land"], ocean=base["albedo_ocean"], ice=base["albedo_ice"])


ICE_NOTE = """\
# Describe how ice line latitude is determined: from the model's own ice. Sea: sea ice thickness above zero
#   for at least half of the final orbit. Land: land temperature at or below -2 C, where the land albedo
#   switches to ice, for at least half of the final orbit. Edges lie on the boundaries between cells.
# Ice-free convention: NMax = NMin = 90 and SMax = SMin = -90.
"""

COLUMNS = ("# Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea "
           "IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob\n")


def edge_values(edges):
    return [ICE_FREE[k] if np.isnan(edges[k]) else edges[k] for k in ICE_KEYS]


def write_global(folder, title, rows, base, extra=""):
    os.makedirs(os.path.join(OUT, folder), exist_ok=True)
    worst_drift = max(r["drift"] for r in rows)
    worst_asym = max(r["asym"] for r in rows)
    longer = sum(r["years"] > base["runlength"] for r in rows)
    with open(os.path.join(OUT, folder, "global_output.dat"), "w") as f:
        f.write(f"# Name of benchmark/experiment: {title}\n")
        f.write(MODEL_NOTES.format(version=VERSION, jmx=base["jmx"], years=base["runlength"],
                                   drift_limit=DRIFT_LIMIT, long_years=base["runlength"] * LONG_RUN_FACTOR,
                                   albedo_note=albedo_note(base)))
        f.write(f"# Diffusion: {'constant D' if not base['hadleyflag'] else 'Hadley profile, D = Diff*[1 + 9*exp(-(sin(lat)/sin 25)^6)]'}."
                f" Heat capacities: land {base['Cl'] * SECONDS_PER_YEAR:.3g}, ocean {base['Cw'] * SECONDS_PER_YEAR:.3g} J m^-2 K^-1.\n")
        f.write(f"# Land: {base['land']}. Instellation unit: 1361 W m^-2.\n")
        f.write(extra)
        f.write(f"# Convergence: largest change in Tglob over the final orbit {worst_drift:.1e} K;"
                f" largest north-south difference in annual-mean temperature {worst_asym:.1e} K;"
                f" {longer} of {len(rows)} cases rerun for longer.\n")
        f.write(ICE_NOTE)
        f.write("#\n# Columns: Inst (S_earth), Obl (deg), XCO2 (ppm), Tglob (K), ice edges (deg),"
                " Diff (W m^-2 K^-1), OLRglob (W m^-2)\n")
        f.write(COLUMNS)
        for i, r in enumerate(rows):
            e = " ".join(f"{v:.2f}" for v in edge_values(r["edges"]))
            f.write(f"{i} {r['inst']:.4f} {r['obl']:.1f} {r['xco2']:.4f} {r['Tglob']:.3f} {e} {r['diff']:.3f} {r['OLRglob']:.3f}\n")


def write_lat(folder, title, r):
    os.makedirs(os.path.join(OUT, folder, "case_0"), exist_ok=True)
    with open(os.path.join(OUT, folder, "case_0", "lat_output.dat"), "w") as f:
        f.write(f"# Name of benchmark/experiment: {title}\n# Code: Shields-Bitz EBM, version {VERSION}\n# Case number: 0\n")
        f.write(f"# Instellation (S_earth): {r['inst']:.4f}\n# XCO2 (ppm): {r['xco2']:.1f}\n# Obliquity (degrees): {r['obl']:.1f}\n")
        f.write("# Cells have equal area. Asurf and ATOA are the same (no atmosphere) and are unweighted time means.\n#\n")
        f.write("# Columns of data (annually averaged for last orbit)\n# Lat Tsurf Asurf ATOA OLR\n")
        L = r["lat"]
        for lat, T, a, o in zip(L["lat"], L["T"], L["alb"], L["olr"]):
            f.write(f"{lat:8.3f} {T:8.3f} {a:.4f} {a:.4f} {o:8.3f}\n")


# --------------------------------------------------------------------------
# Benchmarks and experiments
# --------------------------------------------------------------------------

def grid(start, stop, step):
    return np.round(np.arange(start, stop + step / 2, step), 6)


def do_benchmarks(pool):
    A_tuned = tune_benchmark1(pool)
    jobs = [dict(key="ben1", cfg=dict(EARTH, A=A_tuned), keep_lat=True),
            dict(key="ben2", cfg=dict(PROTOCOL, obl=23.5), keep_lat=True),
            dict(key="ben3", cfg=dict(PROTOCOL, obl=60.0), keep_lat=True)]
    r1, r2, r3 = pool.map(run_case, jobs)
    write_global("ben1", "Benchmark 1 (pre-industrial Earth, tuned)", [r1], EARTH,
                 extra=f"# Tuning: longwave constant A = {A_tuned:.3f} W m^-2 (model value 203.3) to reach {BENCHMARK1_TARGET_K} K."
                       " Earth-like land, the model's own diffusion and heat capacities.\n")
    write_lat("ben1", "Benchmark 1 (pre-industrial Earth, tuned)", r1)
    write_global("ben2", "Benchmark 2 (un-tuned, obliquity 23.5)", [r2], PROTOCOL)
    write_lat("ben2", "Benchmark 2 (un-tuned, obliquity 23.5)", r2)
    write_global("ben3", "Benchmark 3 (un-tuned, obliquity 60)", [r3], PROTOCOL)
    write_lat("ben3", "Benchmark 3 (un-tuned, obliquity 60)", r3)
    for name, r in (("Benchmark 1", r1), ("Benchmark 2", r2), ("Benchmark 3", r3)):
        print(f"{name}: Tglob = {r['Tglob']:.2f} K, northern sea ice from {edge_values(r['edges'])[3]:.1f} to"
              f" {edge_values(r['edges'])[2]:.1f} deg, OLR = {r['OLRglob']:.1f} W/m2")


BIFURCATION_NOTE = ("# Configuration: that of Benchmark 2 (untuned), so that Experiments 1 to 4 describe one model."
                    " Protocol v1.0 words\n#   Experiments 3 and 4 as starting from Benchmark 1.\n")


def do_sweep(pool, folder, title, insts, obls, coldstart, extra=""):
    # instellation is the outer loop, obliquity the inner one
    jobs = [dict(key=(s, o), cfg=dict(PROTOCOL, scaleQ=float(s), obl=float(o), coldstart=coldstart))
            for s in insts for o in obls]
    rows = pool.map(run_case, jobs, chunksize=1)
    write_global(folder, title, rows, PROTOCOL, extra=extra)
    print(f"{folder}: {len(rows)} cases written")


def do_exp4(pool):
    xco2 = np.logspace(0, 5, 50)
    for folder, cold, name in (("exp4_warm", 0.0, "warm start"), ("exp4_cold", 1.0, "cold start")):
        jobs = [dict(key=x, xco2=float(x), cfg=dict(PROTOCOL, A=longwave_constant_for_co2(x), coldstart=cold)) for x in xco2]
        rows = pool.map(run_case, jobs, chunksize=1)
        write_global(folder, f"Experiment 4 (bifurcation, varying CO2, {name})", rows, PROTOCOL,
                     extra=BIFURCATION_NOTE +
                           "# CO2: the model has no CO2 term. A is shifted by the change in the Williams & Kasting (1997) OLR fit\n"
                           "#   between 280 ppm and each CO2 value, at 288 K and 1 bar. B is held at the model's value: in that fit\n"
                           "#   it changes by less than 0.04 W m^-2 K^-1 between 10 and 100,000 ppm. At 280 ppm the model is\n"
                           "#   identical to the one used in Experiments 1 to 3. The fit is extrapolated below 10 ppm.\n")
        print(f"{folder}: {len(rows)} cases written")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", default=["ben", "exp1", "exp2", "exp3", "exp4"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--albedo", choices=["protocol", "native"], default="protocol",
                        help="protocol: land 0.3, ocean 0.2, ice 0.6 (default). native: the model's own albedos.")
    args = parser.parse_args()
    global OUT
    if args.albedo == "native":
        PROTOCOL.update(NATIVE_ALBEDO)
        OUT = os.path.join(HERE, "Results_native_albedo", "shields_bitz")
    print(f"Albedos for Benchmarks 2-3 and the experiments: {args.albedo}. Writing to {os.path.relpath(OUT, HERE)}/")
    obls = grid(0, 90, 10)
    with Pool(args.workers) as pool:
        if "ben" in args.parts:
            do_benchmarks(pool)
        if "exp1" in args.parts:
            do_sweep(pool, "exp1", "Experiment 1 (G dwarf, warm start)", grid(0.8, 1.25, 0.025), obls, 0.0)
        if "exp2" in args.parts:
            do_sweep(pool, "exp2", "Experiment 2 (G dwarf, cold start)", grid(1.05, 1.5, 0.025), obls, 1.0)
        if "exp3" in args.parts:
            do_sweep(pool, "exp3_warm", "Experiment 3 (bifurcation, varying S, warm start)", grid(0.8, 1.5, 0.0125), [23.5], 0.0,
                     extra=BIFURCATION_NOTE)
            do_sweep(pool, "exp3_cold", "Experiment 3 (bifurcation, varying S, cold start)", grid(0.8, 1.5, 0.0125), [23.5], 1.0,
                     extra=BIFURCATION_NOTE)
        if "exp4" in args.parts:
            do_exp4(pool)


def _model_version():
    path = os.path.join(os.path.dirname(HERE), "citation.cff")
    try:
        for line in open(path):
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "unknown"


VERSION = _model_version()

if __name__ == "__main__":
    main()
