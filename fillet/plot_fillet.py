"""
Plot the FILLET benchmarks and experiments written by run_fillet.py.

    python plot_fillet.py

Reads Results/shields_bitz/ next to this script and writes
fillet_shields_bitz.png. It also prints the summary numbers used in the figure.
"""
import glob
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "Results", "shields_bitz")

# Use the Lato fonts that ship with the documentation if they are there, so the
# figure looks the same on every machine. Otherwise fall back to a system sans.
for path in glob.glob(os.path.join(HERE, "..", "docs", "_static", "fonts", "Lato", "*.ttf")):
    font_manager.fontManager.addfont(path)

# Ink and surface
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#c3c2b7"

# Benchmarks: three lines
BENCH_COLORS = ["#5e5ce9", "#eb6834", "#1baf7a"]

# Climate states, in the colors of the FILLET protocol paper's state maps:
# indigo for ice free, orchid for ice caps, periwinkle for an ice belt and light
# gray for a snowball. The orchid is a little pinker than in that paper so it
# stays distinct from the indigo for color-blind readers.
STATES = ["Ice free", "Ice caps", "Ice belt", "Snowball"]
STATE_COLORS = ["#5e5ce9", "#cc68cc", "#aaaaec", "#dcdbdc"]

# Hysteresis branches: warm start in orange, cold start in dodger blue
WARM, COLD = "#eb6834", "#1e90ff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.fontsize": 9.5, "mathtext.default": "regular",
})


def read_global(folder):
    d = np.loadtxt(os.path.join(RES, folder, "global_output.dat"), comments="#", ndmin=2)
    return dict(inst=d[:, 1], obl=d[:, 2], xco2=d[:, 3], T=d[:, 4], nmax_sea=d[:, 7], nmin_sea=d[:, 8], olr=d[:, 14])


def read_lat(folder):
    d = np.loadtxt(os.path.join(RES, folder, "case_0", "lat_output.dat"), comments="#")
    return dict(lat=d[:, 0], T=d[:, 1], alb=d[:, 2], olr=d[:, 4])


def climate_state(nmax, nmin):
    """0 ice free, 1 caps, 2 belt, 3 snowball, from the northern sea-ice edges."""
    s = np.full(len(nmax), 1)
    s[(nmax >= 90) & (nmin >= 90)] = 0
    s[(nmax >= 90) & (nmin <= 0)] = 3
    s[nmax < 90] = 2
    return s


def titled(ax, title, subtitle):
    """A bold title with one line of plain summary under it."""
    ax.set_title(title, pad=22)
    ax.text(0, 1.018, subtitle, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def state_map(ax, g, title):
    """Climate state on the instellation-obliquity grid: instellation across, obliquity up."""
    insts, obls = np.unique(g["inst"]), np.unique(g["obl"])
    state = climate_state(g["nmax_sea"], g["nmin_sea"])
    grid = np.full((len(obls), len(insts)), np.nan)
    for s, o, v in zip(g["inst"], g["obl"], state):
        grid[np.searchsorted(obls, o), np.searchsorted(insts, s)] = v
    ax.pcolormesh(np.arange(len(insts) + 1), np.arange(len(obls) + 1), grid, cmap=ListedColormap(STATE_COLORS),
                  vmin=-0.5, vmax=3.5, edgecolors=SURFACE, linewidth=1.6)
    xt = np.arange(0, len(insts), 4)
    ax.set_xticks(xt + 0.5, [f"{insts[i]:.2f}".rstrip("0").rstrip(".") for i in xt])
    yt = np.arange(0, len(obls), 3)
    ax.set_yticks(yt + 0.5, [f"{obls[i]:.0f}" for i in yt])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Instellation (S$_\\oplus$)")
    ax.set_ylabel("Obliquity (°)")
    counts = [int(np.sum(state == i)) for i in range(4)]
    titled(ax, title, "  ·  ".join(f"{n} {name.lower()}" for n, name in zip(counts, STATES) if n))
    return counts


def branches(ax, warm, cold, xkey, xlabel, logx=False):
    """Global mean temperature of the warm-start and cold-start branches, with the bistable range shaded."""
    ws = climate_state(warm["nmax_sea"], warm["nmin_sea"]) == 3
    cs = climate_state(cold["nmax_sea"], cold["nmin_sea"]) == 3
    x = warm[xkey]
    both = ws != cs          # the two starts end in different states
    if both.any():
        lo, hi = x[both].min(), x[both].max()
        ax.axvspan(lo, hi, color="#efeee9", lw=0, zorder=0)
        ax.text(np.sqrt(lo * hi) if logx else (lo + hi) / 2, 0.035, "two stable states",
                transform=ax.get_xaxis_transform(), ha="center", color=INK2, fontsize=9)
    ax.plot(cold[xkey], cold["T"], color=COLD, lw=2.2, solid_capstyle="round", label="Cold start")
    ax.plot(warm[xkey], warm["T"], color=WARM, lw=2.2, solid_capstyle="round", label="Warm start")
    if logx:
        ax.set_xscale("log")
    ax.grid(axis="y")
    ax.tick_params(length=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Global mean temperature (K)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, reverse=True)
    return ws, cs


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.22, wspace=0.05, hspace=0.07)
    (a, b, c), (d, e, f) = axes

    # Benchmarks
    bens = [("ben1", "Benchmark 1, tuned Earth"), ("ben2", "Benchmark 2, 23.5° obliquity"), ("ben3", "Benchmark 3, 60° obliquity")]
    print("Benchmarks")
    for (folder, name), color in zip(bens, BENCH_COLORS):
        L, G = read_lat(folder), read_global(folder)
        label = f"{name}: {G['T'][0]:.1f} K"
        a.plot(L["lat"], L["T"], color=color, lw=2.2, solid_capstyle="round", label=label)
        b.plot(L["lat"], L["alb"], color=color, lw=2.2, solid_capstyle="round", label=label)
        print(f"  {name}: Tglob {G['T'][0]:.2f} K, northern sea ice edge {G['nmin_sea'][0]:.1f} deg, OLR {G['olr'][0]:.1f} W/m2,"
              f" mean albedo {np.mean(L['alb']):.3f}")
    for ax, ylabel in ((a, "Surface temperature (K)"), (b, "Albedo")):
        ax.set_xlim(-90, 90)
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.grid(axis="y")
        ax.tick_params(length=3)
        ax.set_xlabel("Latitude (°)")
        ax.set_ylabel(ylabel)
    titled(a, "Benchmarks: temperature", "Annual mean at each latitude")
    titled(b, "Benchmarks: albedo", "Annual mean at each latitude")
    a.legend(frameon=False, loc="lower center", handlelength=1.5, borderaxespad=0.2)

    # Experiments 1 and 2
    c1 = state_map(d, read_global("exp1"), "Experiment 1: warm start")
    c2 = state_map(e, read_global("exp2"), "Experiment 2: cold start")
    print("Experiment 1 (ice free / caps / belt / snowball):", "/".join(map(str, c1)))
    print("Experiment 2 (ice free / caps / belt / snowball):", "/".join(map(str, c2)))
    fig.legend(handles=[Patch(facecolor=col, label=name) for col, name in zip(STATE_COLORS, STATES)], frameon=False,
               ncol=4, loc="outside lower left", handlelength=1.3, handleheight=1.0, columnspacing=1.8,
               title="Climate state in Experiments 1 and 2", title_fontsize=9.5, alignment="left")

    # Experiment 3
    w3, k3 = read_global("exp3_warm"), read_global("exp3_cold")
    ws, cs = branches(c, w3, k3, "inst", "Instellation (S$_\\oplus$)")
    glac = w3["inst"][ws].max() if ws.any() else np.nan
    degl = k3["inst"][~cs].min() if (~cs).any() else np.nan
    titled(c, "Experiment 3: instellation",
           f"Snowball up to {glac:.4f}  ·  thaws from {degl:.4f}  ·  width {degl - glac:.2f} S$_\\oplus$")
    print(f"Experiment 3: warm start is a snowball up to {glac:.4f} S; cold start thaws from {degl:.4f} S; width {degl - glac:.4f} S")

    # Experiment 4
    w4, k4 = read_global("exp4_warm"), read_global("exp4_cold")
    ws4, cs4 = branches(f, w4, k4, "xco2", "CO$_2$ (ppm)", logx=True)
    glac4 = w4["xco2"][ws4].max() if ws4.any() else np.nan
    degl4 = k4["xco2"][~cs4].min() if (~cs4).any() else np.nan
    titled(f, "Experiment 4: CO$_2$", f"Snowball up to {glac4:.2g} ppm  ·  thaws from {round(degl4, -1):,.0f} ppm")
    print(f"Experiment 4: warm start is a snowball up to {glac4:.3g} ppm; cold start thaws from {degl4:.3g} ppm")

    fig.suptitle("Shields-Bitz EBM: FILLET benchmarks and experiments", x=0.012, ha="left", fontsize=16, fontweight="bold")
    out = os.path.join(HERE, "fillet_shields_bitz.png")
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
