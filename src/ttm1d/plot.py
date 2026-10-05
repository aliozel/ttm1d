"""Figures from finished runs (post-processing only): `ttm1d plot RESULT_FOLDER ...`.

Takes up to 8 folders written by `ttm1d run` and writes, by default to figures/ next to the first:
    surface_temperature.png   peak surface temperature in every micropulse period, then the cooling
                              (the curve itself for a single pulse)
    depth_profiles.png        lattice temperature against depth, one panel per run
    first_pulses.png          electron and lattice temperatures over the first five micropulses
                              (the whole run for a single pulse)
Times are shown in ps, ns or µs, whichever suits the length plotted.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# chart chrome and series colours (light surface), in the order runs are given
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TIME_RAMP = LinearSegmentedColormap.from_list("time", ["#9ec5f4", "#3987e5", "#184f95", "#0d366b"])  # early -> late

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
    "legend.labelcolor": INK2, "savefig.dpi": 200,
})


class Run:
    """A finished run, read from results/<name>/.  Also reads runs made before the YAML input."""

    def __init__(self, folder: Path):
        if not (folder / "summary.json").exists():
            raise SystemExit(f"no results in {folder}")
        self.name = name = folder.name
        self.summary = json.loads((folder / "summary.json").read_text())
        self.history = np.load(folder / "history.npz")
        self.profiles = np.load(folder / "profiles.npz")
        s = self.summary
        self.label = s.get("label", name)
        self.period = (s.get("micropulse_period_ns") or s["inputs"]["period_ns"]) * 1e-9
        self.train_end = (s.get("pulse_train_end_us") or s["macropulse_end_us"]) * 1e-6
        first = s["layers"][0]
        self.covered = not first.get("electrons", first["material"].startswith(("Au", "Cu")))
        # runs made before the YAML input have no pulse counts; they were all 6000-pulse trains
        self.single = s.get("micropulses_per_macropulse", 6000) * s.get("number_of_macropulses", 1) == 1
        self.end = float(self.history["t"][-1])


def time_unit(span: float) -> tuple[float, str]:
    """Scale factor and unit name for an axis running to `span` seconds."""
    for scale, name in ((1e12, "ps"), (1e9, "ns")):
        if span * scale <= 2000:
            return scale, name
    return 1e6, "µs"


def time_label(t: float) -> str:
    scale, name = (1e12, "ps") if t < 1e-9 else (1e9, "ns") if t < 1e-6 else (1e6, "µs")
    return f"{t * scale:.3g} {name}"


def per_pulse_peak(t: np.ndarray, y: np.ndarray, period: float, t_end: float):
    """Highest value in each micropulse period up to t_end, then the raw curve."""
    during = t <= t_end
    index = np.floor(t[during] / period).astype(int)
    starts = np.flatnonzero(np.diff(index, prepend=-1))
    peak = np.maximum.reduceat(y[during], starts)
    return np.concatenate((t[during][starts] + period, t[~during])), np.concatenate((peak, y[~during]))


def log_kelvin_axis(ax, lo: float, hi: float) -> None:
    steps = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
    lo = max([k for k in steps if k <= lo], default=lo)  # extend the axis to the neighbouring ticks
    hi = min([k for k in steps if k >= hi], default=hi)
    ax.set_yscale("log")
    ticks = [k for k in steps if lo <= k <= hi]
    ax.set_yticks(ticks, labels=[f"{k:,}" for k in ticks])
    ax.minorticks_off()
    ax.set_ylim(lo, hi)


def surface_temperature(runs: list[Run], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    span = max(r.end if r.single else min(2 * r.train_end, r.end) for r in runs)
    scale, unit = time_unit(span)
    lo, hi, t_max = np.inf, 0.0, 0.0
    for run, colour in zip(runs, SERIES):
        h = run.history
        if run.single:
            t, T = h["t"], h["T_surface"]
        else:
            t, T = per_pulse_peak(h["t"], h["T_surface"], run.period, run.train_end)
        ax.plot(t * scale, T, color=colour, label=run.label)
        lo, hi, t_max = min(lo, T.min()), max(hi, T.max()), max(t_max, run.train_end)
    log_kelvin_axis(ax, 0.9 * lo, 1.3 * hi)
    ax.set_xlim(0, span * scale)
    ax.axvspan(0, t_max * scale, color=INK, alpha=0.035, linewidth=0)
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Temperature of the vacuum-facing surface (K)")
    ax.set_title("Surface temperature" if all(r.single for r in runs)
                 else "Surface temperature: peak in each micropulse period")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out / "surface_temperature.png")
    plt.close(fig)


def depth_profiles(runs: list[Run], out: Path) -> None:
    shown = runs[:4]
    fig, axes = plt.subplots(1, len(shown), figsize=(3.6 * len(shown) + 1.4, 3.9), sharey=True, squeeze=False)
    hi = 0.0
    for ax, run in zip(axes[0], shown):
        p = run.profiles
        x, edges = np.maximum(p["x"], 3e-10) * 1e9, p["edges"]
        colours = TIME_RAMP(np.linspace(0, 1, len(p["t"])))
        for k, (t, colour) in enumerate(zip(p["t"], colours)):
            ax.plot(x, p["Tl"][k], color=colour, label=time_label(t), linewidth=1.5)
            hi = max(hi, p["Tl"][k].max())
        ax.set_xscale("log")
        ax.set_xlim(0.3, x[-1])
        ax.minorticks_off()
        for e in edges[1:-1]:
            ax.axvline(e * 1e9, color=AXIS, linewidth=1.0)
        ax.set_title(run.label, fontsize=8.5, loc="left", wrap=True)
        ax.set_xlabel("Depth below the surface (nm)")
    log_kelvin_axis(axes[0][0], 0.9 * min(r.profiles["Tl"].min() for r in shown), 1.3 * hi)
    axes[0][0].set_ylabel("Lattice temperature (K)")
    axes[0][-1].legend(title="Time", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "depth_profiles.png")
    plt.close(fig)


def first_pulses(runs: list[Run], out: Path) -> None:
    shown = runs[:4]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.0 * len(shown) + 1.6, 3.7), squeeze=False)
    for ax, run in zip(axes[0], shown):
        h = run.history
        window = run.end if run.single else 5 * run.period
        scale, unit = time_unit(window)
        sel = h["t"] <= 1.02 * window
        series = [("First metal, electrons", "Te_gold_front", SERIES[0]), ("First metal, lattice", "Tl_gold_front", SERIES[1])]
        if run.covered:
            series.append(("Vacuum-facing surface", "T_surface", SERIES[2]))
        for label, key, colour in series:
            ax.plot(h["t"][sel] * scale, h[key][sel], color=colour, label=label)
        ax.set_xlabel(f"Time ({unit})")
        ax.set_xlim(0, window * scale)
        ax.set_ylim(bottom=0)
        ax.set_title(run.label, fontsize=8.5, loc="left", wrap=True)
    axes[0][0].set_ylabel("Temperature (K)")
    axes[0][-1].legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / "first_pulses.png")
    plt.close(fig)


def plot_runs(folders: list[Path], out: Path | None = None) -> None:
    if len(folders) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per figure")
    runs = [Run(Path(f)) for f in folders]
    out = Path(out) if out is not None else Path(folders[0]).resolve().parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    surface_temperature(runs, out)
    depth_profiles(runs, out)
    first_pulses(runs, out)
    print(f"wrote {out}/surface_temperature.png, depth_profiles.png, first_pulses.png")
