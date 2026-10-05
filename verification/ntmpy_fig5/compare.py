"""Compare ttm1d with NTMpy on Fig. 5 of Alber et al., Comput. Phys. Commun. 265, 107990 (2021).

    python run_ntmpy.py substrate; python run_ntmpy.py substrate_fine; python run_ntmpy.py free
    ttm1d run ../../cases/ntmpy_fig5/pt_si.yaml ../../cases/ntmpy_fig5/pt.yaml
    python compare.py [RESULTS_FOLDER]          # default ../../results

Writes fig5_compare.png and prints the numbers compared.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "../../results"
PROBE = 11.2e-9  # optical decay length of Pt at 400 nm, lambda / (4 pi k)
T0 = 300.0


def probe_mean(x, T):
    """Mean of T(x) weighted by exp(-x / PROBE), as ttm1d writes it."""
    w = np.exp(-x / PROBE) * np.gradient(x)
    return T @ w / w.sum()


def ntmpy(which):
    d = np.load(HERE / f"ntmpy_{which}.npz")
    x, t, Te, Tl = d["x"], d["t"], d["Te"], d["Tl"]
    return dict(t=t, Te_surface=Te[:, 0], Tl_surface=Tl[:, 0], Te_probe=probe_mean(x, Te), x=x, Te=Te, Tl=Tl)


def ttm1d(name):
    h, p = np.load(RESULTS / name / "history.npz"), np.load(RESULTS / name / "profiles.npz")
    return dict(t=np.r_[0.0, h["t"]], Te_surface=np.r_[T0, h["Te_gold_front"]], Tl_surface=np.r_[T0, h["Tl_gold_front"]],
                Te_probe=np.r_[T0, h["Te_probe_11.2nm"]], x=p["x"], profile_t=p["t"], Te=p["Te"], Tl=p["Tl"])


runs = {
    "ttm1d, Pt on Si": (ttm1d("ntmpy_fig5_pt_si"), dict(color="tab:blue", lw=2)),
    "NTMpy, Pt on Si (Si in 7 layers)": (ntmpy("substrate_fine"), dict(color="black", ls="--", lw=1.2)),
    "NTMpy, Pt on Si (addSubstrate, as in the paper)": (ntmpy("substrate"), dict(color="0.55", ls=":", lw=1.5)),
    "ttm1d, free-standing Pt": (ttm1d("ntmpy_fig5_pt"), dict(color="tab:red", lw=2)),
    "NTMpy, free-standing Pt": (ntmpy("free"), dict(color="darkred", ls="--", lw=1.2)),
}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
print(f"{'run':50s} {'peak Te_probe':>13s} {'at':>7s} {'norm 2 ps':>9s} {'3 ps':>6s} {'7 ps':>6s} "
      f"{'Te(0) max':>9s} {'Tl(0) 7 ps':>10s}")
for label, (r, style) in runs.items():
    t_ps, rise = r["t"] * 1e12, r["Te_probe"] - T0
    k = int(np.argmax(rise))
    norm = rise / rise[k]
    at = lambda s: np.interp(s, t_ps, norm)
    print(f"{label:50s} {r['Te_probe'][k]:11.1f} K {t_ps[k]:5.2f}ps {at(2):9.3f} {at(3):6.3f} {at(7):6.3f} "
          f"{r['Te_surface'].max():7.1f} K {np.interp(7, t_ps, r['Tl_surface']):8.1f} K")
    ax[0].plot(t_ps, norm, label=label, **style)
    ax[1].plot(t_ps, r["Te_surface"], label=label, **style)
    ax[1].plot(t_ps, r["Tl_surface"], **{**style, "lw": style["lw"] * 0.7, "alpha": 0.6})

# depth profiles of the Pt/Si stack
ours, theirs = runs["ttm1d, Pt on Si"][0], runs["NTMpy, Pt on Si (Si in 7 layers)"][0]
for when, shade in zip((1.2, 2.0, 7.0), ("tab:orange", "tab:green", "tab:purple")):
    i = int(np.argmin(np.abs(ours["profile_t"] * 1e12 - when)))
    j = int(np.argmin(np.abs(theirs["t"] * 1e12 - when)))
    ax[2].plot(ours["x"] * 1e9, ours["Te"][i], color=shade, lw=2, label=f"Te, {when:g} ps (ttm1d)")
    ax[2].plot(ours["x"] * 1e9, ours["Tl"][i], color=shade, lw=1, alpha=0.7)
    ax[2].plot(theirs["x"] * 1e9, theirs["Te"][j], color="black", ls="--", lw=1)
    ax[2].plot(theirs["x"] * 1e9, theirs["Tl"][j], color="black", ls=":", lw=1)

ax[0].set(xlim=(0, 7), xlabel="time (ps)", ylabel="normalised probe-weighted $T_e$",
          title="Fig. 5: $T_e$ weighted by exp(-x / 11.2 nm), normalised")
ax[1].set(xlim=(0, 7), xlabel="time (ps)", ylabel="surface temperature (K)",
          title="surface $T_e$ (thick) and $T_l$ (thin)")
ax[2].set(xlim=(0, 300), xlabel="depth (nm)", ylabel="temperature (K)",
          title="Pt on Si: $T_e$ (thick), $T_l$ (thin); NTMpy dashed/dotted")
ax[2].axvline(10, color="0.7", lw=0.8)
ax[0].legend(fontsize=7.5, loc="upper right")
ax[2].legend(fontsize=7.5)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig5_compare.png", dpi=150)
print(f"wrote {HERE / 'fig5_compare.png'}")
