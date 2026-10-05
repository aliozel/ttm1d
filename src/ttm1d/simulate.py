"""Run a case built by config.load_case and write its results.

results/<name>/
    input.yaml           the case file as written
    input_resolved.yaml  merged with its base file: every value the run used
    summary.json         peaks, absorbed energy, energy balance, desorption estimate
    history.npz          time series, every `history_every_steps` steps
    profiles.npz         temperature against depth at the times in `profile_times_us`
"""
from __future__ import annotations

import json
import time

import numpy as np
import yaml

from .config import Case
from .solver import TwoTemperature1D

GAS_CONSTANT = 8.314462618  # J/(mol K)
HISTORY = ("t", "T_surface", "T_top_max", "T_top_mean", "Tl_gold_front", "Te_gold_front", "Te_max",
           "Tl_gold_back", "energy_absorbed", "energy_stored", "Tl_max", "Tl_stack_mean")
# The "gold" columns refer to the first layer with electrons (the top layer if there is none).


def run(case: Case) -> dict:
    out = case.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.yaml").write_text(case.source)
    (out / "input_resolved.yaml").write_text(yaml.safe_dump(case.config, sort_keys=False, allow_unicode=True))

    pulse, layers = case.pulse, case.layers
    sim = TwoTemperature1D(layers, case.optics.poynting, pulse, T0=case.T0, front=case.front,
                           back=case.back, back_conductance=case.back_conductance)
    print(f"case {case.name}: {' | '.join(l.material.name for l in layers)}")
    print(f"  reflectance {case.optics.R:.4f}; incident fluence {pulse.fluence:.3f} J/m^2 per micropulse, "
          f"{pulse.macropulses} x {pulse.n_pulses} micropulses")
    for l, n, a in zip(layers, case.indices, sim.layer_absorptance):
        print(f"  {l.material.name:24s} {l.thickness*1e9:12.1f} nm   n = {n.real:.3f} + {n.imag:.3f}i   absorbs {100*a:.3f} %")
    print(f"  mesh: {len(sim.x)} nodes")

    times = pulse.step_times(case.end_time, **case.step_options)
    train_end = pulse.end
    profile_at = [t for t in case.profile_times if t <= case.end_time * (1 + 1e-9)]

    top = sim.layer_slice(0)  # the layer facing vacuum
    metal = next((i for i, l in enumerate(layers) if l.material.has_electrons), 0)  # first metal layer
    i_front = int(np.argmin(np.abs(sim.x - sim.edges[metal])))
    i_back = int(np.argmin(np.abs(sim.x - sim.edges[metal + 1])))
    w_top, w_all = np.gradient(sim.x[top]), np.gradient(sim.x)
    e_bind = np.array(case.binding_energies)
    # probe-weighted temperatures: mean of T(x) with weight exp(-x / decay length)
    probe_w = [np.exp(-sim.x / d) * w_all for d in case.probe_lengths]
    probe_names = [f"{s}_probe_{d*1e9:g}nm" for d in case.probe_lengths for s in ("Te", "Tl")]
    names = HISTORY + tuple(probe_names)

    rows, peaks, at_train_end = [], {}, None
    desorbed = np.zeros((2, len(e_bind)))  # time integral of the desorption rate: pulse train, whole run
    profiles, wall = [], time.time()
    for k, t1 in enumerate(times):
        t_prev = sim.t
        sim.advance(t1)
        Te, Tl = sim.temperatures()
        Te_max = Te[sim.node_has_electrons].max() if sim.node_has_electrons.any() else Tl.max()
        row = (t1, Tl[0], Tl[top].max(), np.sum(Tl[top] * w_top) / w_top.sum(), Tl[i_front], Te[i_front],
               Te_max, Tl[i_back], sim.energy_in, sim.energy_gain(),
               Tl.max(), np.sum(Tl * w_all) / w_all.sum())
        row += tuple(v for w in probe_w for v in (np.sum(Te * w) / w.sum(), np.sum(Tl * w) / w.sum()))
        for name, value in zip(names, row):
            peaks[name] = max(peaks.get(name, -np.inf), value)
        if k % case.history_every == 0 or k == len(times) - 1:
            rows.append(row)
        if t1 <= train_end:
            at_train_end = row
        rate = case.attempt_frequency * np.exp(-e_bind / (GAS_CONSTANT * Tl[0])) * (t1 - t_prev)
        desorbed[1] += rate
        if t1 <= train_end:
            desorbed[0] += rate
        while profile_at and t1 >= profile_at[0] * (1 - 1e-9):
            profiles.append((profile_at.pop(0), Te, Tl))
        if (k + 1) % case.progress_every == 0:
            print(f"  t = {t1*1e6:10.3f} us  surface {Tl[0]:7.2f} K  first metal {Tl[i_front]:7.2f} K  "
                  f"[{time.time()-wall:5.0f} s]", flush=True)

    hist = np.array(rows)
    end = dict(zip(names, at_train_end if at_train_end is not None else rows[-1]))
    balance = (end["energy_stored"] - end["energy_absorbed"]) / end["energy_absorbed"]
    summary = {
        "name": case.name,
        "label": case.label,
        "layers": [{"material": l.material.name, "thickness_nm": l.thickness * 1e9, "n": n.real, "k": n.imag,
                    "electrons": bool(l.material.has_electrons), "absorbed_fraction": float(a)}
                   for l, n, a in zip(layers, case.indices, sim.layer_absorptance)],
        "reflectance": float(case.optics.R),
        "fluence_per_micropulse_J_m2": pulse.fluence,
        "micropulse_period_ns": pulse.period * 1e9,
        "micropulses_per_macropulse": pulse.n_pulses,
        "number_of_macropulses": pulse.macropulses,
        "pulse_train_end_us": train_end * 1e6,
        "absorbed_per_macropulse_J_m2": float(end["energy_absorbed"]) / pulse.macropulses,
        "peak_surface_temperature_K": float(peaks["T_surface"]),
        "peak_top_layer_temperature_K": float(peaks["T_top_max"]),
        "peak_gold_lattice_temperature_K": float(peaks["Tl_gold_front"]),  # front face of the first metal layer
        "peak_electron_temperature_K": float(peaks["Te_max"]),
        "peak_lattice_temperature_anywhere_K": float(peaks["Tl_max"]),
        "stack_mean_temperature_at_macropulse_end_K": float(end["Tl_stack_mean"]),
        "surface_temperature_at_end_K": float(rows[-1][1]),
        "stored_minus_absorbed_fraction_at_macropulse_end": float(balance),  # negative: heat left through a boundary
        "thermal_desorption_estimate": {
            "note": f"first-order, nu = {case.attempt_frequency:g} 1/s, surface lattice temperature; "
                    "fraction of adsorbate lost",
            "binding_energy_kJ_mol": (e_bind / 1e3).tolist(),
            "fraction_desorbed_during_macropulse": (1 - np.exp(-desorbed[0])).tolist(),
            "fraction_desorbed_whole_run": (1 - np.exp(-desorbed[1])).tolist(),
        } if len(e_bind) else None,
        "temperatures": case.temperatures,
        "steps": len(times),
        "wall_time_s": time.time() - wall,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "history.npz", **{n: hist[:, j].astype(np.float64 if n == "t" else np.float32)
                                                for j, n in enumerate(names)})
    np.savez_compressed(out / "profiles.npz", x=sim.x, edges=sim.edges, t=np.array([p[0] for p in profiles]),
                        Te=np.array([p[1] for p in profiles]), Tl=np.array([p[2] for p in profiles]))
    print(f"  peak surface temperature {summary['peak_surface_temperature_K']:.1f} K, "
          f"peak electron temperature {summary['peak_electron_temperature_K']:.1f} K, "
          f"energy balance {balance:+.1e}, {summary['wall_time_s']:.0f} s")
    print(f"  wrote {out}")
    return summary
