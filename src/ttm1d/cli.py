"""The `ttm1d` command.

    ttm1d check cases/ntmpy_fig5/pt_si.yaml      read and check case files, show what would run
    ttm1d run cases/ntmpy_fig5/pt_si.yaml [...]  run case files one after another
    ttm1d plot results/ntmpy_fig5_pt_si results/ntmpy_fig5_pt [--out results/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ttm1d", description="1D two-temperature model of pulsed-laser heating. "
                                  "Every input is in a YAML case file; see the README.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    check = commands.add_parser("check", help="read and check case files and show what would run")
    check.add_argument("cases", nargs="+", type=Path, metavar="CASE.yaml")
    run = commands.add_parser("run", help="run case files one after another")
    run.add_argument("cases", nargs="+", type=Path, metavar="CASE.yaml")
    plot = commands.add_parser("plot", help="figures from finished runs")
    plot.add_argument("results", nargs="+", type=Path, metavar="RESULT_FOLDER",
                      help="folders written by `ttm1d run`, e.g. results/ntmpy_fig5_pt_si (at most 8)")
    plot.add_argument("--out", type=Path, default=None,
                      help="folder for the figures (default: figures/ next to the first result folder)")
    args = parser.parse_args(argv)

    if args.command == "plot":
        from .plot import plot_runs
        plot_runs(args.results, args.out)
        return

    from .config import CaseError, load_case
    try:
        cases = [load_case(path) for path in args.cases]  # check every file before running any
    except CaseError as err:
        sys.exit(f"case file error in {err}")
    if args.command == "check":
        for case in cases:
            describe(case)
        return
    from .simulate import run as run_case
    for case in cases:
        run_case(case)


def describe(case) -> None:
    """What a case will run: optics, pulse train, boundaries, numerics, output."""
    pulse, opt = case.pulse, case.optics
    edges = [0.0]
    for layer in case.layers:
        edges.append(edges[-1] + layer.thickness)
    flux = [float(v) for v in opt.poynting(edges)]
    absorbed = [flux[i] - flux[i + 1] for i in range(len(case.layers))]
    steps = len(pulse.step_times(case.end_time, **case.step_options))
    print(f"{case.name}: {case.label}")
    print(f"  results    {case.results_dir}")
    print(f"  laser      {pulse.fluence:.4g} J/m^2 per micropulse, {pulse.macropulses} x {pulse.n_pulses} micropulses "
          f"of {pulse.fwhm*1e12:g} ps every {pulse.period*1e9:g} ns; reflectance {opt.R:.4f}, transmitted {flux[-1]:.4f}")
    print("  stack      layer                         thickness        n + ik         absorbs")
    for layer, index, a in zip(case.layers, case.indices, absorbed):
        d = layer.thickness
        size = f"{d*1e9:g} nm" if d < 1e-6 else f"{d*1e6:g} um" if d < 1e-3 else f"{d*1e3:g} mm"
        print(f"             {layer.material.name:28s} {size:>12s}  "
              f"{index.real:7.3f} + {index.imag:7.3f}i  {100*a:7.3f} %")
    sink = f" ({case.back_conductance:g} W/m^2/K)" if case.back_conductance is not None else ""
    print(f"  boundaries start at {case.T0:g} K; vacuum side {case.front}; far side {case.back}{sink}")
    print(f"  numerics   {case.end_time*1e6:g} us in {steps} time steps")
    print(f"  output     history every {case.history_every} steps; "
          f"{len(case.profile_times)} profiles from {case.profile_times[0]*1e6:g} to {case.profile_times[-1]*1e6:g} us")
