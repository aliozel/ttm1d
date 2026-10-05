"""1D two-temperature model of pulsed-laser heating of layered stacks, solved with DOLFINx.

Command line: `ttm1d run | check | plot` (see `ttm1d --help`).  From Python:

    from ttm1d.config import load_case
    from ttm1d.simulate import run
    run(load_case("cases/ntmpy_fig5/pt_si.yaml"))
"""
__version__ = "0.1.0"
