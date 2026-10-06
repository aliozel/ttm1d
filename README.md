# 1D two-temperature model of pulsed-laser heating (DOLFINx)

Heating of a stack of layers by laser pulses, from a single femtosecond pulse to a train of
thousands, with electrons and lattice at separate temperatures (or one temperature). Any
number of layers; materials are either built-in laws or defined in the case file by formulas.
Implicit time stepping in DOLFINx and exact energy bookkeeping.

## Install

The solver needs DOLFINx, PETSc and MPI from conda-forge; the package itself installs with pip.

```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx petsc4py mpi4py numpy pyyaml matplotlib sympy
conda activate fenicsx-env
cd fenicsx1D
pip install -e .              # installs the `ttm1d` command; edits to src/ttm1d take effect at once
```

**macOS 27 and newer:** the linker that conda-forge ships (ld64 956) cannot read this macOS
version's system libraries, so compiling the solver's equations fails with
`libm.tbd: unknown architecture` when the environment is activated. Make the environment use
the system compiler instead:

```bash
conda env config vars set CC=/usr/bin/clang -n fenicsx-env    # undo: conda env config vars unset CC -n fenicsx-env
```

## Run

```bash
conda activate fenicsx-env
cd ttm1d
ttm1d check cases/ntmpy_fig5/pt_si.yaml                       # read and check the file, show what would run
ttm1d run cases/ntmpy_fig5/pt_si.yaml cases/ntmpy_fig5/pt.yaml  # run them one after another
ttm1d plot results/ntmpy_fig5_pt_si results/ntmpy_fig5_pt     # figures into results/figures/
```

`python -m ttm1d ...` does the same. Every input is in the case file; the command takes only
file names. Each run writes to `<results_folder>/<name>/`; the Fig. 5 cases set
`results_folder: ../../results`, i.e. `results/` at the top of the repository.

Several cases at once:

```bash
ls cases/ntmpy_fig5/*.yaml | xargs -P 4 -n 1 ttm1d run
```

Run one case of each kind alone first: the solver compiles its equations on the first run of
each new set of material parameters, and several processes compiling the same equations at
once time out. On a laptop, keep the machine awake (`caffeinate -ims ...` on macOS).

From Python:

```python
from ttm1d.config import load_case
from ttm1d.simulate import run
summary = run(load_case("cases/ntmpy_fig5/pt_si.yaml"))
```

## Case files

A case file can name a base file and list only what differs, for example:

```yaml
base: base.yaml
name: co3_ice_p_0.0059
label: "CO:H2O ice on gold, 3 um, 36 mJ, 0.0059 cm2, p-pol"   # used in plots

laser:
  wavelength_um: 3.0
  macropulse_energy_mJ: 36        # or micropulse_fluence_J_m2 instead of energy and area
  spot_area_cm2: 0.0059

stack:                            # vacuum side first
  - {material: amorphous_ice, thickness_nm: 87.7}
  - {material: gold_film, thickness_nm: 100}
  - {material: copper_ofhc, thickness_nm: 3.175e6}
```

- **Merging.** Sections are merged key by key with the base file; a list (`stack`,
  `profile_times_us`) and a `refractive_index` replace the base one as a whole. A case can itself be the base of another.
- **Units.** Every quantity carries its unit in its key name (`thickness_nm`,
  `spot_area_cm2`, `coupling_room_temperature_W_m3_K`) and is converted to SI on reading.
- **Checks.** A missing key, a misspelt key, a value of the wrong type, a wavelength outside an
  optical table, or both an energy and a fluence stop the run with a message naming the key.
- **Paths** (optical tables, `results_folder`) are relative to the YAML file they are written in.

### Sections

| Section | Contents |
|---|---|
| `temperatures` | `2`: electrons and lattice, in layers whose material has electrons; `1`: one temperature everywhere (each layer's electron and lattice heat capacities and conductivities are added) |
| `laser` | wavelength, angle, polarisation, micropulse width and spacing, micropulses per macropulse, number and spacing of macropulses, energy and spot area (or fluence), absorption model, centre of the first micropulse (`first_micropulse_time_ps`, default 3 × width) |
| `materials` | the material library (below) |
| `stack` | the layers, vacuum side first: `material`, `thickness_nm`, and any material key to override for that layer |
| `boundaries` | initial temperature; vacuum side `insulated` or `fixed`; far side `fixed`, `insulated` or `conductance` (with `back_conductance_W_m2_K`) |
| `numerics` | simulated time, time step across a micropulse, growth of the step between micropulses, maximum step |
| `output` | results folder, how often to save the time series, the times of the depth profiles, probe-weighted temperatures (`probe_decay_lengths_nm`), the desorption estimate (optional) |

### Materials

Each entry in `materials` names a temperature-dependent law (`model`, implemented in
`src/ttm1d/materials.py`) and gives its parameters; a layer can override any of them.

| Model | What it describes | Fixed in the code |
|---|---|---|
| `custom` | any material: heat capacities and conductivities given as numbers or formulas in T (below) | — |
| `amorphous_ice` | insulator; light absorbed here heats the lattice | heat capacity: fit to IAPWS ice Ih data |
| `gold_film` | metal: Debye lattice, Wiedemann–Franz conductivity with residual resistivity, T⁵ coupling at low T | — |
| `copper_ofhc` | metal; NIST cryogenic fits | NIST heat capacity and conductivity fits |
| `power_law_metal` | the laws of SI Table S1: C ∝ T, C ∝ T³, k ∝ T, constant coupling | — |
| `thermal_contact` | a thin layer standing in for a boundary conductance min(aT³, G_max) | takes heat capacity and refractive index from the layer above |

#### Custom materials

A `custom` material gives a `lattice` and, for a material with free electrons, an `electron`
part, each with a heat capacity (`heat_capacity_J_m3_K`, or `heat_capacity_J_kg_K` together
with `density_kg_m3`) and a `thermal_conductivity_W_m_K`. Each value is a number or a formula
in `T`, the temperature of that part. Formulas may use `+ - * / **`, `exp`, `log`, `sqrt`,
`abs`, `min`, `max` and `where(condition, a, b)`. The electron–lattice coupling is
`coupling_W_m3_K` (a constant G), or `coupling_room_temperature_W_m3_K` with
`coupling_low_temperature_W_m3_K5` (the built-in T⁵ law joined to G at room temperature).
Without an `electron` part the material is an insulator: light absorbed in it heats the lattice.

```yaml
materials:
  platinum:                                   # Alber et al. 2021, Table 1
    model: custom
    refractive_index: {n: 1.7176, k: 2.844}
    electron: {heat_capacity_J_m3_K: "740 * T", thermal_conductivity_W_m_K: 72}
    lattice:  {heat_capacity_J_m3_K: 2.78e6, thermal_conductivity_W_m_K: 72}
    coupling_W_m3_K: 2.5e17
    mesh: {smallest_cell_nm: 0.1, largest_cell_nm: 0.4, growth_factor: 1.2}
  silicon:
    model: custom
    ...
    lattice:
      heat_capacity_J_m3_K: 1.6e6
      thermal_conductivity_W_m_K: "where(T <= 120.7, 9*T**3*(0.016*exp(-0.05*T) + exp(-0.14*T)), 1.3e6*T**-1.6)"
```

Heat capacities are integrated exactly (with sympy) into the energy density the solver
conserves, so a heat capacity must stay finite at 0 K. `ttm1d check` reports a formula it cannot
read or integrate, naming the key.

Two keys that may be unfamiliar:

- `refractive_index` is the complex index n + ik that sets reflection and absorption. Give
  either `{table: ../data/<file>}`, a table of wavelength (µm), n and k interpolated at the laser
  wavelength, or constants `{n: 1.47, k: 1.95}`. For amorphous ice the table is for compact ice
  and the pores are mixed in (Maxwell Garnett) from `density_kg_m3 / compact_density_kg_m3`.
- `residual_resistivity_ratio` (often written RRR) is the electrical resistivity at room
  temperature divided by that at 4 K. It measures purity and grain size, and sets the thermal
  conductivity of a metal at low temperature: about 3 for a thin evaporated gold film, about
  100 for OFHC copper. For copper it picks one of the NIST fits (50, 100, 150, 300 or 500).

## Outputs

`<results_folder>/<name>/` contains:

- `input.yaml`: the case file as written; `input_resolved.yaml`: merged with its base file,
  i.e. every value the run used.
- `summary.json`: reflectance and absorbed fraction per layer, fluence, absorbed energy per
  macropulse, peak temperatures (from every time step), energy balance, desorption estimate.
- `history.npz`: time series every `history_every_steps` steps: `t`, surface temperature,
  maximum and mean of the top layer, lattice and electron temperature at the front face of the
  first metal layer (keys `Tl_gold_front`, `Te_gold_front`), lattice temperature at its back
  face, absorbed and stored energy (J/m²), maximum and mean lattice temperature of the stack;
  with `probe_decay_lengths_nm`, also `Te_probe_<d>nm` and `Tl_probe_<d>nm`, the temperatures
  averaged over depth with weight exp(−x/d), as a reflectivity probe of that penetration depth sees them.
- `profiles.npz`: depth `x`, layer `edges`, and `Te`, `Tl` at the times in `profile_times_us`
  (`pulse_train_end` stands for the end of the last micropulse).

Of the case files `cases/ice_gold_copper/` and `cases/ntmpy_fig5/` are in the repository, and of the
results only the two Fig. 5 runs, `results/ntmpy_fig5_pt/` and `results/ntmpy_fig5_pt_si/`.
Runs made before the YAML input (3–4 Oct 2026) recorded their settings under `inputs` in
`summary.json` instead of `input*.yaml`; `ttm1d plot` reads both.

## Plot

`ttm1d plot` draws figures from finished runs; it runs nothing. Give it up to 8 folders
written by `ttm1d run`:

```bash
ttm1d plot results/ntmpy_fig5_pt_si results/ntmpy_fig5_pt             # figures into results/figures/
ttm1d plot results/ntmpy_fig5_pt_si results/ntmpy_fig5_pt --out figs  # or into a folder of your choice
```

By default the figures go to `figures/` next to the first result folder. Each run is labelled
with the `label` of its case file. Three PNG files are written:

| File | What it shows |
|---|---|
| `surface_temperature.png` | Lattice temperature of the vacuum-facing surface against time, all runs on one plot, temperature on a log axis. For a pulse train, the highest value in each micropulse period, then the cooling after the train, up to twice the train length; the time the laser is on is shaded. For a single pulse, the curve itself. |
| `depth_profiles.png` | Lattice temperature against depth (both log axes) at the times in `profile_times_us`, one panel per run, layer boundaries marked. |
| `first_pulses.png` | Electron and lattice temperature at the front face of the first metal layer, and the vacuum-facing surface when an insulator covers it, over the first five micropulses (the whole run for a single pulse); one panel per run. |

The time axis is in ps, ns or µs, whichever suits the length plotted. `depth_profiles.png`
and `first_pulses.png` show the first four runs given.

## Files

- `pyproject.toml` — package definition; the `ttm1d` command.
- `src/ttm1d/cli.py` — the `ttm1d check | run | plot` command.
- `src/ttm1d/config.py` — reads and checks a case file and builds the solver objects.
- `src/ttm1d/simulate.py` — time loop and output.
- `src/ttm1d/solver.py` — mesh, variational form, time stepping, pulse train.
- `src/ttm1d/materials.py` — material laws and their sources; formula-defined materials.
- `src/ttm1d/formulas.py` — reads property formulas and turns them into NumPy and UFL.
- `src/ttm1d/optics.py` — transfer-matrix absorption; also NTMpy's Lambert–Beer source.
- `src/ttm1d/plot.py` — figures from finished runs.
- `cases/ice_gold_copper/` — CO:H₂O ice on a gold film on copper at 3, 4.68 and 12 µm, and their
  `base.yaml` with the material library.
- `cases/ntmpy_fig5/` — the two stacks of Alber et al. 2021, Fig. 5 (Pt on Si, free-standing Pt).
- `verification/ntmpy_fig5/` — Fig. 5 solved with NTMpy and compared with this code (scripts,
  NTMpy output and `fig5_compare.png`).
- `data/` — optical constants with citations in the file headers.

## Verification against NTMpy (Alber et al. 2021, Fig. 5)

10 nm Pt on 100 µm Si, and free-standing 10 nm Pt, heated by one 400 nm pulse (p, 45°,
6 mJ/cm²), materials of Table 1 written as `custom` materials (`cases/ntmpy_fig5/`). NTMpy's
`FWHM` is a half width, so its `s.FWHM = 0.1e-12` is a 0.2 ps pulse, and the case files use
0.2 ps. The transfer-matrix reflectance and transmittance agree with NTMpy's to six digits
(Pt/Si: R = 0.4673, Pt absorbs 16.5 %, Si 36.7 %; free Pt: R = 0.1759, T = 0.374).

| | ttm1d | NTMpy |
|---|---|---|
| free Pt: peak of T_e weighted over 11.2 nm | 2644 K at 1.21 ps | 2647 K at 1.21 ps |
| free Pt: lattice at the surface, 7 ps | 990.6 K | 991.3 K |
| Pt/Si: peak of weighted T_e | 1270 K at 1.11 ps | 1282 K at 1.12 ps |
| Pt/Si: weighted T_e, normalised, at 2 / 3 / 7 ps | 0.449 / 0.228 / 0.174 | 0.455 / 0.240 / 0.185 |
| Pt/Si: lattice at the surface, 7 ps | 484.0 K | 498.1 K |

Halving the time step and the cell sizes changes the ttm1d values by less than 1.5 K. Each run
takes about 2 s.

## Limits

- One-dimensional, using the mean fluence over the spot: a Gaussian beam is hotter at its centre,
  and lateral conduction in the copper, which this model lacks, cools the surface after the macropulse.
- With a fixed far face the stack is back at the initial temperature within about 50 µs, so
  successive macropulses (100 ms apart) do not interact. With `back: conductance` they do, but
  the heat then stays under the spot, so the build-up is overestimated.
- Optical constants of gold and copper are room-temperature values.
- No sublimation, desorption cooling or phase change; temperatures above about 150 K in the ice
  mean the film would not survive, not that it reaches that temperature.
