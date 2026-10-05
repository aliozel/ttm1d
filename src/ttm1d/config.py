"""Read a case file (YAML) and build the objects the solver needs.

A case file may start with `base: <file>`; the base is read first and the case is merged
into it (mappings key by key, everything else replaced), so a case lists only what differs.
Every quantity carries its unit in its key name and is converted to SI here.  A missing,
misspelt or unused key stops the run with a message naming it.  File paths (optical tables,
the results folder) are relative to the YAML file they are written in.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import formulas
from . import materials as M
from . import optics
from .solver import Layer, PulseTrain


# Parameters each material model takes, as (key in the case file, keyword in materials.py, factor to SI).
MODELS = {
    "amorphous_ice": [
        ("density_kg_m3", "density", 1.0),
        ("compact_density_kg_m3", "compact_density", 1.0),
        ("thermal_conductivity_scale", "k_scale", 1.0),
        ("thermal_conductivity_floor_W_m_K", "k_floor", 1.0),
        ("phonon_velocity_m_s", "velocity", 1.0),
        ("phonon_mean_free_path_nm", "mean_free_path", 1e-9),
    ],
    "gold_film": [
        ("density_kg_m3", "density", 1.0),
        ("molar_mass_g_mol", "molar_mass", 1e-3),
        ("electron_heat_capacity_J_m3_K2", "gamma", 1.0),
        ("debye_temperature_K", "theta_d", 1.0),
        ("resistivity_295K_ohm_m", "rho_295", 1.0),
        ("resistivity_debye_temperature_K", "theta_r", 1.0),
        ("residual_resistivity_ratio", "rrr", 1.0),
        ("lattice_thermal_conductivity_W_m_K", "k_lattice", 1.0),
        ("coupling_room_temperature_W_m3_K", "g_inf", 1.0),
        ("coupling_low_temperature_W_m3_K5", "sigma", 1.0),
        ("fermi_velocity_m_s", "v_fermi", 1.0),
        ("electron_electron_scattering_per_K2_per_s", "a_ee", 1.0),
    ],
    "copper_ofhc": [
        ("density_kg_m3", "density", 1.0),
        ("molar_mass_g_mol", "molar_mass", 1e-3),
        ("electron_heat_capacity_J_m3_K2", "gamma", 1.0),
        ("debye_temperature_K", "theta_d", 1.0),
        ("residual_resistivity_ratio", "rrr", 1.0),
        ("lattice_thermal_conductivity_W_m_K", "k_lattice", 1.0),
        ("coupling_room_temperature_W_m3_K", "g_inf", 1.0),
        ("coupling_low_temperature_W_m3_K5", "sigma", 1.0),
        ("fermi_velocity_m_s", "v_fermi", 1.0),
        ("electron_electron_scattering_per_K2_per_s", "a_ee", 1.0),
    ],
    "power_law_metal": [
        ("density_kg_m3", "rho", 1.0),
        ("electron_heat_capacity_J_kg_K2", "ce", 1.0),
        ("lattice_heat_capacity_J_kg_K4", "cl", 1.0),
        ("thermal_conductivity_W_m_K2", "kcoef", 1.0),
        ("coupling_W_m3_K", "G", 1.0),
    ],
    "thermal_contact": [
        ("conductance_coefficient_W_m2_K4", "a", 1.0),
        ("max_conductance_W_m2_K", "g_max", 1.0),
    ],
    "custom": [],  # properties given as numbers or formulas; read by custom_material()
}
CUSTOM_KEYS = ("density_kg_m3", "electron", "lattice", "coupling_W_m3_K",
               "coupling_room_temperature_W_m3_K", "coupling_low_temperature_W_m3_K5")
COPPER_RRR = (50, 100, 150, 300, 500)  # the NIST conductivity fits that exist
TRAIN_END = "pulse_train_end"           # may appear in output.profile_times_us


class CaseError(ValueError):
    """A problem in the case file."""


class _Loader(yaml.SafeLoader):
    """Safe YAML that also reads 2.2e16 and 1e5 as numbers (plain YAML 1.1 wants 2.2e+16)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+|[0-9][0-9_]*)[eE][-+]?[0-9]+$"),
    list("-+0123456789."))


# ------------------------------------------------------------------------------------------
# Reading and checking
# ------------------------------------------------------------------------------------------
def read_yaml(path: Path) -> dict:
    """The case file merged into its base file(s)."""
    try:
        data = yaml.load(path.read_text(), Loader=_Loader)
    except (OSError, yaml.YAMLError) as err:
        raise CaseError(f"cannot read {path}: {err}") from None
    if not isinstance(data, dict):
        raise CaseError(f"{path}: expected a mapping of sections at the top level")
    anchor_paths(data, path.parent)
    base = data.pop("base", None)
    return merge(read_yaml((path.parent / base).resolve()), data) if base else data


def anchor_paths(data: dict, folder: Path) -> None:
    """Make the file paths written in one YAML file absolute, relative to that file's folder."""
    def anchor(mapping: dict, key: str) -> None:
        value = mapping.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            mapping[key] = str((folder / value).resolve())

    entries = []
    if isinstance(data.get("materials"), dict):
        entries += list(data["materials"].values())
    if isinstance(data.get("stack"), list):
        entries += data["stack"]
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("refractive_index"), dict):
            anchor(entry["refractive_index"], "table")
    if isinstance(data.get("output"), dict):
        anchor(data["output"], "results_folder")


REPLACED_WHOLE = ("refractive_index",)  # a table and constants n, k must not mix


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACED_WHOLE:
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def integer(value) -> int:
    if isinstance(value, bool) or float(value) != int(value):
        raise ValueError
    return int(value)


def number(value) -> float:
    if isinstance(value, bool):
        raise ValueError
    return float(value)


class Keys:
    """Hands out the keys of one mapping in the case file.  Keys outside `allowed`, or left
    unread when done() is called, are reported as unknown (usually a misspelling)."""

    def __init__(self, data, where: str, allowed=None):
        if not isinstance(data, dict):
            raise CaseError(f"{where}: expected a mapping, found {data!r}")
        self.data, self.where, self.known = data, where, set()
        if allowed is not None:
            unknown = sorted(set(data) - set(allowed))
            if unknown:
                raise CaseError(f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")

    def __call__(self, key: str, kind=number, choices=None):
        self.known.add(key)
        if key not in self.data:
            raise CaseError(f"{self.where}: '{key}' is missing")
        value = self.data[key]
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise CaseError(f"{self.where}.{key}: {value!r} is not a valid {kind.__name__}") from None
        if choices is not None and value not in choices:
            raise CaseError(f"{self.where}.{key}: {value!r} must be one of {', '.join(map(str, choices))}")
        return value

    def optional(self, key: str, kind=number, choices=None):
        self.known.add(key)
        return self(key, kind, choices) if key in self.data else None

    def done(self) -> None:
        unknown = sorted(set(self.data) - self.known)
        if unknown:
            raise CaseError(f"{self.where}: unknown key(s) {', '.join(unknown)}; "
                            f"the keys read here are {', '.join(sorted(self.known))}")


# ------------------------------------------------------------------------------------------
# The case
# ------------------------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    label: str
    source: str                       # the case file as written
    config: dict                      # merged with its base: every value the run used
    layers: list[Layer]
    indices: list[complex]            # refractive index n + ik of each layer at the laser wavelength
    optics: object                    # optics.Stack or optics.LambertBeer
    pulse: PulseTrain
    T0: float
    front: str
    back: str
    back_conductance: float | None
    end_time: float
    step_options: dict
    history_every: int
    profile_times: list[float]
    progress_every: int
    binding_energies: list[float]     # J/mol; empty: no desorption estimate
    attempt_frequency: float          # 1/s
    probe_lengths: list[float]        # m: decay lengths of the probe-weighted temperatures
    temperatures: int                 # 1 or 2
    results_dir: Path                 # <results_folder>/<name>


def load_case(path) -> Case:
    path = Path(path).resolve()
    try:
        return _load_case(path)
    except CaseError as err:
        raise CaseError(f"{path.name}: {err}") from None


def _load_case(path: Path) -> Case:
    cfg = read_yaml(path)
    top = Keys(cfg, "case", ("name", "label", "temperatures", "laser", "materials", "stack", "boundaries",
                             "numerics", "output"))
    name = top.optional("name", str) or path.stem
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise CaseError(f"case.name: {name!r} may use only letters, digits, '.', '_' and '-'")
    label = top.optional("label", str) or name
    temperatures = top("temperatures", integer, (1, 2))

    laser = Keys(top("laser", dict), "laser", (
        "wavelength_um", "absorption_model", "angle_of_incidence_deg", "polarisation", "micropulse_fwhm_ps",
        "micropulse_period_ns", "micropulses_per_macropulse", "number_of_macropulses", "macropulse_period_ms",
        "macropulse_energy_mJ", "spot_area_cm2", "micropulse_fluence_J_m2", "first_micropulse_time_ps"))
    wavelength_um = laser("wavelength_um")
    model = laser("absorption_model", str, ("transfer_matrix", "lambert_beer"))
    angle = math.radians(laser("angle_of_incidence_deg"))
    polarisation = laser("polarisation", str, ("p", "s"))
    fwhm = laser("micropulse_fwhm_ps") * 1e-12
    period = laser("micropulse_period_ns") * 1e-9
    per_macro = laser("micropulses_per_macropulse", integer)
    n_macro = laser("number_of_macropulses", integer)
    macro_period = laser("macropulse_period_ms") * 1e-3
    energy = laser.optional("macropulse_energy_mJ")
    area = laser.optional("spot_area_cm2")
    fluence = laser.optional("micropulse_fluence_J_m2")
    if fluence is None and (energy is None or area is None):
        raise CaseError("laser: give macropulse_energy_mJ with spot_area_cm2, or micropulse_fluence_J_m2")
    if fluence is not None and (energy is not None or area is not None):
        raise CaseError("laser: give either micropulse_fluence_J_m2 or macropulse_energy_mJ with spot_area_cm2, not both")
    if fluence is None:
        fluence = energy * 1e-3 / per_macro / (area * 1e-4)
    t_first = laser.optional("first_micropulse_time_ps")
    t_first = 3 * fwhm if t_first is None else t_first * 1e-12  # centre of the first micropulse
    laser.done()
    try:
        pulse = PulseTrain(fluence, fwhm, period, per_macro, t_first=t_first,
                           macropulses=n_macro, macropulse_period=macro_period)
    except ValueError as err:
        raise CaseError(f"laser: {err}") from None

    library = Keys(top("materials", dict), "materials")
    stack = top("stack", list)
    if not stack:
        raise CaseError("stack: list at least one layer")
    layers, indices = [], []
    for i, spec in enumerate(stack):
        layer, index = build_layer(spec, library, f"stack[{i}]", wavelength_um,
                                   layers[-1].material if layers else None, indices[-1] if indices else None)
        layers.append(layer)
        indices.append(index)
    if temperatures == 1:  # one temperature per layer: electrons and lattice lumped
        for layer in layers:
            layer.material = M.SingleTemperature(layer.material)

    if model == "transfer_matrix":
        stack_optics = optics.Stack(indices, [l.thickness for l in layers], wavelength_um * 1e-6, angle, polarisation)
    else:
        stack_optics = optics.LambertBeer(indices, [l.thickness for l in layers], wavelength_um * 1e-6)

    bounds = Keys(top("boundaries", dict), "boundaries",
                  ("initial_temperature_K", "front", "back", "back_conductance_W_m2_K"))
    T0 = bounds("initial_temperature_K")
    front = bounds("front", str, ("insulated", "fixed"))
    back = bounds("back", str, ("fixed", "insulated", "conductance"))
    back_conductance = bounds.optional("back_conductance_W_m2_K")
    if (back == "conductance") != (back_conductance is not None):
        raise CaseError("boundaries: back_conductance_W_m2_K is needed for, and only for, back: conductance")
    bounds.done()

    num = Keys(top("numerics", dict), "numerics",
               ("end_time_us", "micropulse_time_step_ps", "time_step_growth", "max_time_step_us"))
    end_time = num("end_time_us") * 1e-6
    step_options = {"dt_pulse": num("micropulse_time_step_ps") * 1e-12,
                    "growth": num("time_step_growth"),
                    "dt_max": num("max_time_step_us") * 1e-6}
    if step_options["growth"] <= 1.0:
        raise CaseError("numerics.time_step_growth must be larger than 1")
    num.done()

    out = Keys(top("output", dict), "output", ("results_folder", "history_every_steps", "progress_every_steps",
                                                "profile_times_us", "probe_decay_lengths_nm", "thermal_desorption"))
    results_dir = Path(out("results_folder", str)) / name
    history_every = out("history_every_steps", integer)
    progress_every = out("progress_every_steps", integer)
    profile_times = []
    for t in out("profile_times_us", list):
        if t == TRAIN_END:
            profile_times.append(pulse.end)
        else:
            try:
                profile_times.append(number(t) * 1e-6)
            except (TypeError, ValueError):
                raise CaseError(f"output.profile_times_us: {t!r} is neither a time nor '{TRAIN_END}'") from None
    probe_lengths = [number(d) * 1e-9 for d in (out.optional("probe_decay_lengths_nm", list) or [])]
    energies, frequency = [], 0.0
    if "thermal_desorption" in out.data:
        desorption = Keys(out("thermal_desorption", dict), "output.thermal_desorption",
                          ("binding_energies_kJ_mol", "attempt_frequency_Hz"))
        energies = [number(e) * 1e3 for e in desorption("binding_energies_kJ_mol", list)]
        frequency = desorption("attempt_frequency_Hz")
        desorption.done()
    out.known.add("thermal_desorption")
    out.done()
    top.done()

    return Case(name=name, label=label, source=path.read_text(), config=cfg, layers=layers, indices=indices,
                optics=stack_optics, pulse=pulse, T0=T0, front=front, back=back,
                back_conductance=back_conductance, end_time=end_time, step_options=step_options,
                history_every=history_every, profile_times=sorted(profile_times), progress_every=progress_every,
                binding_energies=energies, attempt_frequency=frequency, probe_lengths=probe_lengths,
                temperatures=temperatures, results_dir=results_dir)


def build_layer(spec, library: Keys, where: str, wavelength_um: float, upper, upper_index):
    """One stack layer: the library entry it names, with the layer's own keys on top."""
    if not isinstance(spec, dict) or "material" not in spec:
        raise CaseError(f"{where}: expected a mapping with 'material' and 'thickness_nm'")
    material = spec["material"]
    entry = library(material, dict) if material in library.data else None
    if entry is None:
        raise CaseError(f"{where}: material {material!r} is not defined under 'materials' "
                        f"(defined: {', '.join(library.data)})")
    merged = merge(entry, {k: v for k, v in spec.items() if k != "material"})
    model = Keys(merged, f"{where} ({material})")("model", str, tuple(MODELS))
    allowed = ["model", "thickness_nm", "mesh"] + [key for key, _, _ in MODELS[model]]
    if model == "custom":
        allowed += CUSTOM_KEYS
    if model != "thermal_contact":
        allowed.append("refractive_index")
    keys = Keys(merged, f"{where} ({material})", allowed)
    keys.known.add("model")
    thickness = keys("thickness_nm") * 1e-9
    params = {kw: keys(key) * factor for key, kw, factor in MODELS[model]}

    if model == "copper_ofhc":
        if params["rrr"] not in COPPER_RRR:
            raise CaseError(f"{where}.residual_resistivity_ratio: NIST fits exist only for {COPPER_RRR}")
        params["rrr"] = int(params["rrr"])
    if model == "custom":
        mat = custom_material(keys, material, where)
        index = refractive_index(keys("refractive_index", dict), f"{where}.refractive_index", wavelength_um)
    elif model == "thermal_contact":
        if upper is None:
            raise CaseError(f"{where}: a thermal_contact layer needs a layer above it")
        mat = M.contact_layer(upper, thickness=thickness, **params)
        index = upper_index  # optically part of the layer above
    else:
        mat = {"amorphous_ice": M.amorphous_ice, "gold_film": M.gold_film, "copper_ofhc": M.copper,
               "power_law_metal": lambda **p: M.PowerLawMetal(material, **p)}[model](**params)
        index = refractive_index(keys("refractive_index", dict), f"{where}.refractive_index", wavelength_um)
        if model == "amorphous_ice" and mat.fill < 1.0:
            index = optics.maxwell_garnett(index, mat.fill)  # the table is for compact ice

    mesh = Keys(keys("mesh", dict), f"{where}.mesh", ("smallest_cell_nm", "largest_cell_nm", "growth_factor"))
    layer = Layer(mat, thickness, h_min=mesh("smallest_cell_nm") * 1e-9,
                  h_max=mesh("largest_cell_nm") * 1e-9, growth=mesh("growth_factor"))
    mesh.done()
    keys.done()
    return layer, index


def refractive_index(spec: dict, where: str, wavelength_um: float) -> complex:
    """n + ik at the laser wavelength, from a table (wavelength_um, n, k) or as constants."""
    keys = Keys(spec, where, ("table",) if "table" in spec else ("n", "k"))
    if "table" in spec:
        path = Path(keys("table", str))
        keys.done()
        try:
            return optics.nk_at(optics.load_nk(path), wavelength_um)
        except (OSError, ValueError) as err:
            raise CaseError(f"{where}: {err}") from None
    index = complex(keys("n"), keys("k"))
    keys.done()
    return index


def custom_material(keys: Keys, name: str, where: str):
    """A material whose properties are numbers or formulas in T (see formulas.py)."""
    density = keys.optional("density_kg_m3")

    def subsystem(part: str) -> dict:
        spec = Keys(keys(part, dict), f"{where}.{part}",
                    ("heat_capacity_J_m3_K", "heat_capacity_J_kg_K", "thermal_conductivity_W_m_K"))
        per_volume = spec.optional("heat_capacity_J_m3_K", formula_value)
        per_mass = spec.optional("heat_capacity_J_kg_K", formula_value)
        if (per_volume is None) == (per_mass is None):
            raise CaseError(f"{where}.{part}: give heat_capacity_J_m3_K or heat_capacity_J_kg_K (times density)")
        if per_mass is not None and density is None:
            raise CaseError(f"{where}: heat_capacity_J_kg_K needs density_kg_m3")
        try:
            capacity = formulas.parse(per_volume if per_volume is not None else per_mass)
            if per_mass is not None:
                capacity = capacity * density
            parts = {"heat_capacity": formulas.Formula(capacity, f"{part} heat capacity"),
                     "conductivity": formulas.Formula(formulas.parse(spec("thermal_conductivity_W_m_K", formula_value)))}
            parts["heat_capacity"].integral()  # check now that the energy can be formed
        except formulas.FormulaError as err:
            raise CaseError(f"{where}.{part}: {err}") from None
        spec.done()
        return parts

    lattice = subsystem("lattice")
    electron = subsystem("electron") if "electron" in keys.data else None
    keys.known.add("electron")
    linear = keys.optional("coupling_W_m3_K")
    g_room, sigma = keys.optional("coupling_room_temperature_W_m3_K"), keys.optional("coupling_low_temperature_W_m3_K5")
    coupling = None
    if electron is not None:
        if linear is not None and g_room is None and sigma is None:
            coupling = M.linear_coupling(linear)
        elif linear is None and g_room is not None and sigma is not None:
            coupling = M.debye_coupling(g_room, sigma)
        else:
            raise CaseError(f"{where}: give coupling_W_m3_K (constant), or coupling_room_temperature_W_m3_K "
                            "with coupling_low_temperature_W_m3_K5")
    return M.FormulaMaterial(name, lattice, electron, coupling)


def formula_value(value):
    """A number or a formula string, checked later by formulas.parse."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError
    return value


formula_value.__name__ = "number or formula"
