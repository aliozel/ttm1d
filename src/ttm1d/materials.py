"""Temperature-dependent material laws for the 1D two-temperature model.

Each material provides, per unit volume and in SI units,

    Ue(Te)      electron energy density            J/m^3
    Ul(Tl)      lattice energy density             J/m^3
    ke(Te, Tl)  electron thermal conductivity      W/(m K)
    kl(Tl)      lattice thermal conductivity       W/(m K)
    F(T)        coupling potential: the electron -> lattice power density is
                F(Te) - F(Tl)                      W/m^3

Energies rather than heat capacities are used so that the time discretisation
conserves energy exactly even where C(T) varies as T^3.  Every law accepts either
NumPy arrays (post-processing) or UFL expressions (variational forms).
"""
from __future__ import annotations

import math

import numpy as np

KB = 1.380649e-23  # J/K
NA = 6.02214076e23  # 1/mol
LORENZ = 2.44e-8  # W Ohm / K^2
LN10 = math.log(10.0)


class _NumpyBackend:
    exp, ln, max_value, min_value = np.exp, np.log, np.maximum, np.minimum


def _backend(x):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return _NumpyBackend
    import ufl

    return ufl


class LogPoly:
    """Smooth fit y(T) = 10**p(log10 T) to tabulated data, usable from NumPy and UFL.

    Outside the fitted range the temperature is clamped to the nearest end.
    """

    def __init__(self, T: np.ndarray, y: np.ndarray, degree: int = 12, name: str = ""):
        x = np.log10(T)
        self.lo, self.hi = float(x[0]), float(x[-1])
        cheb = np.polynomial.chebyshev.chebfit(self._scaled(x), np.log10(y), degree)
        self.coef = [float(c) for c in np.polynomial.chebyshev.cheb2poly(cheb)]
        self.Tmin, self.Tmax, self.name = float(T[0]), float(T[-1]), name
        self.max_rel_error = float(np.max(np.abs(self(T) / y - 1)))

    def _scaled(self, x):
        return (2 * x - (self.hi + self.lo)) / (self.hi - self.lo)

    def __call__(self, T):
        b = _backend(T)
        Tc = b.min_value(b.max_value(T, self.Tmin), self.Tmax)
        s = self._scaled(b.ln(Tc) / LN10)
        p = self.coef[-1]
        for c in self.coef[-2::-1]:
            p = p * s + c
        return b.exp(LN10 * p)


# --------------------------------------------------------------------------------------
# Debye-model integrals, tabulated once
# --------------------------------------------------------------------------------------
_X = np.concatenate(([0.0], np.geomspace(1e-8, 700.0, 60001)))


def _cumulative(integrand: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(_X))))


with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    _em1 = np.expm1(_X)
    _J3 = _cumulative(np.where(_X > 0, _X**3 / _em1, 0.0))
    _J4 = _cumulative(np.where(_X > 0, _X**4 / _em1, 0.0))
    _J5 = _cumulative(np.where(_X > 0, _X**5 / (_em1 * -np.expm1(-_X)), 0.0))


def _J(table: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(y, _X, table)


def debye_energy(T: np.ndarray, theta: float, atoms_per_m3: float) -> np.ndarray:
    """Debye lattice energy density (J/m^3), zero at T = 0."""
    return 9 * atoms_per_m3 * KB * T * (T / theta) ** 3 * _J(_J3, theta / T)


def debye_heat_capacity(T: np.ndarray, theta: float, atoms_per_m3: float) -> np.ndarray:
    y = theta / T
    return 9 * atoms_per_m3 * KB * (4 * _J(_J3, y) / y**3 - y / np.expm1(y))


def coupling_potential(T: np.ndarray, g_inf: float, theta: float) -> np.ndarray:
    """Electron-phonon energy exchange for a Debye phonon spectrum.

    The power density from electrons to lattice is F(Te) - F(Tl).  It reduces to
    g_inf (Te - Tl) for T >> theta and to Sigma (Te^5 - Tl^5) for T << theta with
    Sigma = 24.886 * 4 g_inf / theta^4.
    """
    return 4 * g_inf / theta**4 * T**5 * _J(_J4, theta / T)


def coupling_theta(g_inf: float, sigma: float) -> float:
    """Effective Debye temperature that reproduces both the room-temperature coupling
    g_inf (W/m^3/K) and the measured low-temperature constant Sigma (W/m^3/K^5)."""
    return (4 * 24.886266 * g_inf / sigma) ** 0.25


def bloch_gruneisen(T: np.ndarray, theta_r: float, rho_ref: float, T_ref: float = 295.0) -> np.ndarray:
    """Phonon part of the electrical resistivity, scaled to rho_ref at T_ref."""
    shape = lambda t: (t / theta_r) ** 5 * _J(_J5, theta_r / t)
    return rho_ref * shape(T) / shape(np.array(T_ref))


# --------------------------------------------------------------------------------------
# NIST cryogenic fits for OFHC copper, 4-300 K
# https://trc.nist.gov/cryogenics/materials/OFHC%20Copper/OFHC_Copper_rev1.htm
# --------------------------------------------------------------------------------------
_NIST_CU_K = {  # a, b, c, d, e, f, g, h, i
    50: (1.8743, -0.41538, -0.6018, 0.13294, 0.26426, -0.0219, -0.051276, 0.0014871, 0.003723),
    100: (2.2154, -0.47461, -0.88068, 0.13871, 0.29505, -0.02043, -0.04831, 0.001281, 0.003207),
    150: (2.3797, -0.4918, -0.98615, 0.13942, 0.30475, -0.019713, -0.046897, 0.0011969, 0.0029988),
    300: (1.357, 0.3981, 2.669, -0.1346, -0.6683, 0.01342, 0.05773, 0.0002147, 0.0),
    500: (2.8075, -0.54074, -1.2777, 0.15362, 0.36444, -0.02105, -0.051727, 0.0012226, 0.0030964),
}
_NIST_CU_CP = (-1.91844, -0.15973, 8.61013, -18.996, 21.9661, -12.7328, 3.54322, -0.3797)


def nist_copper_conductivity(T, rrr: int):
    """Thermal conductivity of OFHC copper, W/(m K); T is clamped to 4-300 K.  NumPy or UFL."""
    a, b, c, d, e, f, g, h, i = _NIST_CU_K[rrr]
    bk = _backend(T)
    t = bk.min_value(bk.max_value(T, 4.0), 300.0)
    s = t**0.5
    return bk.exp(LN10 * (a + c * s + e * t + g * t * s + i * t**2) / (1 + b * s + d * t + f * t * s + h * t**2))


def nist_copper_cp(T: np.ndarray) -> np.ndarray:
    """Specific heat of OFHC copper, J/(kg K); valid 4-300 K."""
    x = np.log10(T)
    return 10 ** sum(c * x**j for j, c in enumerate(_NIST_CU_CP))


def _integrate(T: np.ndarray, C: np.ndarray, first: float) -> np.ndarray:
    """Cumulative integral of C dT on a fine grid; `first` is the energy at T[0]."""
    return first + np.concatenate(([0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * np.diff(T))))


_T_LATTICE = np.geomspace(1.0, 3000.0, 6001)
_T_ELECTRON = np.geomspace(1.0, 3.0e4, 6001)


# --------------------------------------------------------------------------------------
# Material classes
# --------------------------------------------------------------------------------------
class Metal:
    """Metal with realistic low-temperature laws."""

    has_electrons = True

    def __init__(self, name, gamma, lattice_energy, k_equilibrium, k_lattice, coupling, v_fermi, a_ee):
        self.name = name
        self.gamma = gamma  # electronic heat capacity coefficient, J/(m^3 K^2)
        self._Ul, self._keq, self._F = lattice_energy, k_equilibrium, coupling
        self.k_lattice = k_lattice
        self.v_fermi, self.a_ee = v_fermi, a_ee

    def Ue(self, Te):
        return 0.5 * self.gamma * Te**2

    def Ul(self, Tl):
        return self._Ul(Tl)

    def ke(self, Te, Tl):
        """Equilibrium conductivity k_eq(Tl) scaled with the electron heat capacity
        (factor Te/Tl) and reduced by electron-electron scattering when Te >> Tl."""
        b = _backend(Te)
        keq = self._keq(Tl)
        tau = 3 * keq / (self.v_fermi**2 * self.gamma * Tl)
        return keq * Te / Tl / (1 + self.a_ee * tau * b.max_value(Te**2 - Tl**2, 0.0))

    def kl(self, Tl):
        return self.k_lattice + 0.0 * Tl

    def F(self, T):
        return self._F(T)


class PowerLawMetal:
    """The laws used in the NTMpy scripts and SI Table S1: C_e = ce*T, C_l = cl*T^3
    (per kg), k_e = k_l = kcoef*T, constant coupling G.  For code-to-code checks."""

    has_electrons = True

    def __init__(self, name, rho, ce, cl, kcoef, G):
        self.name, self.rho, self.ce, self.cl, self.kcoef, self.G = name, rho, ce, cl, kcoef, G

    def Ue(self, Te):
        return 0.5 * self.rho * self.ce * Te**2

    def Ul(self, Tl):
        return 0.25 * self.rho * self.cl * Tl**4

    def ke(self, Te, Tl):
        return self.kcoef * Te

    def kl(self, Tl):
        return self.kcoef * Tl

    def F(self, T):
        return self.G * T


class Dielectric:
    """Insulating layer: lattice only.  Laser energy absorbed here heats the lattice directly."""

    has_electrons = False

    def __init__(self, name, lattice_energy, k_lattice, fill=1.0):
        self.name, self._Ul, self._kl, self.fill = name, lattice_energy, k_lattice, fill

    def Ul(self, Tl):
        return self._Ul(Tl)

    def kl(self, Tl):
        return self._kl(Tl)


# --------------------------------------------------------------------------------------
# Specific materials
# --------------------------------------------------------------------------------------
def copper(rrr: int = 100, g_inf: float = 1.0e17, sigma: float = 2.0e9, k_lattice: float = 5.0,
           density: float = 8960.0, molar_mass: float = 63.546e-3, gamma: float = 0.695e-3 / 63.546e-3 * 8960.0,
           theta_d: float = 343.0, v_fermi: float = 1.57e6, a_ee: float = 1.28e7) -> Metal:
    """OFHC copper substrate (SI units throughout).

    Heat capacity and conductivity: NIST cryogenic fits (4-300 K); `rrr` picks the
    conductivity fit and must be 50, 100, 150, 300 or 500.  Above 300 K the heat capacity
    follows a Debye model with `theta_d`.
    gamma = 0.695 mJ/(mol K^2) (Martin 1966/Kittel) = 98.0 J/(m^3 K^2).
    Coupling: g_inf at room temperature (Hohlfeld et al. 2000) and Sigma at low
    temperature (Giazotto et al. 2006) joined with a Debye spectrum.
    """
    T = _T_LATTICE
    atoms = density / molar_mass * NA
    c_total = density * nist_copper_cp(np.clip(T, 4.0, 300.0))
    c_lat = np.where(
        T < 4.0,
        (c_total - gamma * 4.0) * (T / 4.0) ** 3,
        np.where(T <= 300.0, c_total - gamma * T,
                 (c_total - gamma * 300.0) * debye_heat_capacity(T, theta_d, atoms)
                 / debye_heat_capacity(np.array(300.0), theta_d, atoms)),
    )
    energy = LogPoly(T, _integrate(T, c_lat, c_lat[0] * T[0] / 4), name="Cu lattice energy")
    k_eq = lambda Tl: nist_copper_conductivity(Tl, rrr) - k_lattice
    coupling = LogPoly(_T_ELECTRON, coupling_potential(_T_ELECTRON, g_inf, coupling_theta(g_inf, sigma)), name="Cu coupling")
    return Metal(f"Cu (OFHC, RRR={rrr})", gamma, energy, k_eq, k_lattice, coupling,
                 v_fermi=v_fermi, a_ee=a_ee)


def gold_film(rrr: float = 3.0, theta_d: float = 165.0, g_inf: float = 2.2e16, sigma: float = 2.4e9,
              k_lattice: float = 3.0, density: float = 19320.0, molar_mass: float = 196.967e-3,
              gamma: float = 67.6, rho_295: float = 2.2e-8, theta_r: float = 175.0,
              v_fermi: float = 1.40e6, a_ee: float = 1.2e7) -> Metal:
    """Polycrystalline gold film (SI units throughout).

    gamma = 0.689 mJ/(mol K^2) = 67.6 J/(m^3 K^2).  Lattice: Debye model with `theta_d`.
    Conductivity: Wiedemann-Franz with Bloch-Grueneisen phonon resistivity (rho_295 at 295 K,
    Debye temperature theta_r) plus a residual resistivity set by `rrr` = rho(295 K)/rho(4 K).
    Coupling: 2.2e16 W/m^3/K at room temperature (Hohlfeld et al. 2000, as quoted by
    Lin et al. 2008) and Sigma = 2.4e9 W/m^3/K^5 (Giazotto et al. 2006).
    """
    T = _T_LATTICE
    atoms = density / molar_mass * NA
    energy = LogPoly(T, debye_energy(T, theta_d, atoms), name="Au lattice energy")
    rho_ph = bloch_gruneisen(T, theta_r, rho_295)
    k_eq = LogPoly(T, LORENZ * T / (rho_ph + rho_295 / (rrr - 1.0)), name="Au k_e")
    coupling = LogPoly(_T_ELECTRON, coupling_potential(_T_ELECTRON, g_inf, coupling_theta(g_inf, sigma)), name="Au coupling")
    return Metal(f"Au film (RRR={rrr:g})", gamma, energy, k_eq, k_lattice, coupling,
                 v_fermi=v_fermi, a_ee=a_ee)


def ice_cp(T: np.ndarray) -> np.ndarray:
    """Specific heat of water ice, J/(kg K).

    Smooth log-log fit through the IAPWS R10-06(2009) ice Ih values at 5-150 K, a T^3
    continuation below 5 K and Murphy & Koop (Q. J. R. Meteorol. Soc. 131, 1539, 2005, Eq. 4)
    above 150 K.  Calorimetry on amorphous ice above ~100 K finds it indistinguishable from
    crystalline ice; no data exist for amorphous ice below 20 K.
    """
    return np.exp(np.polyval(_ICE_CP_FIT, np.log(np.clip(T, 1.0, 273.0))))


def _fit_ice_cp() -> np.ndarray:
    murphy_koop = lambda t: (-2.0572 + 0.14644 * t + 0.06163 * t * np.exp(-((t / 125.1) ** 2))) / 18.015e-3
    T_low = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    T_tab = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0])
    cp_tab = np.array([1.38, 14.8, 53.8, 111.0, 231.0, 437.0, 668.0, 874.0, 1226.0])
    T_high = np.array([175.0, 200.0, 225.0, 250.0, 273.0])
    T = np.concatenate((T_low, T_tab, T_high))
    cp = np.concatenate((1.38 * (T_low / 5.0) ** 3, cp_tab, murphy_koop(T_high)))
    return np.polyfit(np.log(T), np.log(cp), 8)


_ICE_CP_FIT = _fit_ice_cp()


def amorphous_ice(density: float = 716.0, k_scale: float = 1.0, k_floor: float = 0.03,
                  compact_density: float = 940.0, velocity: float = 2500.0,
                  mean_free_path: float = 0.5e-9) -> Dielectric:
    """Vapour-deposited amorphous solid water (SI units throughout).

    density: film density (0.716 g/cm^3 is the value the paper uses for its thickness).
    Conductivity: k = k_scale * max(k_floor, v l rho c(T) / 3) with v = 2500 m/s and
    l = 0.5 nm (Klinger 1980), i.e. about 0.03 W/m/K at 10 K rising to 0.26 at 100 K.
    No measurement exists for this material below 70 K; published values for amorphous ice
    span 1e-5 to about 1 W/m/K, so treat k_scale as a sensitivity parameter.
    Optics: Mastrapa et al. 2009 (amorphous, 15 K) with Maxwell Garnett pores for the
    density deficit relative to compact amorphous ice (`compact_density`).
    """
    T = _T_LATTICE
    c_vol = density * ice_cp(T)
    energy = LogPoly(T, _integrate(T, c_vol, c_vol[0] * T[0] / 4), name="ice lattice energy")
    kinetic = LogPoly(T, velocity * mean_free_path / 3 * c_vol, name="ice kinetic conductivity")

    def conductivity(Tl):
        return k_scale * _backend(Tl).max_value(kinetic(Tl), k_floor)

    return Dielectric("amorphous H2O ice", energy, conductivity,
                      fill=min(1.0, density / compact_density))


def contact_layer(upper, a: float, g_max: float, thickness: float = 0.4e-9) -> Dielectric:
    """Thin layer standing in for a thermal boundary (Kapitza) conductance
    G(T) = min(a T^3, g_max) in W/(m^2 K) below the layer `upper`, whose heat capacity it takes."""

    def conductivity(Tl):
        return thickness * _backend(Tl).min_value(a * Tl**3, g_max)

    return Dielectric("thermal contact", upper.Ul, conductivity, fill=getattr(upper, "fill", 1.0))


def table_s1_gold() -> PowerLawMetal:
    return PowerLawMetal("Au (Table S1)", 19320.0, 0.003264, 4.758433e-05, 0.0457047, 6.089e16)


def table_s1_copper() -> PowerLawMetal:
    return PowerLawMetal("Cu (Table S1)", 8960.0, 0.007949, 1.2244e-05, 0.0734766, 3.95e16)


class FormulaMaterial:
    """A material whose properties are given in the case file (numbers or formulas in T).

    electron and lattice: dicts with `heat_capacity` and `conductivity`, each a formulas.Formula
    (heat capacity per unit volume).  Without `electron` the material is an insulator.
    coupling: callable T -> F(T); the electron -> lattice power density is F(Te) - F(Tl).
    """

    def __init__(self, name, lattice, electron=None, coupling=None, fill=1.0):
        self.name, self.fill = name, fill
        self.has_electrons = electron is not None
        self._Ul, self._kl = lattice["heat_capacity"].integral(), lattice["conductivity"]
        if self.has_electrons:
            self._Ue, self._ke = electron["heat_capacity"].integral(), electron["conductivity"]
            self._F = coupling

    def Ue(self, Te):
        return self._Ue(Te)

    def Ul(self, Tl):
        return self._Ul(Tl)

    def ke(self, Te, Tl):
        return self._ke(Te)

    def kl(self, Tl):
        return self._kl(Tl)

    def F(self, T):
        return self._F(T)


class SingleTemperature:
    """Electrons and lattice of `material` lumped into one temperature (1-temperature model):
    the heat capacities and the conductivities add, and absorbed light heats it directly."""

    has_electrons = False

    def __init__(self, material):
        self.base, self.name, self.fill = material, material.name, getattr(material, "fill", 1.0)

    def Ul(self, T):
        b = self.base
        return b.Ue(T) + b.Ul(T) if b.has_electrons else b.Ul(T)

    def kl(self, T):
        b = self.base
        return b.ke(T, T) + b.kl(T) if b.has_electrons else b.kl(T)


def linear_coupling(g: float):
    """Constant coupling G: F(T) = G T, so the power density is G (Te - Tl)."""
    return lambda T: g * T


def debye_coupling(g_inf: float, sigma: float):
    """Coupling that is g_inf at high temperature and Sigma (Te^5 - Tl^5) at low temperature."""
    return LogPoly(_T_ELECTRON, coupling_potential(_T_ELECTRON, g_inf, coupling_theta(g_inf, sigma)), name="coupling")
