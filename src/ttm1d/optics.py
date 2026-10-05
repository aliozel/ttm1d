"""Thin-film optics for the layered stack: where the laser energy is absorbed.

Coherent transfer-matrix method for a stack  vacuum | layer 1 | ... | layer N | vacuum.
A layer that absorbs practically all the light entering it ends the optical stack: it is
treated as semi-infinite, and the layers behind it receive no light.  Light transmitted
through the whole stack leaves it.  Complex refractive indices use the n + ik convention.  The absorbed fraction between two depths is obtained from the
difference of the normal Poynting flux, so the profile handed to the heat solver
integrates exactly to the total absorptance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_nk(path: str | Path) -> np.ndarray:
    """Read a three-column table: wavelength (um), n, k."""
    data = np.loadtxt(path)
    return data[np.argsort(data[:, 0])]


def nk_at(table: np.ndarray, wavelength_um: float) -> complex:
    """Linearly interpolate a tabulated refractive index at one wavelength."""
    lam = table[:, 0]
    if not lam[0] <= wavelength_um <= lam[-1]:
        raise ValueError(f"{wavelength_um} um is outside the table range {lam[0]}-{lam[-1]} um")
    return complex(np.interp(wavelength_um, lam, table[:, 1]), np.interp(wavelength_um, lam, table[:, 2]))


def maxwell_garnett(n_host: complex, fill: float) -> complex:
    """Effective index of a host with vacuum pores; `fill` is the solid volume fraction."""
    eps_h = n_host**2
    f_pore = 1.0 - fill
    eps = eps_h * (1 + 2 * eps_h + 2 * f_pore * (1 - eps_h)) / (1 + 2 * eps_h - f_pore * (1 - eps_h))
    return complex(np.sqrt(eps))


class Stack:
    """Transfer-matrix solution for one wavelength, incidence angle and polarisation.

    Parameters
    ----------
    n_layers : complex refractive index of each layer, front (vacuum side) first.
    d_layers : thickness of each layer in metres.
    wavelength : vacuum wavelength in metres.
    theta : angle of incidence from the surface normal, radians.
    pol : "s" or "p".
    exit_index : refractive index of the medium behind the stack.
    """

    OPAQUE = 60.0  # a layer whose intensity attenuation exceeds exp(-OPAQUE) ends the stack

    def __init__(self, n_layers, d_layers, wavelength: float, theta: float, pol: str, exit_index=1.0 + 0j):
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's' or 'p'")
        self.pol = pol
        n_layers = np.asarray(n_layers, dtype=complex)
        d_layers = np.asarray(d_layers, dtype=float)
        self.layer_faces = np.concatenate(([0.0], np.cumsum(d_layers)))
        k0, kx = 2 * np.pi / wavelength, np.sin(theta)  # kx in units of k0; conserved across layers

        def cos_kz(n):
            cos = np.sqrt(1 - (kx / n) ** 2)
            cos = np.where((n * cos).imag < 0, -cos, cos)  # decays (or propagates) towards +z
            return cos, k0 * n * cos

        opaque = [2 * cos_kz(n)[1].imag * d > self.OPAQUE for n, d in zip(n_layers, d_layers)]
        last = opaque.index(True) if any(opaque) else None
        if last is None:  # light can cross the stack: vacuum (or exit_index) behind it
            self.n = np.concatenate(([1.0 + 0j], n_layers, [complex(exit_index)]))
        else:             # layer `last` is optically semi-infinite
            self.n = np.concatenate(([1.0 + 0j], n_layers[: last + 1]))
        self.cos, self.kz = cos_kz(self.n)
        self.d = d_layers[: len(self.n) - 2]                 # thickness of each finite medium
        self.edges = self.layer_faces[: len(self.n) - 1]     # front face of each medium after vacuum
        self._solve()

    def _interface(self, i: int, j: int) -> tuple[complex, complex]:
        ni, nj, ci, cj = self.n[i], self.n[j], self.cos[i], self.cos[j]
        if self.pol == "s":
            return (ni * ci - nj * cj) / (ni * ci + nj * cj), 2 * ni * ci / (ni * ci + nj * cj)
        return (nj * ci - ni * cj) / (nj * ci + ni * cj), 2 * ni * ci / (nj * ci + ni * cj)

    def _solve(self) -> None:
        m = len(self.n)  # vacuum + layers
        # Forward/backward amplitudes (v, w) at the front face of each medium, built backwards
        # from the last medium, which carries only a forward wave.
        vw = np.zeros((m, 2), dtype=complex)
        vw[-1] = (1.0, 0.0)
        for i in range(m - 2, -1, -1):
            r, t = self._interface(i, i + 1)
            phase = self.kz[i] * self.d[i - 1] if i > 0 else 0.0
            prop = np.array([[np.exp(-1j * phase), 0], [0, np.exp(1j * phase)]])
            vw[i] = prop @ (np.array([[1, r], [r, 1]]) / t) @ vw[i + 1]
        self.r = vw[0, 1] / vw[0, 0]
        self.vw = vw / vw[0, 0]
        self.R = abs(self.r) ** 2

    def poynting(self, z: np.ndarray) -> np.ndarray:
        """Fraction of the incident power crossing depth z (z = 0 is the front surface)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        layer = np.clip(np.searchsorted(self.edges, z, side="right"), 1, len(self.n) - 1)
        dz = z - self.edges[layer - 1]
        v, w = self.vw[layer, 0], self.vw[layer, 1]
        ef = v * np.exp(1j * self.kz[layer] * dz)
        # The last medium has no backward wave; skip its growing exponential.
        last = layer == len(self.n) - 1
        eb = np.where(last, 0.0, w * np.exp(-1j * self.kz[layer] * np.where(last, 0.0, dz)))
        n, c = self.n[layer], self.cos[layer]
        if self.pol == "s":
            return (n * c * np.conj(ef + eb) * (ef - eb)).real / self.cos[0].real
        return (n * np.conj(c) * (ef + eb) * np.conj(ef - eb)).real / self.cos[0].real

    def absorbed_between(self, z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
        """Fraction of the incident power absorbed between depths z0 and z1."""
        return self.poynting(z0) - self.poynting(z1)

    def layer_absorptance(self) -> np.ndarray:
        """Absorbed fraction in each layer."""
        flux = self.poynting(self.layer_faces)
        return flux[:-1] - flux[1:]

    @property
    def T(self) -> float:
        """Fraction of the incident power transmitted through the whole stack."""
        return float(self.poynting(self.layer_faces[-1:])[0])


class LambertBeer:
    """NTMpy's "LB" source: the full incident fluence enters the stack with no reflection
    and decays in each layer with depth lambda / (4 pi k); angle and polarisation are
    ignored.  Kept for code-to-code checks against NTMpy."""

    def __init__(self, n_layers, d_layers, wavelength: float):
        self.depth = wavelength / (4 * np.pi * np.imag(np.asarray(n_layers, dtype=complex)))
        d = np.asarray(d_layers, dtype=float)
        self.edges = np.concatenate(([0.0], np.cumsum(d[:-1])))
        self.before = np.concatenate(([0.0], np.cumsum(d[:-1] / self.depth[:-1])))
        self.R = 0.0

    def poynting(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        layer = np.clip(np.searchsorted(self.edges, z, side="right") - 1, 0, len(self.depth) - 1)
        return np.exp(-(self.before[layer] + (z - self.edges[layer]) / self.depth[layer]))
