"""One-dimensional two-temperature model of a laser-heated layered stack, in DOLFINx.

    dUe/dt = d/dx( ke dTe/dx ) - [F(Te) - F(Tl)] + S(x, t)        (metals)
    dUl/dt = d/dx( kl dTl/dx ) + [F(Te) - F(Tl)]  (+ S in insulators)

x = 0 is the vacuum-facing surface.  Linear elements, backward Euler in time, and
vertex ("lumped") quadrature for the storage and coupling terms.  The storage term is
written with energy densities, so the scheme conserves energy for any C(T).
Insulating layers carry no electron equation: laser energy absorbed there goes
straight to the lattice and the electron unknown is pinned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc


@dataclass
class Layer:
    material: object
    thickness: float  # m
    h_min: float = 0.5e-9  # cell size at the layer faces
    h_max: float = 5.0e-9  # largest cell size inside the layer
    growth: float = 1.2


@dataclass
class PulseTrain:
    """Gaussian micropulses, grouped in macropulses.

    fluence: incident energy per unit surface area in one micropulse, J/m^2.
    fwhm: true full width at half maximum of the intensity, s.
    period: spacing of the micropulses, s.
    n_pulses: micropulses in one macropulse.
    t_first: centre of the first micropulse, s.
    macropulses, macropulse_period: number of macropulses and the spacing of their starts, s.
    """

    fluence: float
    fwhm: float
    period: float
    n_pulses: int
    t_first: float
    macropulses: int = 1
    macropulse_period: float = 0.0

    def __post_init__(self):
        self.sigma = self.fwhm / (2 * math.sqrt(2 * math.log(2)))
        if self.macropulses > 1 and self.macropulse_period <= self.n_pulses * self.period:
            raise ValueError("the macropulse period must be longer than one macropulse")
        offsets = np.arange(self.macropulses) * self.macropulse_period
        self.centres = (self.t_first + offsets[:, None] + np.arange(self.n_pulses) * self.period).ravel()
        self.end = self.centres[-1] + 3 * self.fwhm  # the last micropulse is over

    def centre(self, p: int) -> float:
        return float(self.centres[p])

    def mean_power(self, t0: float, t1: float) -> float:
        """Incident power per unit area averaged over [t0, t1] (exact), W/m^2."""
        reach = 8 * self.sigma
        lo = np.searchsorted(self.centres, t0 - reach, side="left")
        hi = np.searchsorted(self.centres, t1 + reach, side="right")
        s = self.sigma * math.sqrt(2)
        frac = sum(0.5 * (math.erf((t1 - c) / s) - math.erf((t0 - c) / s)) for c in self.centres[lo:hi])
        return self.fluence * frac / (t1 - t0)

    def step_times(self, t_end: float, dt_pulse: float | None = None, half_window: float | None = None,
                   growth: float = 1.4, dt_max: float = math.inf) -> np.ndarray:
        """Step end times: uniform steps dt_pulse across each micropulse, then steps
        growing geometrically until the next pulse (or t_end)."""
        dt_pulse = dt_pulse or self.fwhm / 8
        half_window = half_window or 1.5 * self.fwhm
        times, t = [], 0.0

        def relax_to(target: float) -> None:
            nonlocal t
            gap = target - t
            if gap <= 1e-9 * dt_pulse:
                return
            steps, dt = [], dt_pulse
            while sum(steps) < gap:
                steps.append(dt)
                dt = min(dt * growth, dt_max)
            scale = gap / sum(steps)
            for dt in steps:
                t += dt * scale
                times.append(t)
            times[-1] = t = target

        for c in self.centres:
            start = c - half_window
            if start >= t_end:
                break
            relax_to(max(start, 0.0))
            stop = min(c + half_window, t_end)
            n = max(1, round((stop - t) / dt_pulse))
            times.extend(np.linspace(t, stop, n + 1)[1:])
            t = stop
        relax_to(t_end)
        return np.array(times)


def _graded(length: float, h_min: float, h_max: float, growth: float, both_ends: bool) -> np.ndarray:
    """Cell sizes that start at h_min, grow geometrically up to h_max and fill `length`."""
    span = length / 2 if both_ends else length
    sizes, h = [], h_min
    while sum(sizes) < span:
        sizes.append(h)
        h = min(h * growth, h_max)
    sizes = np.array(sizes) * span / sum(sizes)
    return np.concatenate((sizes, sizes[::-1])) if both_ends else sizes


class TwoTemperature1D:
    """Two-temperature solver for a stack of `Layer`s.

    flux: callable z -> fraction of the incident power crossing depth z (see optics.py);
          the energy absorbed in each cell is the drop in flux across it.
    front: "insulated" (zero heat flux) or "fixed" (held at T0).
    back: as front, or "conductance": heat leaves the lattice to a sink at T0 at the rate
          back_conductance * (Tl - T0), in W/m^2.
    """

    def __init__(self, layers: list[Layer], flux, pulse: PulseTrain, T0: float,
                 front: str = "insulated", back: str = "fixed", back_conductance: float | None = None,
                 snes_atol: float = 1e-6):
        if front not in ("insulated", "fixed") or back not in ("insulated", "fixed", "conductance"):
            raise ValueError(f"unknown boundary condition: front={front!r}, back={back!r}")
        if (back == "conductance") != (back_conductance is not None):
            raise ValueError("back_conductance is needed for, and only for, back='conductance'")
        self.layers, self.pulse, self.T0 = layers, pulse, T0
        self.edges = np.concatenate(([0.0], np.cumsum([l.thickness for l in layers])))

        # --- mesh: graded towards every interface, coarse deep in the last layer
        sizes = [_graded(l.thickness, l.h_min, l.h_max, l.growth, both_ends=i < len(layers) - 1)
                 for i, l in enumerate(layers)]
        nodes = np.concatenate(([0.0], np.cumsum(np.concatenate(sizes))))
        for e in self.edges:  # put interface nodes exactly on the interfaces
            nodes[np.argmin(np.abs(nodes - e))] = e
        n_cells = len(nodes) - 1
        self.msh = msh = mesh.create_interval(MPI.COMM_SELF, n_cells, [0.0, 1.0])
        msh.geometry.x[:, 0] = nodes[np.rint(msh.geometry.x[:, 0] * n_cells).astype(int)]

        cell_x = msh.geometry.x[msh.geometry.dofmap, 0]
        self.cell_lo, self.cell_hi = cell_x.min(axis=1), cell_x.max(axis=1)
        mid = 0.5 * (self.cell_lo + self.cell_hi)
        self.cell_layer = np.clip(np.searchsorted(self.edges, mid) - 1, 0, len(layers) - 1).astype(np.int32)
        tags = mesh.meshtags(msh, 1, np.arange(n_cells, dtype=np.int32), self.cell_layer)

        # --- spaces and unknowns
        P1 = basix.ufl.element("Lagrange", "interval", 1)
        self.W = W = fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
        self.u, self.u_n = fem.Function(W), fem.Function(W)
        Te, Tl = ufl.split(self.u)
        Te_n, Tl_n = ufl.split(self.u_n)
        ve, vl = ufl.TestFunctions(W)

        # absorbed fraction of the incident power per unit depth, constant in each cell
        Q = fem.functionspace(msh, ("DG", 0))
        self.absorb = fem.Function(Q)
        self.cell_absorbed = np.asarray(flux(self.cell_lo)) - np.asarray(flux(self.cell_hi))  # the rest leaves
        self.absorb.x.array[Q.dofmap.list[:, 0]] = self.cell_absorbed / (self.cell_hi - self.cell_lo)
        self.absorptance = float(self.cell_absorbed.sum())
        self.layer_absorptance = np.array([self.cell_absorbed[self.cell_layer == i].sum() for i in range(len(layers))])

        self.dt = fem.Constant(msh, PETSc.ScalarType(1.0))
        self.power = fem.Constant(msh, PETSc.ScalarType(0.0))  # incident W/m^2, averaged over the step

        # --- residual
        dx = ufl.Measure("dx", domain=msh, subdomain_data=tags)
        lumped = {"quadrature_rule": "vertex", "quadrature_degree": 1}
        gauss = {"quadrature_degree": 2}
        grad = lambda f: f.dx(0)
        R = 0
        for i, layer in enumerate(layers):
            m = layer.material
            source = self.power * self.absorb
            R += (m.Ul(Tl) - m.Ul(Tl_n)) / self.dt * vl * dx(i, metadata=lumped)
            R += m.kl(Tl) * grad(Tl) * grad(vl) * dx(i, metadata=gauss)
            if m.has_electrons:
                R += (m.Ue(Te) - m.Ue(Te_n)) / self.dt * ve * dx(i, metadata=lumped)
                R += m.ke(Te, Tl) * grad(Te) * grad(ve) * dx(i, metadata=gauss)
                R += (m.F(Te) - m.F(Tl)) * (ve - vl) * dx(i, metadata=lumped)
                R -= source * ve * dx(i)
            else:
                R -= source * vl * dx(i)

        # --- boundary conditions
        msh.topology.create_connectivity(0, 1)
        L = self.edges[-1]
        # electrons exist on nodes that touch a layer with electrons; elsewhere their unknown is pinned
        metal_cell = np.array([layers[i].material.has_electrons for i in self.cell_layer])
        vertex_cells = msh.topology.connectivity(0, 1)
        n_vertices = msh.topology.index_map(0).size_local
        electron_vertex = np.array([metal_cell[vertex_cells.links(v)].any() for v in range(n_vertices)])
        if back == "conductance":
            sink = mesh.locate_entities_boundary(msh, 0, lambda x: np.isclose(x[0], L, rtol=0, atol=1e-12))
            faces = mesh.meshtags(msh, 0, np.sort(sink).astype(np.int32), np.ones(len(sink), dtype=np.int32))
            ds = ufl.Measure("ds", domain=msh, subdomain_data=faces)
            self.back_conductance = fem.Constant(msh, PETSc.ScalarType(back_conductance))
            R += self.back_conductance * (Tl - fem.Constant(msh, PETSc.ScalarType(T0))) * vl * ds(1)
        value = PETSc.ScalarType(T0)

        def pin(sub: int, marker):
            vertices = mesh.locate_entities(msh, 0, marker)
            dofs = fem.locate_dofs_topological(W.sub(sub), 0, vertices)
            return fem.dirichletbc(value, dofs, W.sub(sub))

        bcs = []
        if back == "fixed":
            bcs += [pin(s, lambda x: np.isclose(x[0], L, rtol=0, atol=1e-12)) for s in (0, 1)]
        if front == "fixed":
            bcs += [pin(s, lambda x: np.isclose(x[0], 0.0, rtol=0, atol=1e-15)) for s in (0, 1)]
        if not electron_vertex.all():  # no electron equation inside insulators
            vertices = np.flatnonzero(~electron_vertex).astype(np.int32)
            bcs.append(fem.dirichletbc(value, fem.locate_dofs_topological(W.sub(0), 0, vertices), W.sub(0)))

        self.problem = NonlinearProblem(
            R, self.u, bcs=bcs, petsc_options_prefix="ttm1d_",
            petsc_options={"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-9,
                           "snes_atol": snes_atol, "snes_stol": 0.0, "snes_max_it": 25,
                           "ksp_type": "preonly", "pc_type": "lu"},
        )
        self.snes_atol = snes_atol
        self.floor_factor = 1e-12  # relative stopping level for the Newton residual
        self._norms: list[float] = []
        self._floor_age = 0
        self.problem.solver.setMonitor(lambda snes, it, norm: self._norms.append(norm))

        # --- nodal access, sorted by depth
        def sorted_dofs(sub: int):
            V, dof_map = W.sub(sub).collapse()
            x = V.tabulate_dof_coordinates()[:, 0]
            order = np.argsort(x)
            return np.asarray(dof_map)[order], x[order]

        self._ie, self.x = sorted_dofs(0)
        self._il, _ = sorted_dofs(1)
        geometry_node = mesh.entities_to_geometry(msh, 0, np.arange(n_vertices, dtype=np.int32)).ravel()
        vertex_x = msh.geometry.x[geometry_node, 0]  # P1 on an interval: one node per vertex
        electrons_at = dict(zip(np.round(vertex_x / L, 13), electron_vertex))
        self.node_has_electrons = np.array([electrons_at[v] for v in np.round(self.x / L, 13)])
        self.u.x.array[:] = T0
        self.u_n.x.array[:] = T0
        self.t = 0.0
        self.energy_in = 0.0  # J/m^2 absorbed so far
        self._e0 = self.stored_energy()

    # ------------------------------------------------------------------
    def temperatures(self) -> tuple[np.ndarray, np.ndarray]:
        """Electron and lattice temperature at the nodes `self.x`.  In insulating
        layers the electron temperature is reported equal to the lattice one."""
        Tl = self.u.x.array[self._il].copy()
        Te = np.where(self.node_has_electrons, self.u.x.array[self._ie], Tl)
        return Te, Tl

    def stored_energy(self) -> float:
        """Energy per unit area in the stack (J/m^2), with the solver's quadrature."""
        Te, Tl = self.temperatures()
        total = 0.0
        for i, layer in enumerate(self.layers):
            sel = np.flatnonzero((self.x[:-1] >= self.edges[i] - 1e-13) & (self.x[1:] <= self.edges[i + 1] + 1e-13))
            h = self.x[sel + 1] - self.x[sel]
            m = layer.material
            dens = m.Ul(Tl[sel]) + m.Ul(Tl[sel + 1])
            if m.has_electrons:
                dens = dens + m.Ue(Te[sel]) + m.Ue(Te[sel + 1])
            total += float(np.sum(0.5 * h * dens))
        return total

    def energy_gain(self) -> float:
        return self.stored_energy() - self._e0

    def layer_slice(self, i: int) -> np.ndarray:
        return (self.x >= self.edges[i] - 1e-13) & (self.x <= self.edges[i + 1] + 1e-13)

    # ------------------------------------------------------------------
    def _roundoff_scales(self) -> tuple[float, float]:
        """Sizes of the storage (J/m^2) and conduction (W/m^2) terms whose cancellation
        sets the round-off floor of the residual."""
        Te, Tl = self.temperatures()
        storage, conduction = 0.0, 0.0
        for i, layer in enumerate(self.layers):
            sel = np.flatnonzero((self.x[:-1] >= self.edges[i] - 1e-13) & (self.x[1:] <= self.edges[i + 1] + 1e-13))
            h = self.x[sel + 1] - self.x[sel]
            m = layer.material
            tl = 0.5 * (Tl[sel] + Tl[sel + 1])
            energy, flux = h * m.Ul(tl), m.kl(tl) * tl / h
            if m.has_electrons:
                te = 0.5 * (Te[sel] + Te[sel + 1])
                energy, flux = energy + h * m.Ue(te), flux + m.ke(te, tl) * te / h
            storage += float(np.sum(energy**2))
            conduction += float(np.sum(flux**2))
        return math.sqrt(storage), math.sqrt(conduction)

    def _attempt(self, t1: float) -> bool:
        dt = t1 - self.t
        self.dt.value = dt
        self.power.value = self.pulse.mean_power(self.t, t1)
        if self._floor_age % 20 == 0:
            self._scales = self._roundoff_scales()
        self._floor_age += 1
        # Newton cannot push the residual below round-off (amplified by the conditioning of the
        # graded mesh); stop at 1e-12 of the terms being balanced instead of stalling there.
        atol = max(self.snes_atol, self.floor_factor * (self._scales[0] / dt + self._scales[1]))
        self.problem.solver.setTolerances(atol=atol)
        self._norms.clear()
        self.problem.solve()
        reason = self.problem.solver.getConvergedReason()
        # A line search that stalls (reason -6) after the residual has collapsed is converged.
        stalled_at_floor = reason == -6 and self._norms[-1] <= max(1e-7 * self._norms[0], 10 * atol)
        ok = (reason > 0 or stalled_at_floor) and bool(np.all(np.isfinite(self.u.x.array))) \
            and float(self.u.x.array.min()) > 0.0
        if ok:
            self.energy_in += float(self.power.value) * dt * self.absorptance
            self.u_n.x.array[:] = self.u.x.array
            self.t = t1
        else:
            self.u.x.array[:] = self.u_n.x.array
        return ok

    def advance(self, t1: float, depth: int = 0) -> int:
        """Advance to time t1, halving the step when Newton fails.  Returns the number of steps."""
        if self._attempt(t1):
            return 1
        if depth >= 12:
            raise RuntimeError(f"time step failed at t = {self.t:.6e} s even after {depth} halvings")
        mid = 0.5 * (self.t + t1)
        return self.advance(mid, depth + 1) + self.advance(t1, depth + 1)
