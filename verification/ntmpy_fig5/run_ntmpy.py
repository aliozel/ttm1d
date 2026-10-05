"""Alber et al. (2021) Fig. 5 with NTMpy itself: 10 nm Pt on Si, or free-standing Pt.

Run with the Python that has NTMpy (it needs the bspline package):
    python run_ntmpy.py substrate [NTMPY_FOLDER]
    python run_ntmpy.py free [NTMPY_FOLDER]
    python run_ntmpy.py substrate_fine [NTMPY_FOLDER]
Writes ntmpy_<case>.npz with x, t, Te and Tl.

"substrate" is the paper's set-up, addSubstrate("Si"): Si layers of 20 nm, 100 nm and 100 um.
NTMpy puts the same number of collocation points (12) in every layer, so in the 100 um layer
they are micrometres apart while the light decays over 82 nm (lambda / 4 pi k); the source
there is smeared over micrometres and far more energy enters the Si than is absorbed.
"substrate_fine" is the same Si cut into thinner layers (20, 100, 200, 500, 1500, 5000 nm and
the rest) so that the absorption and the first picoseconds of conduction are resolved.
"""
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/Users/ao57/Desktop/2TM")
from NTMpy import NTMpy as ntm

which = sys.argv[1]
s = ntm.source()
s.spaceprofile = "TMM"
s.timeprofile = "Gaussian"
s.FWHM = 0.1e-12        # NTMpy's "FWHM" (a half width: the pulse is 0.2 ps wide)
s.fluence = 6e-3 / 1e-4  # 6 mJ/cm^2 in J/m^2
s.t0 = 1e-12
s.lambda_vac = 400
s.theta_in = np.pi / 4
s.polarization = "p"

rho = 21450.0  # cancels: NTMpy multiplies the per-kg heat capacities below by it
sim = ntm.simulation(2, s)
sim.addLayer(10e-9, 1.7176 + 2.844j, [72, 72], [lambda Te: 740 / rho * Te, 2.78e6 / rho], rho, [2.5e17])
if which == "substrate":
    sim.addSubstrate("Si")
elif which == "substrate_fine":  # the Si of addSubstrate("Si"), in thinner layers
    k_lat = lambda T: np.piecewise(T, [T <= 120.7, T > 120.7],
                                   [lambda T: 100*(0.09*T**3*(0.016*np.exp(-0.05*T) + np.exp(-0.14*T))),
                                    lambda T: 100*(13e3*T**(-1.6))])
    rho_si = 2.32e3
    layers = [20e-9, 100e-9, 200e-9, 500e-9, 1500e-9, 5000e-9]
    for d in layers + [100.12e-6 - sum(layers)]:
        sim.addLayer(d, 5.5674 + 0.38612j, [130, k_lat], [lambda Te: 150/rho_si*Te, 1.6e6/rho_si], rho_si, [18e17])
sim.final_time = 7e-12

start = time.time()
x, t, T = sim.run()
Te, Tl = T[0], T[1]
print(f"NTMpy: {len(t)} steps of {t[1]-t[0]:.3e} s, {len(x)} output points, {time.time()-start:.0f} s")


def stored(k):
    """Heat in the solution at time index k, J/m^2: integral of U(Te) + U(Tl) - U(300 K) over depth."""
    pt = x <= 10e-9 * (1 + 1e-9)
    Ue = np.where(pt, 370.0, 75.0) * (Te[k]**2 - 300.0**2)   # C_e = 740 T (Pt), 150 T (Si)
    Ul = np.where(pt, 2.78e6, 1.6e6) * (Tl[k] - 300.0)
    dU = Ue + Ul
    return np.trapezoid(dU[pt], x[pt]), np.trapezoid(dU[~pt], x[~pt])  # each material on its own


in_pt, in_si = stored(-1)
print(f"heat stored at {t[-1]*1e12:.1f} ps: {in_pt + in_si:.2f} J/m^2 (Pt {in_pt:.2f}, Si {in_si:.2f})")
keep = slice(None, None, max(1, len(t) // 700))  # about every 0.01 ps
np.savez_compressed(f"ntmpy_{which}.npz", x=x, t=t[keep], Te=Te[keep], Tl=Tl[keep])
print(f"wrote ntmpy_{which}.npz")
