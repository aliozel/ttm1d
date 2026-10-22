# co3_ice_p_0.0059_k0.3

The reference case at 3 µm, `cases/ice_gold_copper/co3_ice_p_0.0059_k0.3.yaml`: 87.7 nm of amorphous
CO:H₂O ice on 100 nm of gold on 3.175 mm of copper, all at 9 K, irradiated by one 36 mJ macropulse
on 0.0059 cm² (6000 micropulses of 4 ps, 1 ns apart), p-polarised at 45°. The ice surface is insulated.
Ice lattice conductivity k = max(k₀, ρc vℓ/3), with ℓ = 5 nm and k₀ = 0.3 W m⁻¹ K⁻¹.

The tables are taken from the output of

```bash
ttm1d run cases/ice_gold_copper/co3_ice_p_0.0059_k0.3.yaml
```

which writes `history.npz`, `profiles.npz`, `summary.json` and the input files to this folder.

| File | Content |
|---|---|
| `layers.csv` | thickness, refractive index n + ik at 3 µm and absorbed fraction of each layer |
| `summary.csv` | reflectance, fluence, absorbed energy, peak temperatures, energy balance |
| `history_macropulse_per_micropulse.csv` | highest and lowest value within each 1 ns micropulse period, 0–6 µs |
| `history_first_5_micropulses.csv` | every time step of the first 5 ns |
| `history_after_macropulse.csv` | every time step from 6 to 100 µs |
| `profiles_lattice_temperature.csv` | lattice temperature against depth at 0.001–100 µs |
| `profiles_electron_temperature.csv` | electron temperature against depth; empty in the ice, which has no electrons |

Columns of the time series, in K:

- `Tl_surface`: lattice temperature of the ice surface (vacuum side)
- `Tl_ice_mean`: mean lattice temperature of the ice film
- `Tl_gold_front`, `Te_gold_front`: lattice and electron temperature of the gold at the ice|gold interface
- `Tl_gold_back`: lattice temperature at the gold|copper interface
- `Tl_stack_mean`: mean lattice temperature of the whole stack

Times are in µs (ns in `history_first_5_micropulses.csv`). Depth is in nm from the ice surface; the
layer faces are at 0, 87.7, 187.7 and 3 175 187.7 nm. The profile labelled `5.999us` is at the end of
the last micropulse, 5.999024 µs. In `summary.csv`, `surface_temperature_at_end` is at 100 µs.
