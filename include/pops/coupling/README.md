# `pops/coupling` -- hyperbolic/elliptic coupling headers

The coupling layer wires the finite-volume transport (`numerics/`) to the elliptic
solve (`numerics/elliptic/`): it fills the auxiliary channel from the field, builds the
elliptic right-hand side from the state, and applies coupled sources. The headers are
grouped by **abstraction family** so the API surface is legible at a glance.

## Families and stability

| Family | Path | What it is | Surface |
| --- | --- | --- | --- |
| base | `base/` | Aux fill and elliptic RHS contracts shared by every coupler. | Stable spatial building blocks. |
| source | `source/` | Coupled-source state and its DSL program. | Stable. |
| single | `single/` | Spatial single-block `Coupler`. | Stable; no time driver. |
| system | `system/` | Spatial `SystemAssembler` plus the static AMR field coupler. | Spatial/reference components only; no time driver. |
| amr | `amr/` | Multipatch AMR coupler (`AmrCouplerMp`) and its storage, regrid, and diagnostics. | AMR production path. |

## Layout

```text
pops/coupling/
  base/            aux_fill.hpp  elliptic_rhs.hpp
  source/          coupled_source.hpp  coupled_source_program.hpp
  single/          coupler.hpp
  system/          system_coupler.hpp  amr_system_coupler.hpp
  amr/             amr_coupler_mp.hpp  amr_level_storage.hpp  amr_regrid_coupler.hpp  amr_diagnostics.hpp
```

`system_coupler.hpp` now contains only `SystemAssembler`: the historical static temporal driver
lives exclusively in `tests/cpp/support/reference_system_driver.hpp` as a numerical oracle.
