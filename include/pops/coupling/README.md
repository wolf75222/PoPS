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
| system | `system/` | Exact-ranked AMR layout coordination. | Thin spatial facade. |
| amr | `amr/` | Ranked multipatch runtime construction, regrid, and diagnostics. | AMR production path. |

## Layout

```text
pops/coupling/
  base/            aux_fill.hpp  elliptic_rhs.hpp
  source/          coupled_source.hpp  coupled_source_program.hpp
  system/          amr_system_coupler.hpp
  amr/             amr_coupler_mp.hpp  amr_regrid_coupler.hpp  amr_diagnostics.hpp
```

Single-level and AMR operator composition is owned by `System<Dim>` and `AmrSystem<Dim>`. This
directory contains no alternative field-storage, hierarchy, or time-stepping authority.
