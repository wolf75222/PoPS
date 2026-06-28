# AMR (adaptive refinement)

`pops.AmrSystem` is the refined counterpart of `pops.System`: one or more blocks (species)
carried on a block-structured AMR hierarchy (with rectangular boxes, AMReX /
FLASH / SAMRAI style). The mesh is refined where the solution requires it, and only there. This
page summarizes how to drive the AMR from Python; for design details see
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 8), [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
(sections 13-15) and the design notes
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) /
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md).

The public path is the same `pops.Case` -> `pops.compile` -> `pops.bind` -> `sim.run` flow as a
uniform run: you refine by changing the layout from `pops.mesh.layouts.Uniform(mesh)` to
`pops.mesh.layouts.AMR(mesh, max_levels=2, ratio=2)` and authoring a refinement criterion with
`case.amr.refine(pops.mesh.amr.Refine.on("density").above(...))`. `pops.bind` then builds an
`AmrSystem` from the layout. The same low-level `AmrSystem` runtime methods that back `pops.bind`
(`add_block` / `add_equation` / `set_poisson` / `set_refinement` / `step_cfl`) stay for the
native/AMR runtime and the tests. The A->Z tutorial compares the uniform and AMR paths on the same
physics (cf.
[tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py), function
`uniform_vs_amr`).

```{toctree}
:maxdepth: 1

shared-hierarchy
tagging-regrid
prolongation-restriction
reflux
multi-block-amr
current-limits
```
