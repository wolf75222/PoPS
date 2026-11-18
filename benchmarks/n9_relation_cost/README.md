# L1-d: the relation solves of `cg()` against the native witness (N9, ROM-90)

The relation route of L1-d (advection-diffusion, `imex_euler` with `solvers={imex: cg()}`) solves
its implicit diffusion with the reference of `cg()`: a region of the Program that applies the
relation's operator, the StencilFlux of `two_point()`, at each iterate. Before N9's lot « La
référence sous les solves de relation », `cg()` bridged that solve to b6's native CG. The witness
is the hand-written b6 Program of the Trace test: the same right-hand side, `v - dt nu Lap(v)`
applied by a field-level `set_apply` and solved by the native CG. Both programs are the builders
of `tests/python/integration/runtime/test_relation_solve_reference_n9_native.py`, whose Trace test
checks that they take the same iterations and reach the same states.

The driver's ruling of 13/11 bounds the reference route at 10 times the native witness, at 64² and
at 256² (`n9/relation_solves/decisions_pilote.md`, 1(c)). `relation_cost.py` measures that ratio.

## Run

From the repository root, with a native build of `pops` on `PYTHONPATH` and the native test
environment of `tests/python` (`POPS_INCLUDE`, `POPS_KOKKOS_ROOT`, `POPS_CACHE_DIR`):

```sh
python3 benchmarks/n9_relation_cost/relation_cost.py 64 --out relation_cost_64.json
python3 benchmarks/n9_relation_cost/relation_cost.py 256 --steps 5 --out relation_cost_256.json
```

Options: `--rtol` (1e-10) and `--atol` (1e-14), the request of both solves; `--steps` per block
(10); `--blocks` of A B B A (3); `--warmup` steps per route (1).

## Output

One JSON document (`pops.n9.relation_cost.v1`): the cells, the request, the A B B A order, the
compile time of each route, `timing` (median, minimum, maximum and samples of the time per step,
per route), the iterations of the last step of each route, `ratio` (the reference's median over
the native's), `bound` (10) and `within_bound`, the threads, host, Slurm job and source revision.
A sample is the time of one `pops.run` block over its accepted steps, so it includes the
explicit advection and the Python run loop of each route.
