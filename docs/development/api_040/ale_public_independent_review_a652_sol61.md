# Independent public ALE review: exact source a652f3cf

Reviewer: GPT-6.1 Sol, independent of the interval connector author. Target:
`a652f3cff1d3c2e75e602398a6018d7ee9286085`, chain
`2a741eff -> 6e09987 -> 19e4adb -> 8ac2501 -> dc29a02 -> a652f3c`.
Exclusive tree: `work/PoPS-sol61-ale-public-review`, branch
`codex/api040-sol61-ale-public-review`. MAIN, installed package, SDK, and
shared native/JIT environment were not modified.

## Concrete outstanding defect

`LevelGeometry.__post_init__` accepts either new known 1D metadata URI on a
rank-two image when explicit finite nodes are supplied:

```python
replace(rank_two_geometry,
        coordinate_system="pops://coordinates/moving-cartesian-1d@1",
        node_coordinates=np.zeros((3, 3, 2)))
replace(rank_two_geometry,
        cell_measure="pops://cell-measures/endpoint-length@1",
        node_coordinates=np.zeros((3, 3, 2)))
```

Both calls succeed with positive arbitrary cell volumes. The shape check uses
the actual rank and the endpoint-difference/monotonicity check only runs for
rank one. These objects can then enter the public output/archive pipeline with
false versioned dimensional metadata. The native installed endpoint provider
refuses rank two, so this finding does not claim a native PDE failure.

The two independent expected-refusal probes fail with `DID NOT RAISE
ValueError` on this exact SHA. A targeted dimensional check for these known
URIs closes the seam; it must not constrain future multidimensional coordinate
providers or arbitrary versioned coordinate descriptors.

## Positive source and host evidence

`tests/review/test_sol61_public_ale_independent.py` authors fresh public
Case/Model/Program objects, then performs actual validate/resolve and resolved
ProgramModelGraph emission. It does not reuse the author's moving-case builder.

Twenty-four parameter cases compile and execute the complete emitted physical
model, emitted relative override, emitted coordinate law, and original source
CSE expressions with only standard math/type scaffolding. They cover scalar,
three components and the `(c,a,b)` permutation, signed physical velocities
`+.3` and `-.3`, mesh speed `.7`, and source measure weights `(1,0)`, `(0,1)`,
`(1/2,1/2)`, `(2,-1)`. Source offsets are keyed by physical component names,
so the permutation check is sensitive to a wrong component route. The actual
emitted measure expression and actual source result expression are checked.

For `UL=1, UR=4`, the independent Rusanov expression applied to emitted
physical quantities gives `Frel=-1.6` at `a=.3,w=.7` and `Frel=-4` at
`a=-.3,w=.7`. The latter spectral radius is `|a-w|=1`; taking `abs(a)` before
subtracting mesh speed would fail this probe. The runtime numerical-flux
header's Rusanov expression agrees with the host oracle. The host probe does
not execute that native header, Kokkos, System, a complete PDE step, or MPI.

Three public noncentral face-weight policies are rejected by real resolve
with the installed centered-Rusanov provider's specific error. Their generic
MovingFieldProjection descriptors remain representable. A shifted initial
coordinate map and an unrealized HLL provider also refuse at public prepare.
The current connector's one-state/Uniform1D/first-order provider limitations
are explicit missing-realization obligations, not permanent production limits.

Output probes preserve physical nodes and exact endpoint differences across
ownership copies, detachment, archive encode/decode, NPZ piece projection and
ParaView point-coordinate projection. Mutating the caller's original array
does not mutate the owned frame. Missing, malformed, nonfinite, repeated and
reversed nodes refuse; changing a volume by one ULP also refuses. No rounded
measure tolerance or substituted reference-grid coordinates are used.

The entire default/forced ledger writer implementation is source-byte equal
to `2a741eff`; changes 19e4adb/dc29a02 are reader changes. Inspection confirms
the new zero-integral reader rejects nonzero consumption before consumption
list allocation and retains the wire-extent count bounds. Native legacy-byte
and positive metadata roundtrips remain a central reception obligation;
source-byte comparison does not replace them.

## Lifecycle and collective inspection

Reynolds authoring requires an issued geometry binding, exact matching state,
block, StateSpace, clock and evaluation point, and the full root `U.next`
interval. The connector defers the selected physical/source rates and rejects
another consumer of those deferred evaluations. It does not emit an ordinary
static RHS and reinterpret that residual as moving flux. It emits the original
physical constitutive body, `F_phys-u*w`, source evaluation and one terminal
coupled commit. The native producer evaluates the beginning/end law and uses
the point's actual `dt`; it does not relabel a requested duration as accepted.

Sampling allocation converges before boundary exchange. Native interval work
uses a local producer within the existing collective preparation/publication
protocol. Moving output first converges its complete request list in Python;
the native binding validates geometry/state/receipt/support locally, converges
errors and exact topology/ownership contracts, then broadcasts real owner
endpoints/volumes. Replicated visualization chooses one owner and does not sum
the same geometry on every rank. These are source findings, not a claim of
MPI convergence from host execution.

The author branch starts at 2a741eff and does not include current MAIN's later
ComputedDt/Scalar authoring implementation. Exact duration and terminal gate
compatibility on the merged MAIN must therefore also be received centrally.
Nonperiodic boundary profiles, invertible/noninvertible evolving meshes,
native rollback/install rollback and genuine restart/PDE receipts were not
executed here. Previous native receptions cannot qualify this new connector.

Separate inspection of central fixture-only `79f9246f` confirms it changes only
the second root transaction begin to `begin_nested_step_transaction`.
System's root begin explicitly requires no open root; nested begin requires
an uncommitted root and preserves its parent snapshot. The moving checkpoint
guard refuses depth two even with provisional capture. The fixture retains
its rejection and rollback-depth assertions. This correction respects the
production transaction protocol and does not relax the checkpoint guard.

## Exact commands/results

Source package is asserted to be under this checkout, with native dimension
selection and inherited PYTHONPATH removed:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c \
  'import sys; from pathlib import Path; root=Path.cwd(); sys.path[:0]=[str(root/"python"),str(root)]; import pops,pytest; assert Path(pops.__file__).is_relative_to(root/"python"); raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_public_ale_independent.py"]))'
```

Before adding the two dimensional-metadata counterexamples: **36 PASS in
61.03 s**. After adding them and the writer-source comparator, focused replay
`-k 'versioned_one or legacy_ledger'`: **1 PASS, 2 FAIL, 36 deselected in
0.39 s**. Aggregate distinct properties: **37 positive, 2 genuine failures**.
No skip/xfail hides the failures. No production or C++ header edit was made.
