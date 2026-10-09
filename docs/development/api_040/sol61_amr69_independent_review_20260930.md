# Independent AMR physical-parent prolongation reception

Date: 30 September 2026. Exact candidate:
`69d4ce3e98b6b17d6b81b2690413ed28837be487`, received in the exclusive
`work/PoPS-sol61-amr69-review`, branch `codex/api040-sol61-amr69-review`.
The earlier native diffusion diagnosis is not reused as qualification of this SHA.
This review changes only its independent probes and report.

## Result and evidence

No new production defect was demonstrated in this bounded reception. **5,770
independent actual-header host assertions pass**, and the same assertions pass
under AddressSanitizer and UndefinedBehaviorSanitizer. The probe includes the real
candidate `transfer_provider.hpp` and dependencies directly; it executes actual
`prepare`, physical preparation, interpolation and restriction kernels on detached
host `FieldView` storage. No transfer algorithm or production numerical kernel is
substituted. The host field allocator and numerical oracles are independently written;
they do not reuse the author's gtest helpers or former saved-state results.

The mathematical coordinate oracle uses parent-cell averages of a separable affine
field and the exact fine-child center under the declared index mapping. Checks cover:

* 1D ratio 3, 2D ratio (2,5), 3D ratio (1,3,2), with shifted negative coarse/fine origins
  and parent indices on both sides of the mapping origin;
* two transferred components with nonzero source/destination offsets, and untouched
  destination components retaining sentinels;
* exact affine fine averages throughout the domain, including physical corners, and
  fine-grid gradients adjacent to both physical faces in each axis;
* restriction back to every original parent average, for affine data and independent
  nonaffine limited data;
* periodic values with a full wrapped halo, compared against an independent scalar
  monotonized-central calculation, and identical output from two adjacent destination
  patches versus one patch;
* missing periodic and interior stencils refused at preparation without writing the
  destination; another axis's physical flags do not authorize periodic clipping;
* parent indices outside an explicitly physical domain refused, refined singleton
  physical domains refused, an unrefined singleton axis accepted, and explicit constant
  injection retained for singleton data while its physical-linear entry point refuses.

These are meaningful checks of the actual detached transfer provider, not native
bootstrap/regrid trajectories, inter-rank ghost exchanges or device execution.

## Runtime route and stencil audit

`tests/review/sol61_amr69_transfer_review.py` authenticates both changed production
files against the exact git object before compiling. Its bounded source audit checks
the definitions and all calls of `prepare_regridded_state_transfer` and
`transfer_regridded_state`, with balanced argument extraction rather than only counting
text occurrences. Bootstrap uses `p_->cfg.periodicity`, regridded live state and history
use `cfg.periodicity`, and the auxiliary route retains `ConstantInjection` while passing
the actual configuration's periodicity. Constant injection has radius zero and remains
on ordinary preparation.

For linear transfer, the physical domain comes from `parent_layout.domain()` and
physical flags are exactly the inverse of the configured periodic axes. Mapping origins
come from the actual parent and child domains. The required-source proof and prepared
kernel both consume the same `PhysicalParentBoundary`; only a declared physical face
clips the required box. Interior and periodic neighbors remain mandatory. Singleton
refined physical axes refuse because no adjacent interior slope exists. The kernel's
child offsets remain antisymmetric about the parent center, explaining conservation
even when the one-sided boundary slope differs from the centered limiter.

The dense carrier retains its per-cell populated proof, gathers all ranks before checking
required sources, and selects valid parent patches before grown patch ghosts. No change
to this collective gather, retained-child overwrite, restriction or publication protocol
was introduced by the candidate. The new periodicity argument is propagated through
all changed callers. Actual invocation of these call chains is reserved for central C++
and native reception; this source audit does not authenticate runtime ghost contents.

A potential zero/stale periodic bootstrap halo must not be inferred solely from initial
candidate storage being zeroed. The real bootstrap-next-level path calls `regrid_parent`,
which executes tagging; `execute_tagging` invokes each prepared parent block-level
`prepare(point, block_state)` before evaluating tags. Its exact native ghost provider,
history-slot provenance and distributed gather still need integrated tests. The host
periodic oracle exercises correctly populated halos and deliberately refuses missing
ones; it cannot qualify their runtime preparation.

## Limits and follow-up checks

The new physical slope is one-sided from two interior parent averages. Parent-average
conservation and affine reproduction are received here; positivity, a maximum principle
or higher-order accuracy at nonlinear boundaries are not implied. This is the declared
linear policy, not a universally valid reconstruction for every physical model. The
runtime's prepared physical recovery and rollback remain relevant acceptance authorities.

The runtime transfer preflight still requires one ghost in every axis for linear transfer,
including an unrefined axis. That pre-existing guard is not relaxed by this change; the
detached provider itself requires stencils only where a ratio refines. No unsupported
zero-ghost runtime configuration is marked received.

Central reception should execute exact-SDK C++ transfer tests and public AMR diffusion,
then add periodic/interpatch parent halo preparation, history remap with nonuniform
ratios, retained-child regrid and auxiliary injection. MPI empty ranks, source-stencil
failures, physical recovery/rollback, EB, GPU and native Dim3 are unrun here. Native
diffusive flux, boundary balances, checkpoint/restart and scientific PDE convergence
are outside this source/host reception.

## Reproduction and integrity

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_amr69_transfer_review.py
rtk proxy /usr/bin/clang++ -std=c++20 -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include tests/review/sol61_amr69_transfer_review.cpp -o outputs/sol61-amr69-independent/probe-sanitized
rtk proxy ./outputs/sol61-amr69-independent/probe-sanitized
```

Compiler `/usr/bin/clang++`, no native/Kokkos library linking, runtime initialization,
package installation, shared environment edit or heavy JIT. Ruff and diff-check pass.
The raw receipt is `outputs/sol61-amr69-independent/receipt.json` and records the exact
compiler invocation and per-file hashes. Include archive SHA256:
`86734f4a65b2905c8a2e4f548eeac86aee1be22e18d780c3520ea6b6a5d44a4d`.
Changed transfer header SHA256:
`ea553c55d70c93f237757d94c1bb73435e8fd63aeb3d0a4376dec1d3de0cb745`.
Changed runtime source SHA256:
`20660abe7e46a547728265f9f55df7ac12966fd774a4051d4f2e3806a109ebc3`.
Independent probe source SHA256:
`df1890b3cbfa84b03a32ebe6a1395d9a3dad7e806ae51cd1670c80d5e1afd89c`.

The positive result is bounded to this exact source and actual-header host execution.
It supplies no replacement for the root's forthcoming native acceptance.
