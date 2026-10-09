# Nonconstant composite EvolvedOriginalFieldStage witness

This separate fixture prepares one genuine installed-native node:

```text
tests/python/integration/runtime/test_public_evolved_stage_amr_spatial.py::test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux[8]
```

The homogeneous builder and its admission AST remain unchanged. The new receipt
is `pops.evolved-stage-amr-spatial-native-fixture@1`; it is not covered by the
homogeneous independent reader/allowlist. Source checks are not native acceptance.
ROOT alone compiles/binds/runs the installed package, serial and MPI, after SDK
reception. There is no native result in this commit.

## Physical problem, method, realization and acceptance

Two distinct cell-average State carriers evolve together with three field
unknowns `(T0,T1,z)`. The unchanged accumulation laws are
`H0=T0+T0²+0.1*T1²`, `H1=T1+T1²+0.2*T0*T1`; the auxiliary constraint remains
`z=0.25*T0+0.5*T1`. The unchanged signed diffusion matrix is
`[[0.012,0.002],[-0.001,0.014]]`. Each original stage solves
`F=H(T)-tau*(div(D grad T)+f)-Qn=0`, with `tau` captured from the real Program
interval. The constant opposite loads are exactly those of the homogeneous
fixture. Initial temperature profiles, used only to prescribe physical initial Q,
are `T0=.15+.02*cos(2*pi*x)` and `T1=.25+.015*sin(2*pi*x)`.

Public `Analytic(..., cell_integrals=...)` supplies exact Cartesian primitives of
H for ConservativeCellAverage initialization, including cross terms. It does
not initialize Q as H at a midpoint, remap Qn, manufacture an inverse PoPS load,
or replace a native solve by a host loop. An independent 32-node quadrature of H
checks the authored primitives at both level spacings before any native run.

The fixture retains `DT=.01`, all seven Newton controls and FD step imported from
the existing fixture, `Arithmetic@1`, and explicit `FullResidualBasisLU@1` with
256 MiB dense resource budget. The realization is replicated dense reference LU,
not a scalable AMR qualification: per Newton it needs two full residual
applications per active scalar DOF, quadratic storage and cubic factorization
work. Its actual quotient size is recorded from the native masks; no DOF cap or
budget/tolerance increase is introduced. Original-F, projection, constraints,
conservation and reference guards stay `3e-8`. The two fixed discriminants
`flux_activity>1e-6` and `Jensen_gap>1e-10` enforce that this witness really
exercises spatial diffusion and nonlinear restriction; they do not widen an
error tolerance.

## Independent spatial reference and its limits

The physical mesh is the unchanged Cartesian unit square, ratio two, periodic in
both axes, with a genuine partial central refinement selected by the prescribed
x-only marker. The oracle admits the **observed** union of patches only if each
fine/covered x column spans full y, each fine mask agrees with repeated coarse
coverage, and saved valid masks agree with the exact native patch boxes. A foreign
shape, changed metric, y hole, missing interface or fully covered coarse level
refuses; there is no homogeneous fallback. These are limitations of this
independent strip oracle, not limitations of PoPS geometry/providers.

The reference computes the periodic 1D composite FV operator and extrudes it in y.
It first averages fine T into covered coarse T. At each missing fine-cell center,
quadratic point interpolation uses its coarse parent at offset +/-1/4 and the
three Lagrange weights. Every face uses the original signed cross-component D.
At coarse/fine interfaces, the coarse flux density is replaced by the mean of the
two transverse fine faces, identical for this y extrusion. Divergence uses the
respective native-level Cartesian spacings. The full F is independently rebuilt
from saved Qn, Q, T, forcing and masks, on the composite-active quotient. No PoPS
operator, emitter expression, solver or generated helper is called by the oracle.

The checker requires covered coarse T to equal mean fine T, and covered coarse Q
to equal mean fine H(T), then records a strictly nonzero
`mean(H(Tfine))-H(mean(Tfine))`. This distinguishes nonlinear accumulation from
restricting temperature and applying H afterward. The independent interface
fluxes telescope over the exact composite measures, testing closed diffusion
exchange, each Q carrier budget and conservation of the total Q with opposite
loads. Source algebra checks independently demonstrate that omitting reflux
breaks this balance.

## Archives and provenance

Actual native T/Q/forcing histories, POPSHID1 sample bytes, latest raw slot1 and
previous raw slot0, valid/active masks, base shape and exact patch boxes are saved
in phase/level NPZs. The Cartesian edges/volumes and kappa=1 are **derived from the
declared Cartesian/no-EB geometry plus native shape**, and are explicitly named
as such; the runtime currently exposes no public native volume getter. They are
not presented as independent native geometry measurements. The receipt records
the domain, ratio, metric authority and box provenance. Independent flux/divergence
arrays are separately named and archived for inspection; they are recomputable
from the original native arrays and are not authoritative solver output.

The fixture binds the actual resolved Case, compiles once collectively, runs two
steps, saves accepted/continuous checkpoints, restarts a fresh bind and replays.
Read-only snapshot arrays, forcing preservation, native original-F diagnostic,
exact history/replay images, authenticated run/restart provenance and all other
checkpoint payload bytes remain guarded. Checkpoints have disjoint names, are
hashed immediately after each native call and checked unchanged after observation
writes. The same compiled Program dumps its real IR/CPP/command/hash, all actual
DSO/sidecar hashes, and source snapshots are archived before scientific guards.
All raw captures are archived with an explicitly unqualified observation index before the science checks, so a failed guard preserves the measured arrays. A fresh directory preserves historical evidence. JUnit records artifact/dimension/
rank/size and the separate spatial receipt path for later offline owner sealing.

This mechanism witness is not the complete scientific M06/M13 campaign, Marshak,
Stefan, arbitrary AMR geometry or spatial convergence. N16 is deliberately not
promoted before the N8 installed-native evidence and independent reader exist.

## Source-only checks and ROOT command

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/python/unit/time/test_evolved_amr_spatial_reference.py
```

ROOT should use its authenticated installed interpreter, remove PYTHONPATH, use a
fresh cache/evidence directory, and run the single native node above with JUnit.
MPI uses the same collective compile, bind, captures, checkpoint and guarded rank0
writers. No native execution, JIT, build or environment mutation was done by the
author of this source tranche.

Recorded local result: 14 source/math tests PASS in 47.42 s, Ruff PASS, all three new Python files parse, and the single native node collects in 1.07 s. JUnit is `/tmp/sol61-evolved-amr-spatial-final-source.xml`. The native node was not run. No existing production or homogeneous fixture file changed.
