# Explicit storage of global field observations and composite Q evolution

This source change closes a concrete public route failure: an independent field
observation has no physical State/block owner, whereas AMR history registration
requires an explicit storage block. Before this change, the genuine original
Stage/Rate problem resolves but AMR emission refuses its T/z histories. Borrowing
a Q State identity, adding a fake zero-valued physical dependency, or changing
the observation to a block-owned field would alter that provenance.

## Contract and implementation

`Program.store_history(name, observation, depth=1, owner_block=block)` selects an
explicit storage scope for a consumed global `field_component`. The original
observation retains `block=None`, its original problem/unknown identity and
width one. This first realization stores cell-sampled scalar observations by
copying valid cells; it does not add a physical State or input to the solve.
Other representations need their own realized, authenticated storage port.

The block must be the exact issued Case BlockHandle with an issued TimeState in
the same Program/clock. A storage-only TimeState is allowed: it does not become
an input to the residual. Authoring authenticates the original consumed solve
point/region, source identity, Case registry, layout witness and physical frame;
cloned handles and stale source aliases refuse. The immutable metadata is
`pops.program.global-field-history-storage@1`, with conditional IR16. Existing
calls without `owner_block` retain their historical serialization and C++.
AMR realization @1 uses the primary logical clock; a child-clock observation
refuses at emission, because the current native boundary-point API authenticates
the primary window. This is a missing realization, not a mathematical restriction.

The AMR generator registers the exact descriptor as the ring's state identity,
then calls the new inline `store_global_field_history` port. It checks the
registered ring's actual runtime owner (a different valid block is insufficient),
descriptor and the emitted TimeState's qualified State witness against the prepared native
State route. That allocation witness is independent of mutable history tables, so changing a valid
owner plus resealing its tables cannot substitute for the originally emitted authority. It does not
promote the State to a physical residual input or set the observation width. The port also checks
clock, current tick/level/substep/stage/window, depth, full layout and
distribution, local rank, component width, and ring/prototype ghost representation.
Source ghost widths enter exact consensus; only valid cells are copied, so source
and retained ghost extents need not be identical. Source reductions and their
launch/allocation failures are fenced locally, then reported on the prepared
execution lane before any subsequent exact-contract consensus or publication.
The existing transactional history store stages/deep-copies and handles rollback.
No native ABI or wire ordinal changes, but the real SDK header signature changes;
all native dimensions must rebuild before interpreting native evidence.

Production files: history.py, contract.py, serialization.py,
global_history_storage.py, program_emit_control.py, program_emit_ops.py and
amr_program_context_history_checkpoint_public.inc. No numerical equation,
spatial discretization, Newton control, MPI ownership rule or scientific guard
was changed.

## Four installed-native fixtures ready for ROOT

The original corpus `context/CORPUS_ORIGINAL.md` rows M06/M13 requires stored
energy distinct from observed temperature, followed by a closed Stefan front;
for M13 it requires explicit networks/constraints/transport/Poisson/nonlocal
radiation. The registry's M06 data is piecewise H with c=2, L=3, Tm=1. This
fixture does not relabel its smooth polynomial H as that scientific problem:
it receives the generic accumulation/observation/constraint mechanism only.
The layout/compile/restart helpers follow `test_public_amr_original_field.py`
and the installed Uniform Stage fixture, with no native runtime replacement.

`test_public_evolved_stage_amr_checkpoint_and_composite_Q[1-8]`, `[1-16]`,
`[2-8]`, `[2-16]` use the real Case/resolve/compile/bind/run route. The scalar
physical accumulation is H(T)=T+T². The coupled case has two distinct Q States,
three solved unknowns T0/T1/z, Q0=T0+T0²+0.1T1²,
Q1=T1+T1²+0.2T0T1 and z=0.25T0+0.5T1. The original additive rates are
f=(Qtarget−Qinitial)/0.01 plus signed cross diffusion. The paired loads are
opposite, so total composite Q0+Q1 is conserved. Physical fields are homogeneous;
the mesh-only marker makes a connected central refinement strip. Saved actual
coverage must contain both active and covered coarse cells, with active volume
exactly one. Thus a fully refined tower refuses this fixture.

FullResidualBasisLU@1 is explicitly selected with 256 MiB dense-resource budget.
The seven controls remain tolerance=1e−10, max_iterations=20,
linear_tolerance=1e−8, linear_max_iterations=240, restart=60,
armijo=1e−4, min_step=1/1024; FD step=1e−6 and science acceptance=3e−8.
Its replicated dense cost is O(Nactive²) memory and O(Nactive³) factorization,
plus two complete F evaluations per active DOF/Newton. This is an explicit
reference realization; the fixture makes no scalability claim.

Actual saved Q, T/z, forcing, active masks, native original-F diagnostic,
history ring contents/durations, carrier manifest and lifecycle are archived
before offline checks. Readonly forcing preservation is exact. Accepted,
continuous and replay checkpoints are sealed immediately after each native
call, authenticated against the actual creator, and remain disjoint from all
observation paths. Restart/replay compare full payloads with authenticated
continuation identities, not unjustified equality of run digests. The same
compiled Program emits the archived CPP/IR; actual compile commands, SDK DSO,
component DSOs and sidecars are hashed. JUnit records artifact/dimension/rank/
size and the receipt path. Receipt schema is
`pops.evolved-stage-amr-native-fixture@1`; it has no positive native receipt yet.

The homogeneous numeric oracle checks Q conservation, Q=H(T), the auxiliary
constraint, and Qnew−Qold−dt*f using saved fields/true active volumes. It is
explicitly named homogeneous_Q_equation_linf. The actual native composite
original-F relative diagnostic must separately meet 1e−10. No nonconstant AMR
flux accuracy or distinction H(restrict T) versus restrict H(T) is qualified
by this homogeneous witness. M06/Marshak/Stefan and M13/H05 complete scientific
campaigns remain outside this mechanism receipt.

## Author source reception

One true generated two-carrier AMR translation unit passes clang++ syntax Dim2
against this checkout and actual Kokkos/MPI headers, with no stub runtime:

    /usr/bin/clang++ -std=c++20 -fsyntax-only -DPOPS_NATIVE_DIM=2 \
      -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -DPOPS_HAS_KOKKOS \
      -DKOKKOS_DEPENDENCE -DPOPS_HAS_MPI -DPOPS_HAS_PARALLEL_HDF5 \
      -I/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-codegen-repair/include \
      -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include \
      -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
      /tmp/sol61-global-history-amr.cpp

The generated image is 117401 bytes with IR digest
`55b2a0de9fbf110940d113b970e64940ecfc988d8afcc778c5c1ed91537bd63a`.
Its generated CPP SHA256 is
`9d49b29de40b0179257afe10d246f0481b94a9d7182c826fc1de5653ad10bca0`.
Three fresh historical Stage profiles have identical full IR/CPP/module hashes/
manifests/requests before and after (image SHA256
`4f5ef2ecd4d118b44b0a5f52bac93c3b9e932b77a1f839df30dcdbb47b455504`).
Six fresh legacy/captured/candidate/Jacobi profiles likewise match exactly
(`dc254843124ffc15378c267cde4cadcc1a50b2b86c157d52d2b42d19889687ec`).
The final source suite receives 29 passes (four actual AMR emissions, seven
storage-authority cases and eighteen existing keep_history policies), with four
native cases deselected. JUnit: `/tmp/sol61-global-history-final-source.xml`.
Ruff and diff whitespace checks pass. Eighteen existing keep_history policy tests pass. Counter-tests exercise cloned
and foreign Case blocks, relabelled point/physical ownership, stale aliases,
changed width metadata, absent TimeState, divergent clock and different frame.
Four real AMR source emissions check the distinct IR16 contract and solve route.
All these are source/math/syntax evidence. ROOT exclusively owns rebuilt native
Serial/MPI reception. MAIN, reception checkout, ENV and SDK were not mutated.
