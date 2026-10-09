# M09 Hoffart: finite source witness and disk PDE

The direct finite-support gap identified in the historical preflight below is
now implemented and received. Native monolithic/condensed 8+4 DOF equations,
permutations, rebound captures, singular chosen pivot and owner-only invalid
evaluation pass ten tests in serial and on each of two MPI ranks. The
[converged reception](finite_amr_converged_reception.md) archives the exact
SDK396719/Dim2aed8 sources, failures and 32 actual saved states; independent
NumPy recomputation satisfies the unchanged 1e-11 original-equation criterion.
This resolves that finite incidence-map witness, without qualifying a global
meshed mixed PDE or extending the separate Hoffart disk campaign.

## Historical preflight before the finite-support extension

This decision is fixed before a new M09 native run. The v0.4.0 corpus M09
specifies the **finite** source witness in
`PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/legacy/v0.2.0/examples/coupled_native.py`:
eight midpoint velocity unknowns, four midpoint potential unknowns, random
`G` of shape 8×4 from NumPy `default_rng(20260928)`, `D=-G.T`, `K=G.T@G`,
`rho=(1,1.1,.9,1.2)`, `s=0.03`, and `alpha=0.5`. The entire original system is

```text
(I-sJ) v + sG phi             = v_old
s alpha D Rho v + K phi      = K phi_old
J = [[0,I],[-I,0]], Rho = diag(rho,rho).
```

The independent [oracle](../../../examples/migration/scientific/api040_m09_hoffart_oracle.py)
forms both the 12×12 monolithic system and the Schur complement from these
original matrices, reconstructs the Crank–Nicolson endpoint, and measures the
original residual, energy defect, velocity norm, and phase separately. It
never loads or executes old C++ kernels. Predeclared finite acceptance is
`max(original residual mono, original residual Schur, mono–Schur error,
CN energy defect) < 1e-11`, exactly as in M09. Norm and phase are reported;
the corpus does not give a target value or threshold for either. The four
independent pure-oracle tests include wrong-coupling and state-perturbation
negatives.

The random dense `G` is **not** a mesh gradient, and the N=8/Nphi=4 witness is
not a disk PDE or a PoPS result. The public `Program.condensed_coeffs`,
`condensed_rhs`, and `condensed_reconstruct` operators implement a spatial
finite-volume gradient/divergence coupled to a cell-local magnetic rotation.
Their `gradient_map` is a cell-local coordinate transform; it is not an
arbitrary dense incidence map from four potential degrees of freedom to eight
velocity degrees of freedom. `LocalResidual` solves one cell-local nonlinear
equation and likewise cannot couple this random 8×4 spatial map. Mapping this
finite witness into those operators would change its mathematical problem.
This is an **EXPR/IMPL gap for direct native reproduction of the finite M09
matrix witness**, not an assertion that the disk PDE is unavailable.

The existing public
[`01_mpi_kokkos_hoffart_euler.py`](../../tutorials/diocotron/01_mpi_kokkos_hoffart_euler.py)
authors the **full** barotropic Euler–Poisson conducting-disk problem with
`Model`: density and both momentum components, mapped radial/angular
transport, rotation and electrostatic source, analytic coordinate maps,
Gauss-law Schur solve, conducting Dirichlet outer wall, regularity Neumann
inner boundary, angular periodicity, and source-first CN plus SSPRK2
transport. It is routed by
[`api040_m09_hoffart.py`](../../../examples/migration/scientific/api040_m09_hoffart.py)
with `disk`. The disk uses the benchmark parameters and FV geometry from that
tutorial; its native scientific criteria and evidence are maintained in the
tutorial's README and campaign status. The finite oracle's `1e-11` threshold
does **not** transfer to the disk's different discretization or field solver.
The handoff states that the paper's disk FV run is still to be established for
M09, so neither witness qualifies the other.

Commands from a repository checkout with the installed `pops-api040` env:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python examples/migration/scientific/api040_m09_hoffart.py finite
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath= tests/python/unit/numerics/test_api040_m09_hoffart_oracle.py
# Full disk: set POPS_RUN_OUTPUT/POPS_T_END/POPS_MAX_LEVELS as documented in
# docs/tutorials/diocotron/README.md, then run through the installed MPI env:
mpiexec -n 2 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python examples/migration/scientific/api040_m09_hoffart.py disk
```

Source-only check on this checkout: the disk tutorial through section 8
completed public `validate` and `resolve`, then
`emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")`.
It emitted 74,258 characters, with one resolved block and the condensed
route present. This check did not compile, bind, run, inspect a native state,
or qualify the disk. The finite oracle measured original residuals
`2.22e-15`, mono–Schur error `4.44e-16`, and CN energy defect `3.55e-15`
with NumPy from the dedicated environment.
