# Original spatial field evolution, version 1

This source change closes the composition seam identified in `28b`: a spatial
FieldProblem can now declare an evolved conserved accumulation, use the duration
actually issued by the native cadence frame, observe the physical unknowns, and
publish that same accumulation through the ordinary Program commit protocol.

The declared equations are, for each evolved unknown,

\[
Q_i(q^+) - \tau R_i(q^+) = Q_i^n,
\qquad \tau = a\,\Delta t_{\mathrm{issued}},\quad a>0.
\]

Auxiliary unknowns retain their explicitly supplied original equations. For the
enthalpy instance, `Q(T)=H(T)` gives `H(T+) - tau div(k(T+) grad(T+)) = Hn`.
Neither a linearized heat capacity nor a second accumulation is substituted.
Other M13 nonlocal, radiative, and Poisson closures remain distinct obligations.

## Public authoring

```python
stage = EvolvedOriginalFieldStage(
    "energy-relation",
    unknowns=(T,),
    evolved_unknowns=(T,),
    accumulation=(Reaction(T, 1 + Value(T)),),
    spatial_rhs=(DivCoeffGrad(T, 1),),
    previous=(energy[0],),
    tau=P.temporal_tau(P.dt, at=u.next.point),
    boundaries=physical_boundaries,
)
field = case.field(stage.problem, original_nonlinear_discretization)
request = field.bind_program_inputs(
    program=P, values={block[energy]: u.n}, at=u.next.point,
    solver=field.default_program_solver(),
)
solution = field.observe(
    P.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
)
physical_temperature = solution[field[T]]
next_energy = solution.evolved_state(target=u.next)
P.commit(u.next, next_energy)
```

`TemporalTau@1` is a contextual authoring capability of this exact Program and
evaluation point. It accepts a positive exact multiple of `P.dt`; a literal
requested step or a duration borrowed from a homonymous Program is refused.
The old context-free `P.dt` coefficient is unchanged. The duration declaration
stays readable after the mutable builder is released, while further authoring
through that expired capability is refused.

The native consumer freezes `ctx.step_dt()` after setting the solve stage time,
votes on positive finiteness and exact all-rank equality, and checks the same
frame/attempt/point authority during residual evaluations. No requested controller
step is substituted for a shortened issued frame. This consumer currently uses
the declared ordinary native Program clock; additional clock realizations need
their own authority implementation.

`EvolvedOriginalFieldStage@1` records the full unknown tuple, Q, R, auxiliary
equations, previous State quantities, and duration authority in the original
FieldProblem. Case/ProgramFieldPlan validation rebuilds `Q-tau*R=Qn` and compares
it to the registered original equations. The publication node reuses the exact
consumed solve and frozen captures. It has no independent Q argument. Its target
is the exact next endpoint of one conservative State, in that State's component
order. Permuting physical field unknowns does not permute the conserved tuple.

Q is explicitly represented as a pointwise function of the piecewise constant
cell field and published as a cell-volume mean. This is not a claim that
`mean(H(T)) == H(mean(T))` for an arbitrary within-cell reconstruction. Nonlocal
accumulations are refused by this realization. Existing physical field
observations remain available separately.

## Native route and resource cost

The emitted code uses the existing original spatial Newton provider, including
full original residual evaluation, original finite-difference JVP, explicit
SolveOutcome consumption, and the existing transaction commit. D(candidate) uses
Euler's frozen PerCandidate@1/TrueCorrectionResidual@1 implementation. This bridge
does not alter Newton tolerances, budgets, preconditioners, or legacy policies.

The Uniform projection allocates one native MultiFab with the target Q width;
the AMR projection uses the existing scratch State at each published hierarchy
level. It copies the previous State before active-cell evaluation, retaining
covered/inactive values, and checks layout, distribution, rank, local patch count,
component widths, local exceptions, the completion fence, and global finiteness
before publication. Added work is one local Q kernel and error/finite votes;
storage is O(local cells × conserved components), beyond the existing field solve.
No model-specific operation is added to the runtime and no Python numerical
solver executes these equations.

ProgramIR version 12 is selected only when this duration or publication contract
is present. Legacy programs retain their previous IR versions and emitted bytes.
There are no C++ header, ABI, wire codec, installed environment, or SDK edits in
this commit.

## Source receipt and limits

Base: `2b4d01d4f2eaeb69bc7f32c7dc063ebde5ca2e81`, containing bind@2 and
Euler's D@3, allocation-drain, and TrueCorrectionResidual follow-ups. The public
probes exercise exact/fractional duration declarations, scalar and two/three
component permutations, coupled auxiliary equations, D(candidate), Uniform and
Composite AMR admission/emission, immutable capabilities, and refusals of
foreign duration, nonlocal Q, changed publication Q, and wrong endpoints.

Source command (no native dimension selection):

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q --tb=short \
  tests/review/test_sol61_evolved_original_field_stage.py \
  tests/python/unit/fields/test_nonlinear_mixed_field_problem.py \
  tests/python/unit/fields/test_amr_original_field_codegen.py \
  tests/python/unit/fields/test_m27_mixed_public_source.py \
  tests/python/unit/time/test_implicit_stage_request.py
```

Legacy comparison runs `tests/review/sol61_spatial_field_legacy_parity.py` against
both this source tree and an exact Git archive of the base. Four complete legacy
IR/C++ receipts cover implicit stage, nonlinear-map implicit stage, mixed linear,
and permuted mixed linear cases. Results are recorded in the source receipt below.

Final coherent result: **59 PASS**, 158.21 seconds. Ruff on all 16 changed/new
Python files and `git diff --check` pass. The four legacy IR and emitted C++ digest
pairs are identical to the base archive; the explicit hashes and source scope are
in `evolved_original_field_stage_source_receipt.json`. An earlier adversarial
test used `dataclasses.replace` on a non-dataclass FieldProblem and failed in the
test itself; it now constructs the altered FieldProblem through its public
constructor and reaches the intended equation-authority refusal.

Native obligations remain pending: compile/relink the generated Uniform/AMR
consumers, execute physical saved-state residual and Q-publication checks with
exact issued/fractional durations, and receive serial/MPI ownership, empty-rank,
rejection, rollback, retry, history, and checkpoint behavior. Source admission and
emission are not a native scientific M06/M13 qualification.
