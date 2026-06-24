# Blackboard-style DSL (`adc.physics`, `adc.math`)

The blackboard DSL is the layer-1 user API of spec 3: it lets you write a model and
a time scheme the way they appear on a blackboard, and lowers them to the
operator-first IR ({doc}`operator-modules`) that the compiler and runtime consume.
It adds no new execution path: `adc.physics` reuses the {doc}`symbolic-dsl` codegen,
and the board time sugar lowers to the same Program IR as the {doc}`time-program`
primitive calls.

```{admonition} Three layers
:class: note
Layer 1 (this page) is the blackboard notation. Layer 2 is the typed operator-first
IR (`adc.model.Module`: spaces, signatures, operators). Layer 3 is the C++ that
executes -- native bricks in `include/adc` and the generated `problem.so`. Python
describes and composes; C++ runs.
```

## Notation (`adc.math`)

`adc.math` is numerics-free notation: `ddt` / `rate` (time derivative), `div`
(flux divergence), `grad` / `dx` / `dy` (field gradient), `laplacian` (elliptic
operator), `sqrt`, `integral` (invariant value), `unknown` (a solve unknown), `==`
(an equation) and `@` (operator application). These build a small board IR that the
model and time APIs destructure; they carry no arrays.

## Authoring a model (`adc.physics.Model`)

A model is written as equations over a state, primitives, parameters, a flux, an
elliptic field solve, sources and local linear operators:

```python
from adc.physics import Model
from adc.math import sqrt, grad, div, laplacian, ddt

m = Model("euler_poisson_lorentz")
U = m.state("U", components=["rho", "mx", "my"],
            roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
rho, mx, my = U
u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
cs2 = m.param("cs2", 1.0)
p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))

F = m.flux("F", on=U,
           x=[mx, mx * u + p, mx * v],
           y=[my, my * u, my * v + p],
           waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})

phi = m.field("phi")
m.solve_field("fields_from_state",
              equation=(-laplacian(phi) == cs2 * (rho - 1.0)),
              outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
              solver="geometric_mg")
E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
A = m.source("electric", on=U, value=[0.0 * rho, rho * E.x, rho * E.y])

Bz = m.aux("B_z")
C = m.local_linear_operator("lorentz", on=U,
                            matrix=[[0, 0, 0], [0, 0, Bz], [0, -Bz, 0]])

m.rate("explicit_rate", ddt(U) == -div(F) + A)
m.operator("implicit_operator", C)
m.check()
```

`m.module` is the typed `adc.model.Module` this lowers to (state space, field space,
and the typed operators `explicit_rate`, `electric`, `lorentz`, `fields_from_state`).
The spec-1 PDE methods (`m.flux` / `m.source_term` / `m.linear_source` /
`m.elliptic_field` on `adc.dsl.Model`) remain valid; the board API is sugar over them.

## Authoring a time scheme (`adc.time.Program` sugar)

The board time sugar mirrors the blackboard stages and lowers to the same IR as the
primitive `solve_fields` / `linear_combine` / `solve_local_linear` / `commit` calls:

```python
from adc.time import Program
from adc.math import unknown

T = Program("predictor_corrector")
dt = T.dt
U_n = T.state("plasma")
f_n = T.fields("fields_n", from_state=U_n)
R_n = T.rhs(name="R_n", state=U_n, fields=f_n, flux=True, sources=["electric"])
U_star = T.solve("U_star",
                 (T.I - dt * T.linear_source("lorentz")) @ unknown("U_star")
                 == U_n + dt * R_n)
T.commit("plasma", U_star)
```

`T.define` names an affine combination or keeps a `rate(U) == operator(...)` right
hand side; `T.commit_many({...})` commits several coupled blocks atomically;
`T.state_set` builds a coherent set of stage states for a multi-block field solve.
The `P.call` / `P.solve_local_linear` builder style is unchanged and still available.

## Typed brick catalog (`adc.lib`)

`adc.lib` is a catalog of descriptors and IR macros, never a Python numerics library.
`adc.lib.riemann.HLLC()` and `adc.lib.reconstruction.WENO5Z()` compute nothing: they
name native C++ bricks (`adc::numerics::fv::HLLCFlux`, `adc::numerics::fv::Weno5`) and
carry the requirements those bricks place on the model. `adc.lib.time.*` are macros
that build Program IR (they forward to `adc.time` `std`). Other namespaces:
`limiters`, `spatial`, `fields`, `solvers`, `preconditioners`, `diagnostics`,
`projections`, `invariants`.

## Generic multi-output and invariants

`adc.model.RateBundle` is a typed multi-output of arbitrary arity: a coupled operator
returns one `Rate(StateSpace)` per participating block, and a wrong rate on a wrong
state is rejected. `adc.physics.Model.invariant(name, expression, over=...)` declares
a generic invariant from an `integral(...)`; nothing about mass, charge, momentum or
energy is built in.

## Examples

See `examples/spec3/`: `board_euler_poisson_lorentz.py`,
`board_time_predictor_corrector.py` (asserts board IR == primitive IR),
`rate_bundle_collisions.py`, `invariant_generic.py`.

## Status

The native HLLC/Roe model-capability hook codegen, the generic multi-species runtime
(`commit_many` across distinct blocks at runtime, `StageStateSet` overrides), the
unified Program scheduler and the per-node profiling report are tracked as follow-ups
(ADC-456, ADC-457, ADC-458, ADC-459). The board authoring, the IR lowering, the typed
descriptors and the invariant declaration land here.
