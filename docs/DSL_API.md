# Python DSL API

The Python DSL is an authoring layer for compiled C++ artifacts. It is not a
Python numerical runtime.

The public route is:

```python
compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)

sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(
    compiled,
    instances={"plasma": {"model": module, "initial": U0, "spatial": spatial}},
    params=params,
    aux=aux,
    solvers=solvers,
)
sim.step_cfl(0.4)
```

## Layers

- `pops.physics` lets users write equations with physical names and lowers to
  `pops.model`.
- `pops.model` is the operator-first IR: states, fields, signatures, operators,
  capabilities, and handles.
- `pops.time` is the time-program language.
- `pops.lib.time` contains ready-made time-program macros.
- `pops.mesh`, `pops.fields`, `pops.numerics`, and `pops.solvers` contain typed
  descriptors used by codegen and runtime validation.
- `pops.codegen` lowers the model/program pair to generated C++ and builds the
  compiled problem artifact.
- `pops.runtime` owns explicit runtime facades and profiling helpers.

## Rules

Strings name user objects. Typed descriptors choose behavior. Handles reference
operators after declaration.

```python
module = physics_model.to_module()
ops = module.operator_registry()

T = Program("advance").bind_operators(module)
U = T.state("U", block="plasma")

fields = T.call(ops.get("fields_from_state"), U.n)
rate = T.call(ops.get("explicit_rate"), U.n, fields)
T.define(U.next, U.n + T.dt * rate)
T.commit("plasma", U.next)
```

There is no public alternate front door for assembling or executing a run.
Private runtime seams may exist for pybind/C++ wiring and tests, but user
documentation must use the compiled problem route above.
