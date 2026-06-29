# Add a DSL model

Use `pops.physics.Model` to write physical equations, then lower to
`pops.model.Module`.

```python
m = Model("gas")
U = m.state("U", components=["rho", "mx", "my"], roles={"rho": "density"})
# declare fluxes, fields, sources, rates...
module = m.to_module()
```

A program calls operator handles from the module:

```python
program = Program("advance").bind_operators(module)
ops = module.operator_registry()
U = program.state("U", block="gas")
R = program.call(ops.get("explicit_rate"), U.n)
program.define(U.next, U.n + program.dt * R)
program.commit("gas", U.next)
```

Compile and install:

```python
compiled = pops.compile_problem(model=module, program=program, backend=Production(), layout=layout)
sim = pops.System(n=mesh.n, L=mesh.L, periodic=mesh.periodic)
sim.install(compiled, instances={"gas": {"model": module, "initial": U0, "spatial": spatial}})
sim.step_cfl(0.4)
```

The DSL does not compile directly and does not run kernels. It writes model IR.
