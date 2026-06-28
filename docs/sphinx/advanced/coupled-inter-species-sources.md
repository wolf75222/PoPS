# Coupled inter-species sources

Coupled sources operate on several block states at the same time. Declare them as
typed operators on the model layer, then consume the returned handle in a
`pops.time.Program`.

```python
T = Program("coupled_step")
e = T.state("Ue", block="electrons")
i = T.state("Ui", block="ions")

fields = T.solve_fields_from_blocks([e.n, i.n])
rates = collision_operator(e.n, i.n, fields)

T.define(e.next, e.n + T.dt * rates["electrons"])
T.define(i.next, i.n + T.dt * rates["ions"])
T.commit_many({"electrons": e.next, "ions": i.next})
```

The operator handle is a typed object returned by model authoring. Do not
reference coupled operators with string names in public program code.

Use the same program with `Uniform(...)` or `AMR(...)` layouts when the source
descriptor declares both routes.
