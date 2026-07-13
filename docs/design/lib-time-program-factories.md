# Canonical `pops.lib.time` Program factories

`pops.lib.time` provides configured Programs, not scheme objects or alternate executors. The two
final factories in this cutover take one Case-authenticated state instance and exact model operator
handles:

```python
U = tracer[model_state]

explicit = pops.lib.time.SSPRK2(U, rate=transport_rate, fields=field_solve)
imex = pops.lib.time.IMEX(
    U,
    explicit_operator=transport_rate,
    implicit_operator=stiff_local_map,
    fields_operator=field_solve,
    tableau=my_additive_runge_kutta_tableau,
)
```

Both calls return an ordinary `pops.Program`. They expand only through public Program operations:
`T.state(U)`, `T.call(...)`, `T.value(...)`, local solves, applications and `T.commit(...)`. The
equivalent manual operations normalize to the same `ProgramGraph` and semantic identity. Order and
SSP evidence are reconstructed from that graph; factory names never select a runtime route.

The state must be the live instance handle produced by `block[state]`. Operators must be typed
handles returned by model declarations. There is no `(block, state)` overload, free-string operator,
`bind_operators`, `linear_combine`, boolean flux selector, legacy IMEX alias or preset-specific native
stepper. A custom IMEX method is configured by an exact `AdditiveRungeKuttaTableau`; unsupported or
incomplete selections fail while the Program is being authored.
