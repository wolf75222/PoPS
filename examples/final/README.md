# Executable acceptance examples

These complete examples exercise the public lifecycle
`validate -> resolve -> compile -> bind -> run`, numerical checks, scientific
outputs and checkpoint continuation. Start with the shorter
[tutorials](../../docs/tutorials/README.md) to learn the API.

| Example | What it exercises |
|---|---|
| [Scalar advection](EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py) | Conservative transport, explicit/preset SSPRK2 parity, AMR regridding and synchronization, leaf-cell error and restart |
| [Multiphysics](EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py) | Two state spaces, a shared elliptic field, qualified diagnostics and refusal of unsupported cross-layout reads |
| [IMEX AMR](EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py) | Explicit/implicit stages, field solves, subcycled AMR, rejected-attempt rollback and output/restart |
| [HyQMOM15](EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py) | Fifteen moments, conservative HLL transport, a periodic Poisson field, realizability and strict restart |

From the repository root, with the [build environment](../../README.md) active:

```sh
env -u PYTHONPATH python examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py --output-dir outputs/scalar-advection
```

All four scripts accept `--output-dir`; use `--help` for their individual options.
Multiphysics and HyQMOM15 also accept `--cells`. The corresponding tests live in
[`tests/python/examples/final`](../../tests/python/examples/final/), and
[`scripts/final_release_contract.py`](../../scripts/final_release_contract.py)
defines the release gate's required scripts and evidence.

The scripts and their assertions are the source of truth for their numerical
limits and output contracts. A successful execution covers its selected native
dimension, backend and MPI context; other configurations need their own runs.
