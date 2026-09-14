# Examples

Start with the [tutorials](../docs/tutorials/README.md) for linear Python scripts
that introduce the physical model before simulation setup.

- [final/](final/README.md) contains the four complete executable acceptance
  examples: scalar AMR, coupled fields, IMEX AMR and HyQMOM15. Their output and
  restart checks are exercised by `tests/python/examples/final/`.
- [migration/](migration/) contains eight smaller public-API workflows and the
  external-component compatibility witness used during the architecture migration.
  The workflow CLI lists the supported cases:

  ```sh
  env -u PYTHONPATH python examples/migration/r1_workflows.py --list
  ```

Performance measurement belongs in [benchmarks/](../benchmarks/README.md); these
examples demonstrate API usage and check numerical behavior.
