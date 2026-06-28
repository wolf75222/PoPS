# Add a new case

A *case* is a named scenario that builds a model, wires it into an `pops.System`, sets the
initial data, and runs the time loop. Cases are not part of `adc_cpp`: they live in the
companion `adc_cases` repository, one folder per case, each importing the installed `pops`
Python package. This guide assumes you can already build and import `pops`; if not, start
with the [installation guide](../getting-started/installation.md).

The convention is one folder per case under `adc_cases`. The folder holds a runnable
Python script that imports `pops`, composes a model, plugs it into a system, and steps it.

## Steps

1. Build and install `pops` so the case can import it. Use the `python` preset, which builds the
   importable `pops` module. The `serial` preset leaves `POPS_BUILD_PYTHON=OFF`, so `import pops`
   would later fail. Use the `python-parallel` preset instead for the multi-thread (Kokkos
   OpenMP) variant.

   ```bash
   cmake --preset python && cmake --build --preset python
   ```

2. In your clone of `adc_cases`, create one folder for the case. Replace `CASE_NAME` with a
   short name for the scenario.

   ```bash
   mkdir adc_cases/CASE_NAME
   ```

3. Write the case script in that folder. Author the physics with `pops.physics.Model`, declare the
   typed elliptic field with `pops.fields.PoissonProblem`, assemble a `pops.Case`, then compile and
   bind it. See the [models overview](../models/index.md) for the model fronts.

   ```python
   import pops
   import pops.time as T
   from pops.mesh.cartesian import CartesianMesh
   from pops.mesh.layouts import Uniform
   from pops.fields import PoissonProblem
   from pops.fields.bcs import Periodic
   from pops.fields.rhs import ChargeDensity
   from pops.solvers.elliptic import GeometricMG
   from pops.codegen import Production
   from pops.math import laplacian

   # m = pops.physics.Model(...): author the physics (see the models overview).
   poisson = PoissonProblem(name="phi", unknown="phi",
                            equation=(-laplacian("phi") == ChargeDensity.from_blocks("ne")),
                            bcs=(Periodic(),), solver=GeometricMG())

   case = (pops.Case(layout=Uniform(CartesianMesh(n=96, L=1.0, periodic=True)))
           .block("ne", physics=m).field(poisson).time(T.Program("euler")))

   compiled = pops.compile(case, backend=Production())
   sim = pops.bind(compiled, state={"ne": ne0})   # ne0: 2D initial density
   sim.run(0.1, cfl=0.4)
   ```

   Replace `ne0` with a 2D array holding the initial density. For an adaptive run, swap the layout
   to `pops.mesh.layouts.AMR(mesh, max_levels=2, ratio=2)` and author the refinement with
   `case.amr.refine(...)`.

4. Run the case from its folder.

   ```bash
   python adc_cases/CASE_NAME/run.py
   ```

## Next steps

- Follow the [tutorial](../getting-started/tutorial.md) for the full reduced-diocotron walkthrough.
- Read the [models overview](../models/index.md) to choose between native bricks, the DSL, and a hybrid model.
