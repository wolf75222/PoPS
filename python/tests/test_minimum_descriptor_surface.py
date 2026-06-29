"""Minimum typed-descriptor surface needed by the first operator-first example.

This guards the Spec corrective rule that public examples must be authorable with typed
objects only. Strings may name user objects, but they must not select algorithms,
layouts, backends, solvers or output policies.
"""

import pytest

pytest.importorskip("pops")


def _module(name):
    import importlib

    return importlib.import_module(name)


def _poisson_equation():
    from pops.ir.expr import Var
    from pops.math import laplacian, unknown

    phi = unknown("phi")
    charge = Var("charge", "cons")
    return phi, (-laplacian(phi) == charge)


def test_first_example_minimum_descriptors_import_and_compose():
    from pops.codegen import KokkosOpenMP, Production
    from pops.fields import PoissonProblem
    from pops.mesh import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.numerics.riemann import Rusanov
    from pops.numerics.reconstruction import MUSCL
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.numerics.spatial import FiniteVolume
    from pops.output import CheckpointPolicy, OutputPolicy
    from pops.params import ConstParam, Positive, RuntimeParam
    from pops.solvers.elliptic import GeometricMG

    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    layout = Uniform(mesh)
    spatial = FiniteVolume(
        riemann=Rusanov(),
        reconstruction=MUSCL(limiter=Minmod()),
    )
    phi, equation = _poisson_equation()
    field = PoissonProblem(
        name="phi",
        unknown=phi,
        equation=equation,
        solver=GeometricMG(),
    )
    backend = Production(platform=KokkosOpenMP())
    output = OutputPolicy(cadence=10, fields=["phi"], diagnostics=["mass"])
    checkpoint = CheckpointPolicy(cadence=50, restartable=True)
    runtime_alpha = RuntimeParam("alpha", default=1.0, domain=Positive())
    gamma = ConstParam("gamma", 5.0 / 3.0)

    assert layout.validate() is True
    assert field.validate() is True
    assert spatial.inspect()["options"] == {
        "reconstruction": "minmod",
        "riemann": "rusanov",
    }
    assert backend.inspect()["options"]["backend"] == "production"
    assert backend.inspect()["platform"]["options"]["device"] == "openmp"
    assert output.inspect()["category"] == "output_policy"
    assert checkpoint.inspect()["category"] == "checkpoint_policy"
    assert runtime_alpha.validate() is True
    assert gamma.inspect()["options"]["value"] == pytest.approx(5.0 / 3.0)


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda: _module("pops.mesh").CartesianMesh(n="64"), "CartesianMesh"),
        (lambda: _module("pops.mesh.layouts").Uniform("uniform"), "String algorithm"),
        (lambda: _module("pops.mesh.layouts").AMR(base="uniform"), "String algorithm"),
        (
            lambda: _module("pops.numerics.spatial").FiniteVolume(riemann="rusanov"),
            "String algorithm",
        ),
        (
            lambda: _module("pops.numerics.spatial").FiniteVolume(reconstruction="minmod"),
            "String algorithm",
        ),
        (
            lambda: _module("pops.codegen").Production(platform="openmp"),
            "platform must be a typed",
        ),
        (
            lambda: _module("pops.codegen.backends").lower_problem_backend("production"),
            "compile_problem",
        ),
        (
            lambda: _module("pops.output").OutputPolicy(format="hdf5"),
            "String algorithm",
        ),
        (
            lambda: _module("pops.output").OutputPolicy(cadence="every"),
            "String algorithm",
        ),
        (
            lambda: _module("pops.output").OutputPolicy(levels="all"),
            "String algorithm",
        ),
        (
            lambda: _module("pops.output").CheckpointPolicy(cadence="every"),
            "String algorithm",
        ),
    ],
)
def test_first_example_rejects_algorithmic_string_selectors(builder, message):
    with pytest.raises((TypeError, ValueError), match=message):
        builder()


def test_field_solver_selector_must_be_typed():
    from pops.fields import PoissonProblem

    phi, equation = _poisson_equation()
    with pytest.raises(TypeError, match="solver must be a typed"):
        PoissonProblem(
            name="phi",
            unknown=phi,
            equation=equation,
            solver="geometric_mg",
        )
