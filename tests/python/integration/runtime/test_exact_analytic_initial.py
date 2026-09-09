"""Public Python Gaussian/mixture qualification through resolved native cell integrals.

Run separately with POPS_NATIVE_DIM=1, 2 and 3. Every run covers uniform N=16/32 and
AMR reprojection N=16->32 and N=32->64, two parameter binds per compiled artifact.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

import pops
from pops.analytic import constant, param
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.codegen import Production
from pops.domain import CartesianDomain
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic, Gaussian
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_dsl_runtime_params import _binary_paths, _binary_fingerprints

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200
ROOT = Path(__file__).resolve().parents[4]


def _normalized_gaussian(frame, centers, width):
    root = math.sqrt(width)
    integral = math.prod(math.sqrt(math.pi) / (2 * root) *
                         (math.erf(root * (1 - center)) + math.erf(root * center))
                         for center in centers)
    return Gaussian(frame=frame, center=dict(zip(frame.axes, centers)),
                     background=1 - integral, inverse_width=width).as_analytic()


def _case(target, n, dim, mixture, *, tail_sign=None):
    bounds = ((0.0,) * dim, (1.0,) * dim)
    if tail_sign is not None:
        assert dim == 1 and tail_sign in (-1, 1)
        bounds = ((8.0,), (8.1,)) if tail_sign == 1 else ((-8.1,), (-8.0,))
    frame = CartesianDomain("renamed-mixture-support" if mixture else "neutral-gaussian-support",
                            *bounds).frame()
    model = pops.Model("zero-transport", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux("flux", frame=frame, state=state,
                       components={axis: (0 * rho,) for axis in frame.axes},
                       waves={axis: (0 * rho,) for axis in frame.axes})
    rate = model.rate("rate", equation=ddt(state) == -div(flux))
    case = pops.Case("exact-initial-%s" % target)
    block = case.block("renamed-density", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                         reconstruction=reconstruction.FirstOrder(),
                                         riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(0.001))
    case.program(program)
    weight = case.param(RuntimeParam("mixture-weight", default=1.0))
    if tail_sign is not None:
        first = Gaussian(frame=frame, center={frame.axes[0]: 0.0}, background=0.0,
                         inverse_width=1.0).as_analytic()
        second = Gaussian(frame=frame, center={frame.axes[0]: 0.1 * tail_sign},
                          background=0.0, inverse_width=1.0,
                          amplitude=1.0 if mixture else 0.0).as_analytic()
    else:
        first = _normalized_gaussian(frame, (0.35,) + (0.55,) * (dim - 1), 80.0)
    if tail_sign is None and mixture:
        second = _normalized_gaussian(frame, (0.65,) + (0.45,) * (dim - 1), 120.0)
    elif tail_sign is None:
        from pops.analytic import CellBounds
        second = Analytic(frame=frame, components=(constant(1),),
                           cell_integrals=(CellBounds(frame).measure,))
    # Both the physical profile and its explicit integral are composed through public Python.
    profile = Analytic(frame=frame,
                       components=(param(weight) * first.components[0] +
                                   (1 - param(weight)) * second.components[0],),
                       cell_integrals=(param(weight) * first.cell_integrals[0] +
                                       (1 - param(weight)) * second.cell_integrals[0],))
    case.initials.add(InitialCondition(state=block_state, value=profile,
                                       projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=(n,) * dim, periodic=PeriodicAxes(frame.axes))
    if target == "system":
        layout = Uniform(grid)
    else:
        threshold = case.param(RuntimeParam("refine-threshold", default=0.5))
        transfer = AMRTransfer()
        transfer.state(block_state, StateTransfer())
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
                     tagging=AMRTagging(rules=(Tag(ValueExpr(block_state) > case.value(threshold)),
                                              Buffer(cells=1)),
                                        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                                        conflict_policy=ConflictPolicy.REFINE_WINS),
                     regrid=AMRRegrid(schedule=every(2, clock=program.clock)), transfer=transfer,
                     execution=AMRExecution.synchronous())
    validated = pops.validate(case)
    return pops.resolve(validated, layout=layout, backend=Production(),
                         compile_options={"include": str(ROOT / "include")}), validated.resolve(weight)


def _oracle(n, centers, width):
    root = math.sqrt(width)
    factors = []
    total_integral = 1.0
    for center in centers:
        total_integral *= math.sqrt(math.pi) / (2 * root) * (
            math.erf(root * (1 - center)) + math.erf(root * center))
        factors.append(np.array([math.sqrt(math.pi) * n / (2 * root) * (
            math.erf(root * ((i + 1) / n - center)) - math.erf(root * (i / n - center)))
                                 for i in range(n)]))
    result = np.ones((n,) * len(centers))
    for axis, factor in enumerate(factors):
        shape = [1] * len(centers)
        shape[len(centers) - axis - 1] = n
        result *= factor.reshape(shape)
    return 1 - total_integral + result


@pytest.mark.parametrize("target", ["system", "amr_system"], ids=["uniform", "amr"])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("mixture", [False, True], ids=["gaussian", "mixture"])
def test_public_exact_initial_neutrality_and_reprojection(target, n, mixture,
                                                         isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    configured = os.environ.get("POPS_NATIVE_DIM")
    assert configured in {"1", "2", "3"}, "qualification requires explicit POPS_NATIVE_DIM=1/2/3"
    dim = int(configured)
    resolved, parameter = _case(target, n, dim, mixture)
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    for weight in ((0.25, 0.75) if mixture else (1.0, 0.5)):
        simulation = pops.bind(artifact, params={parameter: weight},
                               resources={"execution_context": artifact_execution_context(artifact)})
        levels = 1 if target == "system" else simulation.n_levels()
        assert levels == (1 if target == "system" else 2)
        for level in range(levels):
            cells = n * 2 ** level
            actual = np.asarray(simulation.get_state("renamed-density") if target == "system"
                                else simulation.block_level_state_global("renamed-density", level),
                                dtype=np.float64).reshape((cells,) * dim)
            first = _oracle(cells, (0.35,) + (0.55,) * (dim - 1), 80.0)
            second = _oracle(cells, (0.65,) + (0.45,) * (dim - 1), 120.0) if mixture else 1.0
            np.testing.assert_allclose(actual, weight * first + (1 - weight) * second,
                                       rtol=0, atol=128 * np.finfo(float).eps)
            residual = np.mean(actual.astype(np.longdouble) - 1)
            assert abs(residual) <= 128 * np.finfo(float).eps
        assert _binary_fingerprints(paths) == before


def _tail_oracle(n, sign, center):
    """Independent CPython erfc bound difference, with no additive background."""
    lower, upper = (8.0, 8.1) if sign == 1 else (-8.1, -8.0)
    dx = (upper - lower) / n
    values = []
    for cell in range(n):
        lo, hi = lower + cell * dx - center, lower + (cell + 1) * dx - center
        # Reflect negative intervals before evaluating positive erfc arguments.
        if hi < 0:
            lo, hi = -hi, -lo
        values.append(math.sqrt(math.pi) / (2 * dx) * (math.erfc(lo) - math.erfc(hi)))
    return np.asarray(values)


@pytest.mark.parametrize("sign", [-1, 1], ids=["negative-tail", "positive-tail"])
@pytest.mark.parametrize("mixture", [False, True], ids=["gaussian", "mixture"])
def test_public_exact_initial_preserves_nonzero_far_tails(sign, mixture,
                                                          isolated_native_cache, native_cxx,
                                                          kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    configured = os.environ.get("POPS_NATIVE_DIM")
    assert configured in {"1", "2", "3"}, "qualification requires explicit POPS_NATIVE_DIM"
    if configured != "1":
        pytest.skip("far-tail qualification is declared for native Dim=1")
    n = 8
    resolved, parameter = _case("system", n, 1, mixture, tail_sign=sign)
    artifact = pops.compile(resolved)
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    first, second = _tail_oracle(n, sign, 0.0), _tail_oracle(n, sign, 0.1 * sign)
    for weight in ((0.25, 0.75) if mixture else (1.0, 0.5)):
        simulation = pops.bind(artifact, params={parameter: weight},
                               resources={"execution_context": artifact_execution_context(artifact)})
        actual = np.asarray(simulation.get_state("renamed-density"), dtype=np.float64).reshape(n)
        expected = weight * first + ((1 - weight) * second if mixture else 0.0)
        assert np.all(expected > 0) and np.all(actual > 0)
        # An erf(hi)-erf(lo) implementation returns zero on every one of these cells.
        # There is no absolute tolerance that could hide cancellation of this tiny signal.
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
        assert _binary_fingerprints(paths) == before
