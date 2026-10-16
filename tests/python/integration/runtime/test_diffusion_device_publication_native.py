"""Native constitutive kernels and explicit device-to-host accepted-face publication."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pops
import pytest
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, TensorDiffusion
from pops.time import FixedDt
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import (
    compile_profile_plan_once,
    native_profile,
    profile_gather_value,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def build_diffusion_publication(kind, *, n=8, dt=1e-4):
    frame = Rectangle(
        "diffusion_publication", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = pops.Model("authored_" + kind, frame=frame)
    components = ("a",) if kind == "nonlinear" else ("a", "b")
    state = model.state("inventory", components=components)
    metric = np.asarray(((0.2, 0.03), (0.03, 0.12)))
    jacobian = np.asarray(((2.0, 0.25), (0.25, 1.5)))
    if kind == "nonlinear":
        flux = model.diffusive_flux(
            "authored_flux", state=state, value=math.CoeffGradient(state[0] ** 3, 0.1)
        )
        numerical = Diffusion(flux=flux)
    else:
        tensor = tuple(tuple(float(value) for value in row) for row in metric)
        variables = tuple(
            sum(
                float(coefficient) * state[column]
                for column, coefficient in enumerate(row)
            )
            for row in jacobian
        )
        flux = model.diffusive_flux(
            "authored_flux",
            state=state,
            value=tuple(math.CoeffGradient(value, tensor) for value in variables),
        )
        numerical = TensorDiffusion(flux=flux)
    rate = model.rate("authored_balance", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("diffusion_publication_" + kind)
    block = case.block("material", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, numerical)
    case.numerics(numerics, block=block)
    program = pops.Program("authored_euler_" + kind)
    q = program.state(block[state])
    derivative = rate(q.n)
    program.commit(
        q.next,
        program.value("accepted", q.n + program.dt * derivative, at=q.next.point),
    )
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block[state], value=BindArray(), projection=ConservativeCellAverage()
        )
    )
    layout = Uniform(
        CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes))
    )
    y, x = np.meshgrid(
        (np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij"
    )
    initial = np.asarray((1.2 + 0.1 * np.cos(2 * np.pi * (x + y)),))
    if kind == "tensor":
        initial = np.concatenate(
            (initial, (1.6 + 0.15 * np.sin(2 * np.pi * (2 * x - y)))[None])
        )
    expected, upper_flux = discrete_oracle(kind, initial, metric, jacobian, dt)
    return case, layout, initial, expected, upper_flux, block[state]


def discrete_oracle(kind, initial, metric, jacobian, dt):
    """Periodic finite differences independently assembled with NumPy rolls."""
    n = initial.shape[-1]
    transformed = (
        initial**3
        if kind == "nonlinear"
        else np.einsum("ab,bij->aij", jacobian, initial)
    )
    if kind == "nonlinear":
        upper = tuple(
            0.1 * n * (np.roll(transformed, -1, axis=2 - axis) - transformed)
            for axis in range(2)
        )
    else:
        gradient = tuple(
            0.5
            * n
            * (
                np.roll(transformed, -1, axis=2 - axis)
                - np.roll(transformed, 1, axis=2 - axis)
            )
            for axis in range(2)
        )
        central = tuple(
            sum(metric[axis, tangent] * gradient[tangent] for tangent in range(2))
            for axis in range(2)
        )
        curvature = tuple(
            0.5
            * n
            * metric[axis, axis]
            * (
                np.roll(transformed, -1, axis=2 - axis)
                - 2 * transformed
                + np.roll(transformed, 1, axis=2 - axis)
            )
            for axis in range(2)
        )
        upper = tuple(
            0.5
            * (
                central[axis]
                + np.roll(central[axis], -1, axis=2 - axis)
                + curvature[axis]
                - np.roll(curvature[axis], -1, axis=2 - axis)
            )
            for axis in range(2)
        )
    rate = sum(
        n * (upper[axis] - np.roll(upper[axis], 1, axis=2 - axis)) for axis in range(2)
    )
    return initial + dt * rate, upper


@pytest.mark.parametrize("kind", ("nonlinear", "tensor"))
def test_installed_diffusion_publishes_exact_device_faces(
    kind, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    native, world = native_profile(
        dimension=2, scope="diffusion device publication " + kind
    )
    case, layout, initial, expected, upper, subject = collective_call(
        world, lambda: build_diffusion_publication(kind)
    )
    plan = collective_call(
        world, lambda: pops.resolve(pops.validate(case), layout=layout)
    )
    artifact = compile_profile_plan_once(
        world,
        plan,
        route="diffusion publication " + kind,
        compile_artifact=pops.compile,
    )
    runtime = collective_call(
        world,
        lambda: pops.bind(
            artifact,
            initial_values={subject: np.ascontiguousarray(initial)},
            resources={"execution_context": artifact_execution_context(artifact)},
        ),
    )
    report = collective_call(
        world, lambda: pops.run(runtime, t_end=1e-4, max_steps=1, console=False)
    )
    actual = collective_call(world, lambda: runtime.state_global("material"))
    local_records = collective_call(world, runtime._executor._program_exchange_records)
    record_batches = profile_gather_value(world, local_records)
    with collective_check(world):
        assert report.accepted_steps == 1
        environment = dict(native.runtime_environment_report())
        if os.environ.get("POPS_REQUIRE_DEVICE") == "1":
            assert environment["kokkos_backend"] == "Cuda", environment
            assert environment["field_memory_space"] == "device", environment
        assert environment["kokkos_backend"] != "unknown", environment
        records = [row for batch in record_batches for row in batch]
        n = initial.shape[-1]
        assert len(records) == initial.shape[0] * 4 * n * n
        increments = np.zeros_like(initial)
        identities = set()
        for row in records:
            pieces = row["quadrature_identity"].split("/")
            i, j = (int(value) for value in pieces[0].split(":")[1:])
            axis, side = (int(piece.split(":")[1]) for piece in pieces[1:3])
            component = int(pieces[3].split(":")[1]) if len(pieces) == 4 else 0
            key = (component, i, j, axis, side)
            assert key not in identities
            identities.add(key)
            a, b = j, i
            if side == 0:
                if axis == 0:
                    b = (b - 1) % n
                else:
                    a = (a - 1) % n
            np.testing.assert_allclose(
                row["numerical_flux"],
                upper[axis][component, a, b],
                rtol=2e-12,
                atol=2e-13,
            )
            assert row["multiplicity"] == 1
            assert row["orientation"] == (-1 if side == 0 else 1)
            assert row["temporal_weight"] == 1e-4
            assert row["face_measure"] == 1 / n
            increments[component, j, i] += row["integrated_amount"]
        np.testing.assert_allclose(
            increments, (expected - initial) / n**2, rtol=2e-12, atol=2e-13
        )
        if actual is not None:
            actual = np.asarray(actual).reshape(initial.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(
                np.mean(actual - initial, axis=(1, 2)), 0, rtol=0, atol=2e-13
            )
        path = Path(artifact.program.so_path)
        record_property("native_environment", str(environment))
        record_property(
            "program_dso_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
        )
        record_property("accepted_face_incidences", len(records))
        record_property(
            "ledger_max_error",
            float(np.max(np.abs(increments - (expected - initial) / n**2))),
        )
