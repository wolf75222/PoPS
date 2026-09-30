"""Closed M26 finite interaction on twelve periodic quadrature degrees of freedom.

One native cell batches finite DOFs. They are not a discretized HPC spatial mesh.
No aggregation-diffusion PDE, spatial flux, or energy decay is claimed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.linalg import FiniteMeasure, FiniteSupport, FiniteSymmetricInteraction
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

from api040_m26_finite_oracle import NODES, PERMUTATION, density_pair, metrics
from api040_receipts import receipt_json


DIAGNOSTICS = (
    "energy_first", "energy_second", "directional_derivative",
    "pair_first_adjoint_second", "pair_action_first_second",
    "mass_first", "mass_second",
)
CRITERIA = {
    "density_error": 2e-14,
    "potential_error": 3e-13,
    "measured_pairing_error": 3e-13,
    "adjoint_defect": 3e-13,
    "energy_increment_identity": 3e-13,
    "time_error": 2e-14,
}


def build_case(*, permuted: bool = False):
    order = PERMUTATION if permuted else tuple(range(NODES))
    frame = CartesianDomain("finite_interaction", lower=(0.,), upper=(1.,)).frame(
        Cartesian1D())
    labels = tuple("q%d" % index for index in order)
    names = ("density_first", "density_second", "potential_first",
             "potential_second", "measured_pairing")
    models = {name: pops.Model(name, frame=frame) for name in names}
    states = {name: models[name].state(
        name, components=DIAGNOSTICS if name == "measured_pairing" else labels)
        for name in names}
    case = pops.Case("m26_finite_symmetric_interaction")
    blocks = {name: case.block(name, models[name], states=(states[name],)) for name in names}
    support = FiniteSupport("periodic_quadrature", labels)
    measure = FiniteMeasure(support, (1. / NODES,) * NODES)
    kernel = tuple(tuple(math.cos(2 * math.pi * abs(i - j) / NODES)
                         for j in order) for i in order)
    interaction = FiniteSymmetricInteraction(measure, kernel)
    program = pops.Program("m26_finite_pairing")
    current = {name: program.state(blocks[name][state]) for name, state in states.items()}
    first = support.bind(current["density_first"].n)
    second = support.bind(current["density_second"].n)
    potential_first = interaction.apply(first).materialize(
        program, "W_density_first", template=current["potential_first"].n,
        at=current["potential_first"].next.point)
    potential_second = interaction.adjoint(second).materialize(
        program, "W_adjoint_density_second", template=current["potential_second"].n,
        at=current["potential_second"].next.point)
    pa, pb = support.bind(potential_first), support.bind(potential_second)
    ones = support.bind((1,) * NODES)
    expressions = (
        .5 * measure.pair(first, pa),
        .5 * measure.pair(second, pb),
        measure.pair(second - first, pa),
        measure.pair(first, pb),
        measure.pair(pa, second),
        measure.pair(first, ones),
        measure.pair(second, ones),
    )
    measured = FiniteSupport("interaction_measures", DIAGNOSTICS).bind(
        expressions).materialize(
            program, "measured_interaction",
            template=current["measured_pairing"].n,
            at=current["measured_pairing"].next.point)
    for name in ("density_first", "density_second"):
        state = current[name]
        program.commit(state.next, program.value(
            name + "_preserved", 1 * state.n, at=state.next.point))
    for name, value in (("potential_first", potential_first),
                        ("potential_second", potential_second),
                        ("measured_pairing", measured)):
        program.commit(current[name].next, value)
    program.step_strategy(FixedDt(1.))
    case.program(program)
    subjects = {name: blocks[name][state] for name, state in states.items()}
    for subject in subjects.values():
        case.initials.add(InitialCondition(
            state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(1,), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subjects, order


def _collective_call(world, label, operation):
    result = None
    failure = b""
    try:
        result = operation()
    except Exception as exception:
        failure = (label + ": " + type(exception).__name__ + ": " + str(exception)).encode()
    failures = world.allgather_bytes(failure)
    if any(failures):
        raise RuntimeError("; ".join(row.decode() for row in failures if row))
    return result


def _initial_values(subjects, order, *, variation: bool):
    first, second = density_pair(variation=variation)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)) \
            or not np.all(first > 0) or not np.all(second > 0):
        raise ValueError("M26 witness requires two finite positive densities")
    return {
        subjects["density_first"]: np.ascontiguousarray(first[list(order)].reshape(NODES, 1)),
        subjects["density_second"]: np.ascontiguousarray(second[list(order)].reshape(NODES, 1)),
        subjects["potential_first"]: np.zeros((NODES, 1)),
        subjects["potential_second"]: np.zeros((NODES, 1)),
        subjects["measured_pairing"]: np.zeros((len(DIAGNOSTICS), 1)),
    }


def _assess_saved(path: Path, *, variation: bool) -> dict:
    with np.load(path) as saved:
        first = np.asarray(saved["density_first"], dtype=float)
        second = np.asarray(saved["density_second"], dtype=float)
        pa = np.asarray(saved["potential_first"], dtype=float)
        pb = np.asarray(saved["potential_second"], dtype=float)
        observed = np.asarray(saved["measured_pairing"], dtype=float)
        time = float(saved["time"])
    if any(value.shape != (NODES,) for value in (first, second, pa, pb)) \
            or observed.shape != (len(DIAGNOSTICS),) \
            or not all(np.all(np.isfinite(value)) for value in (
                first, second, pa, pb, observed)):
        raise AssertionError("M26 saved native finite vectors have invalid shapes/values")
    expected_first, expected_second = density_pair(variation=variation)
    expected = metrics(expected_first, expected_second)
    expected_measures = np.asarray([expected[name] for name in DIAGNOSTICS])
    energy_increment = expected["energy_second"] - expected["energy_first"]
    direction = second - first
    increment_from_saved = observed[2] \
        + .5 / NODES * float(np.dot(direction, pb - pa))
    values = {
        "density_error": float(max(np.max(np.abs(first - expected_first)),
                                   np.max(np.abs(second - expected_second)))),
        "potential_error": float(max(np.max(np.abs(pa - expected["potential_first"])),
                                     np.max(np.abs(pb - expected["potential_second"])))),
        "measured_pairing_error": float(np.max(np.abs(observed - expected_measures))),
        "adjoint_defect": float(abs(observed[3] - observed[4])),
        "energy_increment_identity": float(abs(increment_from_saved - energy_increment)),
        "time_error": abs(time - 1.),
    }
    for name, limit in CRITERIA.items():
        if not math.isfinite(values[name]) or values[name] > limit:
            raise AssertionError("M26 saved-state criterion failed: " + name)
    return values


def run_and_archive(destination: Path) -> list[dict]:
    """Run three native finite batches; return only root-authored saved-state metrics."""
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator

    native = select_native_dimension(1)
    bootstrap = native.mpi_world()

    def authenticate():
        package = Path(pops.__file__).resolve()
        if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("M26 native reception requires installed PoPS and no PYTHONPATH")
        pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
        return Path(destination)

    destination = _collective_call(bootstrap, "installed identity", authenticate)
    route = _collective_call(bootstrap, "MPI route", lambda: native_mpi_communicator(native))
    if len(set(bootstrap.allgather_bytes(route.encode()))) != 1:
        raise RuntimeError("M26 MPI route differs across ranks")
    compile_once = None
    if route == "MPI_COMM_WORLD":
        def helper():
            repository = str(Path(__file__).resolve().parents[3])
            sys.path.insert(0, repository)
            try:
                from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
            finally:
                sys.path.remove(repository)
            return compile_resolved_plan_once

        compile_once = _collective_call(bootstrap, "MPI compiler helper", helper)
    records = []
    for permuted in (False, True):
        case, layout, subjects, order = _collective_call(
            bootstrap, "finite authoring", lambda permuted=permuted:
            build_case(permuted=permuted))
        resolved = _collective_call(
            bootstrap, "finite resolve", lambda case=case, layout=layout:
            pops.resolve(pops.validate(case), layout=layout))
        if compile_once is not None:
            artifact = compile_once(
                bootstrap, resolved, route="M26 finite symmetric interaction",
                compile_artifact=pops.compile)
        else:
            artifact = pops.compile(resolved)
        def require_dim1(artifact=artifact):
            if artifact.resolved_dimension != 1:
                raise RuntimeError("M26 finite batch requires a genuine Dim1 artifact")
            return True

        _collective_call(bootstrap, "Dim1 artifact", require_dim1)
        execution = _collective_call(
            bootstrap, "execution context", lambda artifact=artifact:
            pops.ExecutionContext.mpi_world(artifact))
        world = _collective_call(
            bootstrap, "execution communicator", lambda execution=execution:
            execution.communicator.handle)
        mismatch = world.rank != bootstrap.rank or world.size != bootstrap.size
        if any(row == b"different" for row in bootstrap.allgather_bytes(
                b"different" if mismatch else b"same")):
            raise RuntimeError("M26 execution communicator differs from preflight")
        for variation in ((False, True) if not permuted else (False,)):
            initial = _collective_call(
                world, "finite input", lambda subjects=subjects, order=order,
                variation=variation: _initial_values(subjects, order, variation=variation))
            runtime = _collective_call(
                world, "bind finite inputs", lambda artifact=artifact, initial=initial,
                execution=execution: pops.bind(
                    artifact, initial_values=initial,
                    resources={"execution_context": execution}))
            report = _collective_call(
                world, "finite native step", lambda runtime=runtime:
                pops.run(runtime, t_end=1., max_steps=1, console=False))
            raw = {name: _collective_call(
                world, name + " gather", lambda name=name, runtime=runtime:
                runtime.state_global(name)) for name in subjects}
            status = _collective_call(
                world, "accepted status", lambda runtime=runtime, report=report:
                json.dumps((runtime.time(), runtime.macro_step(),
                            report.accepted_steps, report.rejected_steps)).encode())
            if len(set(world.allgather_bytes(status))) != 1:
                raise RuntimeError("M26 native status differs across ranks")
            failure = b""
            if world.rank == 0:
                try:
                    if report.accepted_steps != 1 or report.rejected_steps != 0:
                        raise AssertionError("M26 finite batch must accept exactly one step")
                    destination.mkdir(parents=True, exist_ok=True)
                    label = ("permuted" if permuted else "canonical") \
                        + ("_variation" if variation else "_base")
                    path = destination / ("state_" + label + ".npz")
                    inverse = np.argsort(order)
                    np.savez_compressed(path,
                        **{name: np.asarray(value, dtype=float).reshape(-1)[inverse]
                           if name != "measured_pairing" else
                           np.asarray(value, dtype=float).reshape(-1)
                           for name, value in raw.items()}, time=runtime.time())
                    values = _assess_saved(path, variation=variation)
                    records.append({
                        **values, "permuted": permuted, "variation": variation,
                        "artifact_identity": artifact.artifact_identity.token,
                        "artifact_abi_key": artifact.abi_key,
                        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                        "mpi_ranks": world.size, "execution_context": execution.to_data(),
                        "run_report": report.to_data(), "saved_state": path.name,
                        "saved_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
                except Exception as exception:
                    failure = (type(exception).__name__ + ": " + str(exception)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
    payload = b""
    failure = b""
    if world.rank == 0:
        try:
            payload = (receipt_json({
                "schema_version": 1, "case": "M26_finite_symmetric_interaction",
                "status": "passed", "criteria": CRITERIA, "records": records,
                "scope": "N12 measured finite DOFs only, not an aggregation-diffusion PDE",
                "mpi_ranks": world.size,
                "threads_requested": int(os.environ.get("POPS_THREADS", "1")),
                "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "oracle_sha256": hashlib.sha256(
                    Path(__file__).with_name("api040_m26_finite_oracle.py").read_bytes()
                ).hexdigest(),
            }) + "\n").encode()
            (destination / "receipt.json").write_bytes(payload)
        except Exception as exception:
            failure = (type(exception).__name__ + ": " + str(exception)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
    return json.loads(world.broadcast_bytes(payload, root=0))["records"]


if __name__ == "__main__":
    run_and_archive(Path(os.environ["POPS_API040_OUTPUT"]))
