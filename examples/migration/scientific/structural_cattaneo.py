#!/usr/bin/env python3
"""Structural C11+T3 prototype: two-state Maxwell--Cattaneo in actual 2D.

No model-specific core opcode. Native qualification is pending central execution.
"""
from __future__ import annotations

from dataclasses import dataclass

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.params import Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import ExternalTimeGrid, FailRun, LocalResidual

CELLS = 16
FINAL_TIME = .04
TIME_STEPS = (.004, .002, .001)
PARAMETERS = ((.1, .025), (.12, .048))
CRITERIA = {"scheme_max_error": 2.e-10, "temperature_inventory": 2.e-12,
            "temporal_order_min": .8, "temporal_order_max": 1.2,
            "permutation_max_error": 2.e-10, "initial_max_error": 2.e-14,
            "rebind_min_difference": 1.e-3}


@dataclass(frozen=True)
class Authored:
    case: object
    layout: object
    states: tuple
    blocks: tuple
    subjects: tuple
    parameters: tuple
    order: tuple
    variant: str


def build_case(*, variant="canonical", cells=CELLS):
    if variant not in ("canonical", "permuted"):
        raise ValueError("variant must be canonical or permuted")
    swapped = variant == "permuted"
    frame = Rectangle("thermal_box" if not swapped else "periodic_plate",
                       (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("cattaneo" if not swapped else "finite_speed_heat", frame=frame)
    temperature = model.species("thermal" if not swapped else "caloric", state=("T" if not swapped else "theta",))
    flux_labels = ("qx", "qy") if not swapped else ("current_y", "current_x")
    heat_flux = model.species("heat" if not swapped else "current", state=flux_labels)
    order = (0, 1) if not swapped else (1, 0)
    qx, qy = (heat_flux[i] for i in order)
    tau_decl = RuntimeParam("tau" if not swapped else "relaxation_time", default=.1, domain=Positive())
    kappa_decl = RuntimeParam("kappa" if not swapped else "conductivity", default=.025, domain=Positive())
    tau_handle, kappa_handle = model.param(tau_decl), model.param(kappa_decl)
    tau, kappa = model.value(tau_handle), model.value(kappa_handle)
    speed = sqrt(kappa / tau)
    flux_t = model.flux("temperature_transport", state=temperature, frame=frame,
                       components={frame.x: (qx,), frame.y: (qy,)},
                       waves={axis: (speed,) for axis in frame.axes})
    canonical_q_flux = {frame.x: (kappa / tau * temperature[0], 0.),
                        frame.y: (0., kappa / tau * temperature[0])}
    flux_q = model.flux("heat_transport", state=heat_flux, frame=frame,
                       components={axis: tuple(row[i] for i in order)
                                   for axis, row in canonical_q_flux.items()})
    rates = (model.rate("temperature_balance", equation=ddt(temperature) == -div(flux_t)),
             model.rate("heat_principal", equation=ddt(heat_flux) == -div(flux_q)))
    model.source("heat_relaxation", on=heat_flux, value=tuple(-q / tau for q in heat_flux))
    source = model.module.operator_handle("heat_relaxation")
    states = (temperature, heat_flux)
    case = pops.Case("structural_cattaneo_" + variant)
    block_names = ("temperature", "flux") if not swapped else ("caloric_store", "current_store")
    insertion = (0, 1) if not swapped else (1, 0)
    table = {i: case.block(block_names[i], model, states=(states[i],)) for i in insertion}
    blocks = tuple(table[i] for i in range(2))
    for i in insertion:
        state, block = states[i], blocks[i]
        plan = DiscretizationPlan()
        plan.rates.add(rates[i], FiniteVolume(
            flux=(flux_t, flux_q)[i], variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov(),
            sampling=(states[1-i],)))
        case.numerics(plan, block=block)
        case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                           projection=ConservativeCellAverage()))
    program = pops.Program("joint_principal_then_implicit_relaxation")
    quantities = tuple(program.state(block[state]) for block, state in zip(blocks, states, strict=True))
    bindings = dict(zip(states, (q.n for q in quantities), strict=True))
    rates_at_old = tuple(rate(q.n, bindings=bindings) for rate, q in zip(rates, quantities, strict=True))
    stars = tuple(program.value("explicit_principal", q.n + program.dt * rate, at=q.next.point)
                  for q, rate in zip(quantities, rates_at_old, strict=True))
    # Deliberately distinct predictor and fixed data: changing a Newton seed must
    # not replace the explicit principal state in the implicit equation.
    seed = program.value("non_solution_seed", .9 * stars[1], at=quantities[1].next.point)

    def residual(p, unknown, explicit):
        relaxation = p.source(source, unknown)
        return p.value("original_relaxation_equation", unknown - explicit - p.dt * relaxation,
                       at=unknown.point)

    solved = program.solve(LocalResidual(residual, seed, captures={"explicit": stars[1]}),
                           solver=LocalNewton(tolerance=1.e-12, max_iterations=8)).consume(action=FailRun())
    program.commit(quantities[0].next, stars[0])
    program.commit(quantities[1].next, solved)
    program.step_strategy(ExternalTimeGrid("steps"))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)))
    subjects = tuple(case.resolve(block[state]) for block, state in zip(blocks, states, strict=True))
    parameters = tuple((block[tau_handle], block[kappa_handle]) for block in blocks)
    return Authored(case, layout, states, blocks, subjects, parameters, order, variant)


def source_receipt():
    """Actual public validation and lowering, without compiling a native module."""
    import hashlib
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    results = {}
    for variant in ("canonical", "permuted"):
        authored = build_case(variant=variant)
        plan = pops.resolve(pops.validate(authored.case), layout=authored.layout)
        graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
        source = emit_cpp_program(plan.time, model=graph)
        groups = source.count("PreparedPrincipalFlux<pops::kNativeDimension,3>")
        solves = source.count("solve_prepared_local_nonlinear(")
        if groups != 1 or solves != 1:
            raise AssertionError("Cattaneo must emit one complete group and one local solve")
        results[variant] = {"validate_resolve": True, "principal_group_width": 3,
                            "principal_resources": groups, "native_local_solves": solves,
                            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    return {"status": "source_only", "native_status": "not_executed", "variants": results}


def _receipt_json(value):
    import json
    def encode(item):
        if isinstance(item, bytes):
            return {"encoding": "hex", "bytes": item.hex()}
        raise TypeError("unsupported Cattaneo receipt value: " + type(item).__name__)
    return json.dumps(value, default=encode, indent=2, allow_nan=False)


def _root_result(world, function):
    result, failure = None, b""
    if world.rank == 0:
        try:
            result = function()
        except Exception as error:
            failure = (type(error).__name__ + ": " + str(error)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
    return result


def main():
    import argparse
    import hashlib
    import os
    from pathlib import Path
    import sys
    import numpy as np
    import structural_cattaneo_oracle as oracle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/structural_cattaneo"))
    args = parser.parse_args()
    if args.authoring_only:
        print(_receipt_json(source_receipt()))
        return
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("native reception requires installed PoPS and PYTHONPATH unset")
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    initial = oracle.exact(CELLS, 0., *PARAMETERS[0])
    records, saved_paths = [], {}
    permutation_errors, convergence_records = [], []
    for variant in ("canonical", "permuted"):
        authored = build_case(variant=variant)
        artifact = pops.compile(pops.resolve(pops.validate(authored.case), layout=authored.layout))
        if artifact.resolved_dimension != 2:
            raise RuntimeError("Cattaneo prototype requires native Dim2")
        context = pops.ExecutionContext.mpi_world(artifact)
        world = context.communicator.handle

        def bind(tau, kappa):
            parameters = {handle: value for pair in authored.parameters
                          for handle, value in zip(pair, (tau, kappa), strict=True)}
            values = {authored.subjects[0]: initial[:1].copy(),
                      authored.subjects[1]: initial[1:][list(authored.order)].copy()}
            return pops.bind(artifact, params=parameters, initial_values=values,
                              resources={"execution_context": context})

        # The same compiled artifact is rebound for all parameters and step grids.
        # Invalid declarations must fail at public bind, on every participating rank.
        refusals = []
        for invalid_tau in (0., -.1):
            refused = False
            try:
                bind(invalid_tau, PARAMETERS[0][1])
            except (ValueError, RuntimeError):
                refused = True
            for rank in range(world.size):
                answer = world.broadcast_bytes(b"refused" if refused else b"accepted", root=rank)
                if answer != b"refused":
                    raise RuntimeError("nonpositive relaxation time was accepted on a rank")
            refusals.append(invalid_tau)

        def gather(simulation):
            rows = [np.asarray(simulation.state_global(block.local_id)) for block in authored.blocks]
            if world.rank != 0:
                return None, None, None
            t = rows[0].reshape(1, CELLS, CELLS)
            q = rows[1].reshape(2, CELLS, CELLS)
            return np.concatenate((t, q[list(authored.order)])), t, q

        for parameter_index, (tau, kappa) in enumerate(PARAMETERS):
            temporal_errors = []
            for dt in TIME_STEPS:
                steps = round(FINAL_TIME / dt)
                grid = tuple(float(t) for t in np.linspace(0., FINAL_TIME, steps + 1))
                simulation = bind(tau, kappa)
                before, _, _ = gather(simulation)
                _root_result(world, lambda: np.testing.assert_allclose(
                    before, initial, rtol=0., atol=CRITERIA["initial_max_error"]))
                report = pops.run(simulation, t_end=FINAL_TIME, max_steps=steps,
                                   time_grid=grid, console=False)
                actual, stored_t, stored_q = gather(simulation)
                destination = args.output / variant / ("params%d_dt%.6g" % (parameter_index, dt))

                def measure():
                    if report.accepted_steps != steps or report.rejected_steps != 0:
                        raise AssertionError("unexpected accepted/rejected step count")
                    if abs(simulation.time() - FINAL_TIME) > 1.e-14:
                        raise AssertionError("incorrect final time")
                    destination.mkdir(parents=True, exist_ok=True)
                    path = destination / "state.npz"
                    np.savez_compressed(path, initial=before, final=actual,
                        temperature_storage=stored_t, flux_storage=stored_q,
                        component_order=np.asarray(authored.order), tau=tau, kappa=kappa,
                        dt=dt, steps=steps, time=simulation.time())
                    with np.load(path) as saved:
                        start, final = saved["initial"], saved["final"]
                        np.testing.assert_allclose(start, initial, rtol=0., atol=CRITERIA["initial_max_error"])
                        scheme = oracle.imex_euler(CELLS, dt, steps, tau, kappa)
                        semi = oracle.exact(CELLS, FINAL_TIME, tau, kappa, discrete_space=True)
                        continuum = oracle.exact(CELLS, FINAL_TIME, tau, kappa)
                        error = float(np.max(np.abs(final - scheme)))
                        temporal = float(np.max(np.abs(final - semi)))
                        spatial = float(np.max(np.abs(semi - continuum)))
                        inventory = float(abs(final[0].mean() - start[0].mean()))
                        assert error < CRITERIA["scheme_max_error"], error
                        assert inventory < CRITERIA["temperature_inventory"], inventory
                        key = (parameter_index, dt)
                        if parameter_index == 1:
                            first_path = args.output / variant / ("params0_dt%.6g" % dt) / "state.npz"
                            with np.load(first_path) as first:
                                rebound_difference = float(np.max(np.abs(final - first["final"])))
                            assert rebound_difference > CRITERIA["rebind_min_difference"]
                        if variant == "canonical":
                            saved_paths[key] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
                        else:
                            peer_path, peer_hash = saved_paths[key]
                            assert hashlib.sha256(peer_path.read_bytes()).hexdigest() == peer_hash
                            with np.load(peer_path) as peer:
                                permutation = float(np.max(np.abs(final - peer["final"])))
                            assert permutation < CRITERIA["permutation_max_error"], permutation
                            permutation_errors.append(permutation)
                    return {"variant": variant, "tau": tau, "kappa": kappa, "dt": dt,
                            "steps": steps, "scheme_max_error": error, "temporal_max_error": temporal,
                            "semidiscrete_to_continuum_error": spatial,
                            "temperature_inventory_defect": inventory, "state_file": str(path),
                            "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "run_report": report.to_data()}

                record = _root_result(world, measure)
                if world.rank == 0:
                    records.append(record)
                    temporal_errors.append(record["temporal_max_error"])
            def temporal_check():
                orders = np.log2(np.asarray(temporal_errors[:-1]) / temporal_errors[1:])
                assert np.all(orders > CRITERIA["temporal_order_min"]), orders
                assert np.all(orders < CRITERIA["temporal_order_max"]), orders
                return orders.tolist()
            orders = _root_result(world, temporal_check)
            if world.rank == 0:
                convergence_records.append({"variant": variant, "tau": tau, "kappa": kappa,
                                            "errors": temporal_errors, "orders": orders})

    def finish():
        from pops import _pops
        native = Path(_pops.__file__).resolve()
        receipt = {"case": "structural_Maxwell_Cattaneo", "status": "passed", "dimension": 2,
            "criteria": CRITERIA, "cells": [CELLS, CELLS], "mode": oracle.MODE,
            "records": records, "permutation_errors": permutation_errors,
            "temporal_convergence": convergence_records,
            "invalid_tau_refused": refusals, "package_file": str(package),
            "package_version": pops.__version__, "native_file": str(native),
            "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest(),
            "abi_key": _pops.abi_key(), "execution_context": context.to_data(),
            "scope": "Uniform Dim2 periodic oblique Fourier; first-order FV and IMEX Euler; no AMR/GPU claim"}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "result.json").write_text(_receipt_json(receipt) + "\n")
        print(_receipt_json(receipt))
    _root_result(world, finish)


if __name__ == "__main__":
    main()
