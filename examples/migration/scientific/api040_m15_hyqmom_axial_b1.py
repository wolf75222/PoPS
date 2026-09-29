"""M15 bounded variant: five one-velocity moments with Appendix B.1 S50.

This is the order-four axial marginal of the available fifteen-moment B.1
closure. It does not claim the unspecified Fox--Laurent recurrence at other
orders. The physical flux is an ordinary Python expression body passed to the
generic Model/FiniteVolume/Program construction; no Gaussian fifth moment is
substituted for the nonlinear B.1 relation.
"""
from __future__ import annotations

import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import hyqmom_b1_axial_flux
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import AdaptiveCFL, FixedDt


def build_case(cells: int = 16, *, order=(0, 1, 2, 3, 4), dt=None):
    """Author a periodic order-four axial B.1 test with exact Jacobian speeds."""
    if type(cells) is not int or cells < 4:
        raise ValueError("M15 axial cells must be an integer >= 4")
    if tuple(sorted(order)) != (0, 1, 2, 3, 4):
        raise ValueError("M15 component order must permute 0..4 exactly")
    frame = CartesianDomain("axial_b1_interval", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("hyqmom_b1_axial_order4", frame=frame)
    state = model.state("M", components=tuple("M%d" % k for k in order))
    canonical = tuple(state["M%d" % k] for k in range(5))
    # The B.1 formula divides by density and variance. Its admissible truncated
    # moment domain is not implied by finite fluxes or real characteristic roots.
    rho, m1, m2, m3, m4 = canonical
    hankel2 = rho * m2 - m1 * m1
    hankel3 = (rho * (m2 * m4 - m3 * m3)
               - m1 * (m1 * m4 - m2 * m3) + m2 * (m1 * m3 - m2 * m2))
    model.primitive_state(*state, conservative=tuple(state))
    model.recovery_admissibility(M0=rho > 0, M2=hankel2 > 0, M4=hankel3 >= 0)
    physical_flux = hyqmom_b1_axial_flux(canonical)
    flux = model.flux(
        "axial_b1_transport", frame=frame, state=state,
        components={frame.axes[0]: tuple(physical_flux[k] for k in order)},
    )
    model.wave_speeds_from_jacobian()
    rate = model.rate("moment_balance", equation=ddt(state) == -div(flux))
    case = pops.Case("M15_axial_B1_order4")
    block = case.block("moments", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.HLL(),
    ))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.1) if dt is None else FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes),
    ))
    return case, layout, state


__all__ = ["build_case"]


def is_native_admission_refusal(message):
    """Only the actual native recovery/face admission seam qualifies a negative."""
    import re
    return ("variable recovery rejected the candidate" in message or
            re.search(r"prepared ND hyperbolic face evaluation refused publication status=1(?:\D|$)",
                      message) is not None)


if __name__ == "__main__":
    # The default authoring API above retains AdaptiveCFL. This explicitly named
    # scientific variant uses the fixed FE calendar declared before reception.
    import hashlib
    import os
    from pathlib import Path
    import sys
    import numpy as np
    from api040_m15_axial_oracle import (
        CRITERIA, ORDERS, RESOLUTIONS, T_END, domain_values, initial_averages,
        invalid_states, require_admissible, trajectory,
    )
    from api040_receipts import receipt_json

    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M15 reception requires installed PoPS with PYTHONPATH unset")
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(1)
    native_file = Path(native.__file__).resolve()
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m15_axial"))
    # All NumPy references and domain/Courant checks precede any native compile.
    references = {n: trajectory(n) for n in RESOLUTIONS}
    records, rejections, canonical_results = [], [], {}
    for order in ORDERS:
        for n in RESOLUTIONS:
            dt, steps = 1/(100*n), 2*n
            case, layout, state = build_case(n, order=order, dt=dt)
            artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
            if artifact.resolved_dimension != 1:
                raise RuntimeError("M15 requires a genuine one-dimensional native artifact")
            execution = pops.ExecutionContext.mpi_world(artifact)
            world = execution.communicator.handle
            initial = np.ascontiguousarray(initial_averages(n)[list(order)])
            simulation = pops.bind(artifact, initial_state={"moments": initial},
                                   resources={"execution_context": execution})
            before = np.asarray(simulation.state_global("moments"))
            failure = b""
            if world.rank == 0:
                try:
                    if before.size != initial.size:
                        raise AssertionError("native initial gather has wrong size")
                    initial_error = float(np.max(np.abs(before.reshape(initial.shape)-initial)))
                    if not np.isfinite(initial_error) or initial_error > CRITERIA["initial_max_error"]:
                        raise AssertionError("initial state is not the prescribed cell averages")
                except Exception as error:
                    failure = (type(error).__name__+": "+str(error)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
            report = pops.run(simulation, t_end=T_END, max_steps=steps, console=False)
            gathered = np.asarray(simulation.state_global("moments"))
            failure = b""
            if world.rank == 0:
                try:
                    destination.mkdir(parents=True, exist_ok=True)
                    path = destination / ("state_%d_%s.npz" % (n, "".join(map(str, order))))
                    np.savez_compressed(path, initial=before.reshape(initial.shape),
                                        final=gathered.reshape(initial.shape),
                                        oracle=references[n][0][list(order)], order=order,
                                        time=simulation.time(), dt=dt, cells=n)
                    with np.load(path) as saved:
                        actual = saved["final"][np.argsort(saved["order"])]
                        require_admissible(actual)
                        error = float(np.max(np.abs(saved["final"]-saved["oracle"])))
                        mass = float(np.max(np.abs(np.mean(saved["final"]-saved["initial"], axis=1))))
                        time_error = abs(float(saved["time"])-T_END)
                        permutation_error = (float(np.max(np.abs(actual-canonical_results[n])))
                                             if n in canonical_results else 0.)
                    if (error > CRITERIA["state_max_error"] or
                            mass > CRITERIA["moment_integral_error"] or
                            time_error > CRITERIA["time_error"] or
                            permutation_error > CRITERIA["permutation_max_error"] or
                            report.accepted_steps != steps or report.rejected_steps != 0 or
                            simulation.macro_step() != steps):
                        raise AssertionError("M15 saved-state/step/permutation criterion failed")
                    canonical_results.setdefault(n, actual.copy())
                    records.append({"cells": [n], "order": order, "time": simulation.time(),
                                    "dt": dt, "accepted_steps": report.accepted_steps,
                                    "initial_max_error": initial_error, "state_max_error": error,
                                    "moment_integral_error": mass, "time_error": time_error,
                                    "permutation_max_error": permutation_error,
                                    "domain_minima": [float(x.min()) for x in domain_values(actual)],
                                    "oracle_max_face_courant": references[n][1],
                                    "mpi_ranks": world.size, "execution_context": execution.to_data(),
                                    "run_report": report.to_data(), "saved_state": path.name,
                                    "saved_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                except Exception as error:
                    failure = (type(error).__name__+": "+str(error)).encode()
                    (destination / "receipt.json").write_text(receipt_json({
                        "case": "M15_axial_B1_order4_fixed_fe", "status": "failed",
                        "reason": failure.decode(), "criteria": CRITERIA, "records": records})+"\n")
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())

            # Reuse the exact compiled artifact. These inputs must be refused,
            # never repaired into a realizable moment vector. Every rank participates.
            if n == RESOLUTIONS[0]:
                for label, canonical_bad in invalid_states(n).items():
                    bad = np.ascontiguousarray(canonical_bad[list(order)])
                    rejected = None
                    try:
                        rejected = pops.bind(artifact, initial_state={"moments": bad},
                                             resources={"execution_context": execution})
                        pops.run(rejected, t_end=dt, max_steps=1, console=False)
                    except Exception as error:
                        reason = type(error).__name__+": "+str(error)
                    else:
                        reason = ""
                    reasons = world.allgather_bytes(reason.encode())
                    bound_flags = world.allgather_bytes(b"bound" if rejected is not None else b"unbound")
                    if len(set(bound_flags)) != 1:
                        raise AssertionError("inadmissible bind did not converge between ranks")
                    after = (np.asarray(rejected.state_global("moments"))
                             if rejected is not None else None)
                    failure = b""
                    if world.rank == 0:
                        try:
                            if not all(is_native_admission_refusal(item.decode()) for item in reasons):
                                raise AssertionError("missing native moment-admission diagnostic: %r" % (reasons,))
                            if rejected is not None:
                                np.testing.assert_array_equal(after.reshape(bad.shape), bad)
                                if rejected.time() != 0 or rejected.macro_step() != 0:
                                    raise AssertionError("inadmissible input advanced native clock")
                            rejections.append({"case": label, "order": order,
                                               "rank_diagnostics": [x.decode() for x in reasons]})
                        except Exception as error:
                            failure = (type(error).__name__+": "+str(error)).encode()
                    failure = world.broadcast_bytes(failure, root=0)
                    if failure:
                        raise RuntimeError(failure.decode())
    failure = b""
    if world.rank == 0:
        receipt = {"schema_version": 1, "case": "M15_axial_B1_order4_fixed_fe", "status": "passed",
                   "scope": "genuine Dim=1 periodic five-moment B.1; not Fox-Laurent multiorder",
                   "resolutions": RESOLUTIONS, "t_end": T_END, "orders": ORDERS,
                   "spatial_method": "FirstOrder/HLL exact signed Jacobian eigenvalue bounds",
                   "temporal_method": "ForwardEuler FixedDt=1/(100N)", "criteria": CRITERIA,
                   "records": records, "inadmissible_initial_rejections": rejections,
                   "package_file": str(package), "package_version": pops.__version__,
                   "native_file": str(native_file),
                   "native_sha256": hashlib.sha256(native_file.read_bytes()).hexdigest(),
                   "threads_requested": os.environ.get("POPS_THREADS", "1")}
        try:
            (destination / "receipt.json").write_text(receipt_json(receipt)+"\n")
            print(receipt_json(receipt))
        except Exception as error:
            failure = (type(error).__name__+": "+str(error)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
