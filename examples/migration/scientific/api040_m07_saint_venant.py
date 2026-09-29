"""M07 Saint-Venant with a public coordinated hydrostatic face construction.

The physical topographic product is retained explicitly. The old face-only
counterexample is preserved separately; the NumPy reference remains independent.
"""
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.math import ddt, div, maximum, minimum, sqrt, where
from pops.numerics import PathConservativeFiniteVolume, SymbolicPath
from pops.numerics import reconstruction, riemann, variables
from pops.numerics import CoordinatedFace, CoordinatedFiniteVolume, FaceBalance


def declarations(order=("h", "q", "z"), *, coordinated=False):
    if len(order) != 3 or set(order) != {"h", "q", "z"}:
        raise ValueError("order must permute h, q, z")
    frame = CartesianDomain("lake", (-1.,), (1.,)).frame(Cartesian1D())
    axis = frame.axes[0]
    model = pops.Model("saint_venant_topography", frame=frame)
    state = model.state("U", components=order)
    h, q, z = (state[name] for name in ("h", "q", "z"))
    g = 1.  # H=1, L=1, velocity unit sqrt(g_physical*H), hence dimensionless g=1.
    physical = {"h": q, "q": q*q/h + .5*g*h*h, "z": 0.}
    flux = model.flux("shallow_water", state=state, frame=frame,
                      components={axis: tuple(physical[name] for name in order)})
    matrix = tuple(tuple(g*h if row == "q" and column == "z" else 0.
                         for column in order) for row in order)
    product = model.nonconservative_product("topography", state=state,
        matrices={axis: matrix}, conservative_components=("h", "z"))
    rate = model.rate("full_balance", equation=ddt(state) == -div(flux) - product)
    model.primitive_state(*state, conservative=tuple(state))
    model.recovery_admissibility(h=h > 0.)
    ih, iq, iz = (order.index(name) for name in ("h", "q", "z"))

    def speed(left, right, _axis=0):
        return maximum(abs(left[iq]/left[ih]) + sqrt(g*left[ih]),
                       abs(right[iq]/right[ih]) + sqrt(g*right[ih]))

    # The straight path represents the physical nonconservative product exactly;
    # it is NOT claimed to be the requested hydrostatic numerical realization.
    path = SymbolicPath(product, frame=frame, quadrature=((.5, 1.),), speed=speed)

    def hydrostatic_face(left, right, _fl, _fr, _speed):
        zstar = maximum(left[iz], right[iz])
        hl = maximum(0., left[ih] + left[iz] - zstar)
        hr = maximum(0., right[ih] + right[iz] - zstar)
        ul, ur = left[iq]/left[ih], right[iq]/right[ih]
        ql, qr = hl*ul, hr*ur
        sl = minimum(0., minimum(ul-sqrt(g*hl), ur-sqrt(g*hr)))
        sr = maximum(0., maximum(ul+sqrt(g*hl), ur+sqrt(g*hr)))
        fl, fr = {"h": ql, "q": ql*ul+.5*g*hl*hl, "z": 0.}, {
            "h": qr, "q": qr*ur+.5*g*hr*hr, "z": 0.}
        jump = {"h": hr-hl, "q": qr-ql, "z": 0.}
        shared = {name: where(sr > sl,
            lambda name=name: (sr*fl[name]-sl*fr[name]+sl*sr*jump[name])/(sr-sl),
            lambda: 0.) for name in order}
        lower = dict(shared)
        upper = dict(shared)
        lower["q"] += .5*g*(left[ih]*left[ih]-hl*hl)
        upper["q"] += .5*g*(right[ih]*right[ih]-hr*hr)
        if coordinated:
            left_source = tuple(-.5*g*(left[ih]*left[ih]-hl*hl) if name == "q" else 0.
                                for name in order)
            right_source = tuple(.5*g*(right[ih]*right[ih]-hr*hr) if name == "q" else 0.
                                 for name in order)
            return FaceBalance(tuple(shared[name] for name in order),
                               left_source, right_source, speed(left, right))
        return tuple(lower[name] for name in order), tuple(upper[name] for name in order)

    return model, state, flux, product, rate, path, hydrostatic_face


def unavailable_hydrostatic_method(order=("h", "q", "z")):
    """Minimal public reproduction; does not fabricate a native descriptor."""
    _model, state, flux, _product, _rate, path, face = declarations(order)
    # Required two-sided output is rejected by the actual public face authoring.
    faces = riemann.User(body=face, state=state,
                         stability=lambda left, right, fl, fr, speed: speed)
    return PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=faces)


def build_case(cells=40, *, order=("h", "q", "z")):
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid
    from pops.lib.time import ForwardEuler
    from pops.time import FixedDt
    from pops.numerics import DiscretizationPlan
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow
    if not isinstance(cells, int) or isinstance(cells, bool) or cells < 1:
        raise ValueError("cells must be a positive integer")
    model, state, flux, product, rate, path, body = declarations(order, coordinated=True)
    face = CoordinatedFace(flux=flux, product=product, frame=path.frame,
        body=lambda left, right, axis: body(left, right, None, None, None))
    method = CoordinatedFiniteVolume(face=face)
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("M07_hydrostatic_lake")
    block = case.block("lake", model)
    boundary_values = equilibrium_boundary_values(cells, order=order)
    plan.boundaries.add(TransportBoundarySet({
        side: Inflow(state=block[state], value=value)
        for side, value in zip(path.frame.boundaries.all, boundary_values, strict=True)
    }))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1./(5*cells)))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(cells,)))
    return case, layout, state


def equilibrium_boundary_values(cells, *, order=("h", "q", "z")):
    """Fixed-face data whose reflected ghosts equal the frozen lake averages.

    Native Inflow uses ghost=2*value-interior. This realizes the same equilibrium
    exactly; it is a Dirichlet closure, not a claim about perturbed ghost dynamics.
    """
    from math import erf, pi, sqrt
    dx = 2./cells
    def mean(a, b):
        return .2*sqrt(pi)/(2*sqrt(50.)*(b-a))*(erf(sqrt(50.)*b)-erf(sqrt(50.)*a))
    result = []
    for outer, inner in (((-1.-dx, -1.), (-1., -1.+dx)),
                         ((1., 1.+dx), (1.-dx, 1.))):
        zface = .5*(mean(*outer)+mean(*inner))
        values = {"h": 1.-zface, "q": 0., "z": zface}
        result.append(tuple(values[name] for name in order))
    return tuple(result)


if __name__ == "__main__":
    import hashlib
    import os
    from pathlib import Path
    import sys
    import numpy as np
    from api040_m07_hydrostatic_oracle import (
        RESOLUTIONS, ORDERS, T_END, EQUILIBRIUM_TOLERANCE, initial_averages, trajectory,
    )
    from api040_receipts import receipt_json
    package = Path(pops.__file__).resolve()
    if "PYTHONPATH" in os.environ or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("M07 reception requires installed PoPS with PYTHONPATH unset")
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(1)
    native_file = Path(native.__file__).resolve()
    pops.set_threads(int(os.environ.get("POPS_THREADS", "1")))
    destination = Path(os.environ.get("POPS_API040_OUTPUT", "outputs/api040_m07_hydrostatic"))
    references = {n: trajectory(n) for n in RESOLUTIONS}
    records, canonical = [], {}
    for order in ORDERS:
        permutation = [ORDERS[0].index(name) for name in order]
        for n in RESOLUTIONS:
            case, layout, state = build_case(n, order=order)
            artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
            if artifact.resolved_dimension != 1:
                raise RuntimeError("M07 requires genuine native Dim=1")
            execution = pops.ExecutionContext.mpi_world(artifact)
            world = execution.communicator.handle
            initial = np.ascontiguousarray(initial_averages(n)[permutation])
            simulation = pops.bind(artifact, initial_state={"lake": initial},
                                   resources={"execution_context": execution})
            before = np.asarray(simulation.state_global("lake"))
            failure = b""
            if world.rank == 0:
                try:
                    np.testing.assert_array_equal(before.reshape(initial.shape), initial)
                except Exception as error:
                    failure = (type(error).__name__+": "+str(error)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
            report = pops.run(simulation, t_end=T_END, max_steps=5*n, console=False)
            gathered = np.asarray(simulation.state_global("lake"))
            failure = b""
            if world.rank == 0:
                try:
                    destination.mkdir(parents=True, exist_ok=True)
                    path = destination / ("state_%d_%s.npz" % (n, "".join(order)))
                    np.savez_compressed(path, initial=initial, final=gathered.reshape(initial.shape),
                        oracle=references[n][0][permutation], order=order,
                        time=simulation.time(), dt=1./(5*n), cells=n)
                    with np.load(path) as saved:
                        actual = saved["final"][np.argsort(permutation)]
                        error = max(float(np.max(np.abs(actual[0]+actual[2]-1.))),
                            float(np.max(np.abs(actual[1]))),
                            float(np.max(np.abs(actual[2]-initial_averages(n)[2]))))
                        reference_error = float(np.max(np.abs(saved["final"]-saved["oracle"])))
                        permutation_error = float(np.max(np.abs(actual-canonical[n]))) if n in canonical else 0.
                    if (not np.isfinite(actual).all() or min(actual[0]) <= 0 or
                            max(error, reference_error, permutation_error) > EQUILIBRIUM_TOLERANCE or
                            abs(simulation.time()-T_END) > 2.e-14 or simulation.macro_step() != 5*n or
                            report.accepted_steps != 5*n or report.rejected_steps != 0):
                        raise AssertionError("M07 frozen equilibrium/time/permutation criterion failed")
                    canonical.setdefault(n, actual.copy())
                    records.append({"cells": [n], "order": order, "equilibrium_error": error,
                        "oracle_error": reference_error, "permutation_error": permutation_error,
                        "boundary": "fixed_face_mirror_exact_for_initial_equilibrium",
                        "oracle_max_face_courant": references[n][1]["max_courant"],
                        "execution_context": execution.to_data(), "mpi_ranks": world.size,
                        "run_report": report.to_data(), "saved_state": path.name,
                        "saved_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                except Exception as error:
                    failure = (type(error).__name__+": "+str(error)).encode()
            failure = world.broadcast_bytes(failure, root=0)
            if failure:
                raise RuntimeError(failure.decode())
    failure = b""
    if world.rank == 0:
        try:
            receipt = {"case": "M07", "variant": "fully_wet_hydrostatic_lake",
                "dimension": 1, "package": str(package), "native_file": str(native_file),
                "native_sha256": hashlib.sha256(native_file.read_bytes()).hexdigest(),
                "criterion": EQUILIBRIUM_TOLERANCE, "time": T_END, "records": records}
            (destination/"receipt.json").write_text(receipt_json(receipt))
        except Exception as error:
            failure = (type(error).__name__+": "+str(error)).encode()
    failure = world.broadcast_bytes(failure, root=0)
    if failure:
        raise RuntimeError(failure.decode())
