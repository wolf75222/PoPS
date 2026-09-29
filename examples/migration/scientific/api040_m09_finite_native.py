"""The exact finite W06/M09 algebra, batched once in native PoPS state storage.

The 8 and 4 component labels denote finite DOFs, not cells of a grid.
"""
import pops
from pops.linalg import FiniteSupport, FiniteLinearMap
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops.time import LocalResidual, FixedDt, FailRun
from pops.solvers.nonlinear import LocalNewton

DT = .06
THRESHOLD = 1.e-11


def build_case(data, *, condensed=False, permuted=False):
    # data carries coefficients and initial captures only, never a computed solve.
    v_order = tuple(reversed(range(8))) if permuted else tuple(range(8))
    p_order = (2, 0, 3, 1) if permuted else tuple(range(4))
    vs = FiniteSupport("velocity", tuple("v%d" % i for i in v_order))
    ps = FiniteSupport("potential", tuple("p%d" % i for i in p_order))
    def matrix(source, target, values, rows, columns):
        return FiniteLinearMap(source, target, tuple(tuple(float(values[i,j]) for j in columns) for i in rows))
    A = matrix(vs, vs, data.a, v_order, v_order)
    B = matrix(ps, vs, data.b, v_order, p_order)
    C = matrix(vs, ps, data.c, p_order, v_order)
    K = matrix(ps, ps, data.k, p_order, p_order)
    frame = Rectangle("finite_batch", (0., 0.), (1., 1.)).frame(Cartesian2D())
    vm, pm = pops.Model("velocity_dofs", frame=frame), pops.Model("potential_dofs", frame=frame)
    vu, pu = vm.state("v", components=vs.dofs), pm.state("phi", components=ps.dofs)
    case = pops.Case("finite_m09")
    if permuted:
        pb, vb = case.block("potential", pm), case.block("velocity", vm)
    else:
        vb, pb = case.block("velocity", vm), case.block("potential", pm)
    P = pops.Program("finite_condensed" if condensed else "finite_monolithic")
    v, p = P.state(vb[vu]), P.state(pb[pu])
    sv = P.value("v_seed", 0*v.n, at=v.next.point)
    sp = P.value("p_seed", 0*p.n, at=p.next.point)

    def original(velocity, potential, old_v, old_p):
        return A.apply(velocity)+B.apply(potential)-old_v, C.apply(velocity)+K.apply(potential)-K.apply(old_p)

    def residual(program, z, *, old_v, old_p):
        del program
        potential = ps.bind(z["potential"])
        force = vs.bind(old_v)
        velocity = A.solve(force-B.apply(potential)) if condensed else vs.bind(z["velocity"])
        first, second = original(velocity, potential, force, ps.bind(old_p))
        return {"potential": tuple(second)} if condensed else {"velocity": tuple(first), "potential": tuple(second)}

    seeds = {"potential": sp} if condensed else {"velocity": sv, "potential": sp}
    result = P.solve(LocalResidual(residual, seeds, captures={"old_v":v.n, "old_p":p.n}),
                     solver=LocalNewton(tolerance=1.e-13)).consume(action=FailRun())
    potential = result[pb]
    if condensed:
        velocity = A.solve(vs.bind(v.n)-B.apply(ps.bind(potential))).materialize(P, "reconstructed_velocity", template=v.n, at=v.next.point)
    else:
        velocity = result[vb]
    r_v, r_p = original(vs.bind(velocity), ps.bind(potential), vs.bind(v.n), ps.bind(p.n))
    rv = r_v.materialize(P, "original_velocity_residual", template=velocity, at=velocity.point)
    rp = r_p.materialize(P, "original_potential_residual", template=potential, at=potential.point)
    good_v, good_p = P.norm_inf(rv) < THRESHOLD, P.norm_inf(rp) < THRESHOLD
    for name, target, value in (("velocity", v.next, velocity), ("potential", p.next, potential)):
        value = P.guard(name+"_original_velocity", value, good_v, action=FailRun())
        value = P.guard(name+"_original_potential", value, good_p, action=FailRun())
        P.commit(target, value)
    P.step_strategy(FixedDt(DT))
    case.program(P)
    subjects = (vb[vu], pb[pu])
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(1,1), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subjects, (v_order, p_order)


def main():
    import argparse
    import json
    from pathlib import Path
    import numpy as np
    import api040_m09_hoffart_oracle as oracle
    from pops._native_collectives import allgather_value, broadcast_value
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condensed", action="store_true")
    parser.add_argument("--permuted", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/m09-finite-native"))
    args = parser.parse_args()
    data = oracle.witness()
    case, layout, subjects, (vo, po) = build_case(data, condensed=args.condensed, permuted=args.permuted)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = pops.ExecutionContext.mpi_world(artifact)
    world = context.communicator.handle
    def collective(operation):
        value, failure = None, None
        try:
            value = operation()
        except Exception as exc:
            failure = (type(exc).__name__, str(exc), isinstance(exc, RuntimeError))
        failures = allgather_value(world, failure)
        if any(failures):
            raise RuntimeError(repr(failures))
        return value

    initial = collective(lambda: {
        subjects[0]: data.velocity_old[list(vo)].reshape(8, 1, 1),
        subjects[1]: data.potential_old[list(po)].reshape(4, 1, 1)})
    runtime = collective(lambda: pops.bind(artifact, initial_values=initial,
                         resources={"execution_context": context}))
    error = None
    try:
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
        assert runtime.time() == DT and runtime.macro_step() == 1
    except Exception as exc:
        error = repr(exc)
    errors = allgather_value(world, error)
    if any(errors):
        raise RuntimeError(repr(errors))
    raw_velocity = collective(lambda: runtime.state_global("velocity"))
    velocity = collective(lambda: np.asarray(raw_velocity))
    raw_potential = collective(lambda: runtime.state_global("potential"))
    potential = collective(lambda: np.asarray(raw_potential))
    if world.rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            state_path = args.output/"accepted_state.npz"
            np.savez_compressed(state_path, velocity=velocity.reshape(8)[np.argsort(vo)],
                potential=potential.reshape(4)[np.argsort(po)], velocity_old=data.velocity_old,
                potential_old=data.potential_old, time=runtime.time())
            with np.load(state_path) as saved:
                values = saved["velocity"], saved["potential"]
                residual = oracle.original_residual(data, *values)
                reference = oracle.monolithic(data)
                difference = max(float(np.max(np.abs(a-b))) for a,b in zip(values, reference, strict=True))
                endpoint = oracle.endpoint(data, values)
                energy_defect = abs(oracle.energy(data, *endpoint)-oracle.energy(data, data.velocity_old, data.potential_old))
                assert max(residual, difference, energy_defect) < THRESHOLD
            receipt = dict(scope="finite W06/M09 DOFs, no global mesh PDE", condensed=args.condensed,
                permuted=args.permuted, mpi_ranks=world.size, original_residual=residual,
                monolithic_difference=difference, cn_energy_defect=energy_defect,
                artifact_identity=artifact.artifact_identity.token, state_path=str(state_path))
            (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            print(json.dumps(receipt, indent=2))
        except Exception as exc:
            error = repr(exc)
    error = broadcast_value(world, error, root=0)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
