"""W10 constrained friction through a native public Program, not a full M11 PDE."""

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual

# Exact matrix from the handoff W10 script; singular, rank two, kernel span{(1,1,1)}.
FRICTION_L = ((3., -2., -1.), (-2., 5., -3.), (-1., -3., 4.))
DT = .01  # One algebraic Program publication; not a physical mixture time discretization.


def build_case(*, permuted=False):
    order = (2, 0, 1) if permuted else (0, 1, 2)
    matrix = tuple(tuple(FRICTION_L[i][j] for j in order) for i in order)
    frame = Rectangle("w10_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    flux_model = pops.Model("collective_flux", frame=frame)
    flux_state = flux_model.state("J", components=tuple("species_%d" % i for i in order))
    multiplier_model = pops.Model("constraint_multiplier", frame=frame)
    multiplier_state = multiplier_model.state("lambda", components=("lambda",))
    case = pops.Case("w10_constrained_friction")
    if permuted:
        multiplier_block = case.block("multiplier", multiplier_model)
        flux_block = case.block("flux", flux_model)
    else:
        flux_block = case.block("flux", flux_model)
        multiplier_block = case.block("multiplier", multiplier_model)
    subjects = (flux_block[flux_state], multiplier_block[multiplier_state])
    program = pops.Program("joint_friction_solve")
    j, multiplier = (program.state(subject) for subject in subjects)
    seed_j = program.value("flux_seed", 0*j.n, at=j.next.point)
    seed_multiplier = program.value("multiplier_seed", 0*multiplier.n, at=multiplier.next.point)

    def residual(P, z, *, force):
        del P
        return {"flux": tuple(sum(matrix[i][k]*z["flux"][k] for k in range(3))
                              + z["multiplier"][0] - force[i] for i in range(3)),
                "multiplier": (sum(z["flux"][k] for k in range(3)),)}

    seeds = ({"multiplier": seed_multiplier, "flux": seed_j} if permuted
             else {"flux": seed_j, "multiplier": seed_multiplier})
    values = program.solve(LocalResidual(residual, seeds, captures={"force": j.n}),
                           solver=LocalNewton(tolerance=1.e-13)).consume(action=FailRun())
    candidate = values[flux_block]
    original = program.value("original_friction_residual",
        tuple(sum(matrix[i][k]*candidate[k] for k in range(3))-j.n[i] for i in range(3)),
        at=candidate.point)
    constraint = program.value("sum_flux_residual",
        tuple(sum(candidate[k] for k in range(3)) for _ in range(3)), at=candidate.point)
    # An incompatible force has an augmented solution with nonzero lambda. The
    # original physical equation remains the authority and forbids its projection.
    valid_original = program.norm_inf(original) < 1.e-12
    valid_constraint = program.norm_inf(constraint) < 1.e-12
    guarded_j = program.guard("original_friction", candidate, valid_original, action=FailRun())
    guarded_j = program.guard("zero_sum_flux", guarded_j, valid_constraint, action=FailRun())
    guarded_multiplier = program.guard("original_friction_multiplier", values[multiplier_block],
                                       valid_original, action=FailRun())
    guarded_multiplier = program.guard("zero_sum_multiplier", guarded_multiplier,
                                       valid_constraint, action=FailRun())
    program.commit_many({j.next: guarded_j, multiplier.next: guarded_multiplier})
    program.step_strategy(FixedDt(DT))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subjects, order


def main():
    import json
    import numpy as np
    from api040_m11_w10_oracle import force_samples, reference, residuals, THRESHOLD

    runs = []
    for permuted in (False, True):
        case, layout, subjects, order = build_case(permuted=permuted)
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        context = pops.ExecutionContext.mpi_world(artifact)
        world = context.communicator.handle
        for sample, force in enumerate(force_samples()):
            runtime = pops.bind(artifact, initial_values={subjects[0]: force[list(order)],
                                subjects[1]: np.zeros((1, 4, 4))},
                                resources={"execution_context": context})
            pops.run(runtime, t_end=DT, max_steps=1, console=False)
            raw_flux = np.asarray(runtime.state_global("flux"))
            raw_lambda = np.asarray(runtime.state_global("multiplier"))
            error = ""
            if world.rank == 0:
                try:
                    flux = raw_flux.reshape(3, 4, 4)[np.argsort(order)]
                    multiplier = raw_lambda.reshape(1, 4, 4)
                    metrics = residuals(force, flux, multiplier)
                    expected, _ = reference(force)
                    assert metrics["original_residual"] < THRESHOLD
                    assert metrics["constraint"] < THRESHOLD
                    assert np.max(np.abs(flux-expected)) < THRESHOLD
                    runs.append({"permuted": permuted, "sample": sample,
                                 "mpi_ranks": world.size, "time": runtime.time(),
                                 "artifact_identity": artifact.artifact_identity.token, **metrics})
                except Exception as exc:
                    error = repr(exc)
            from pops._native_collectives import broadcast_value
            error = broadcast_value(world, error, root=0)
            if error:
                raise RuntimeError(error)
    if world.rank == 0:
        print(json.dumps({"scope": "W10 finite constrained witness, not complete M11", "runs": runs},
                         indent=2))


if __name__ == "__main__":
    main()
