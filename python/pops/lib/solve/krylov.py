"""The reference conjugate gradient of ``pops.lib.solve`` (spec 0.3 SV9, 2.6.9 row ``cg``).

The body mirrors the native ``solve_cg`` and its wrapper branch by branch (N9 lot 4, §8 C1 and
§9 D1-D3), so that the native is a substitution of the Trace class (SV13):

* it stops at once when the residual of the guess meets the request, and fails at once when that
  residual is not finite;
* each step tests the curvature and ``r.r`` for finiteness, the curvature for definiteness, then
  alpha, the updated residual norm and beta for finiteness, and ``r.r > 0`` before beta;
* the recurrence runs in the units of its cycle scale, the norm of the first true residual, then
  of the last confirmed one (``generic_krylov.hpp:2036-2040``, ``rebase_cycle_residual``): r and d
  are the residual and the direction divided by the scale, the operator is A / scale, so that
  r.r, the curvature, alpha and beta are of order one whatever the magnitude of the data, and x
  stays in physical units; the norms of true residuals are measured scale-safely
  (``request.norm``);
* a step confirms the true residual when its estimate meets the request or when the normalized
  norm reaches the subnormal zone, and restarts the whole recurrence from it otherwise, with the
  confirmed norm as the new scale and r.r = 1;
* the last step allowed does not confirm inside the recurrence: it measures the true residual of
  the candidate once, as the wrapper does, when the estimate is reached, and is
  ``BudgetExhausted`` otherwise.

The failures are variants of T39 with their reason, the same on the native route: the cap is
``BudgetExhausted`` (« iteration limit »), both breakdowns are ``SolveFailed`` (« breakdown »), a
value that is not finite is ``SolveFailed`` (« invalid evaluation »), and a candidate whose true
residual fails the request is ``SolveFailed`` (« unverified solved value »).
"""
from __future__ import annotations

import sys

import pops.surface as pops
from pops.surface.math import infinity, sqrt

# sqrt of the smallest normal double: below it, the scaled residual norm is in the subnormal zone
# and the recursion is replaced by the true residual (needs_extreme_recurrence_rebase).
SQRT_MIN_NORMAL = sys.float_info.min ** 0.5


def finite(v):
    """A value is finite when it is not NaN and its magnitude is below infinity (D2)."""
    return (v == v) & (abs(v) < infinity)


@pops.solver
def cg(problem, region, preconditioner=None, max_iterations=2000):
    """The conjugate gradient of an operator that is symmetric and positive definite."""
    # preconditioner= is refused when cg is called, by @pops.solver, naming G10.
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) \
            or max_iterations < 1:
        raise ValueError("cg(max_iterations=n) takes an integer n >= 1, got %r"
                         % (max_iterations,))
    problem.require(symmetric=True, positive_definite=True)
    A, dot, request = problem.apply, problem.pairing, problem.request

    def invalid():
        return pops.fail(pops.SolveFailed, payload="invalid evaluation")

    def breakdown():
        return pops.fail(pops.SolveFailed, payload="breakdown")

    def budget():
        return pops.fail(pops.BudgetExhausted, payload="iteration limit")

    def unverified():
        return pops.fail(pops.SolveFailed, payload="unverified solved value")

    x0 = problem.guess
    r0 = problem.rhs - A(x0)
    norm0 = request.norm(r0)
    # The recurrence runs on the residual divided by its cycle scale, the norm of the last true
    # residual (generic_krylov.hpp:2036-2040, rebase_cycle_residual): r and d stay of order one
    # whatever the magnitude of the data, and x stays in physical units. The division is guarded
    # so that a zero start (which stops at once) never divides 0 by 0.
    def measured():
        return norm0

    def one():
        return 1.0

    unit0 = r0 / pops.choose(norm0 > 0, then=measured, otherwise=one)

    def paired():
        return dot(unit0, unit0)

    def unpaired():
        return 0.0

    rr0 = pops.choose(finite(norm0), then=paired, otherwise=unpaired)
    start = pops.record(x=x0, r=unit0, d=unit0, rr=rr0, scale=norm0, confirmed=norm0, k=0,
                        done=0)

    def stop(s):
        # The guess that meets the request stops before the first step (0 iterations); after a
        # step, only an accepted candidate stops, so that the count is the native's (D3).
        return request.met(s.confirmed) | (s.done > 0)

    def reached(normalized, scale):
        # The estimate in the cycle's units meets the physical threshold (normalized_threshold).
        return request.met(normalized * scale)

    def step(s):
        def started():
            # The operator in the cycle's units: (A d) / scale, the native's normalized
            # difference, so that alpha carries the scale and x moves in physical units.
            Ad = A(s.d) / s.scale
            curvature = dot(s.d, Ad)

            def checked():
                def definite():
                    alpha = s.rr / curvature

                    def advance():
                        return advanced(s, Ad, alpha)

                    return pops.choose(finite(alpha), then=advance, otherwise=invalid)

                return pops.choose(curvature > 0, then=definite, otherwise=breakdown)

            return pops.choose(finite(curvature) & finite(s.rr), then=checked, otherwise=invalid)

        # A start whose residual is not finite never pairs it: it fails at 0 iterations, as the
        # wrapper does (generic_krylov.hpp:2027).
        return pops.choose(finite(s.scale), then=started, otherwise=invalid)

    def advanced(s, Ad, alpha):
        x = s.x + alpha * s.d
        r = s.r - alpha * Ad
        rr = dot(r, r)
        norm = sqrt(rr)
        k = s.k + 1

        def judged():
            estimate = reached(norm, s.scale)

            def at_cap():
                # The last step does not confirm in the recurrence (l. 938-945); the wrapper
                # measures the candidate's true residual once (l. 2118).
                def candidate():
                    true = problem.rhs - A(x)
                    measured = request.norm(true)

                    def checked():
                        def accepted():
                            return pops.record(x=x, r=r, d=s.d, rr=rr, scale=s.scale,
                                               confirmed=measured, k=k, done=1)

                        return pops.choose(request.met(measured), then=accepted,
                                           otherwise=unverified)

                    return pops.choose(finite(measured), then=checked, otherwise=invalid)

                return pops.choose(estimate, then=candidate, otherwise=budget)

            def inside():
                extreme = (norm > 0) & (norm < SQRT_MIN_NORMAL)

                def confirm():
                    true = problem.rhs - A(x)
                    measured = request.norm(true)

                    def checked():
                        def solved():
                            return pops.record(x=x, r=r, d=s.d, rr=rr, scale=s.scale,
                                               confirmed=measured, k=k, done=1)

                        def restart():
                            # rebase_cycle_residual: the confirmed norm is the new cycle scale,
                            # the residual in its units has norm one, and the whole recurrence
                            # restarts from it (r.r = 1 * 1, l. 957).
                            unit = true / measured

                            def restarted():
                                return pops.record(x=x, r=unit, d=unit, rr=1.0, scale=measured,
                                                   confirmed=measured, k=k, done=0)

                            return restarted()

                        return pops.choose(request.met(measured), then=solved, otherwise=restart)

                    return pops.choose(finite(measured), then=checked, otherwise=invalid)

                def carry():
                    def positive():
                        beta = rr / s.rr

                        def carried():
                            return pops.record(x=x, r=r, d=r + beta * s.d, rr=rr, scale=s.scale,
                                               confirmed=s.confirmed, k=k, done=0)

                        return pops.choose(finite(beta), then=carried, otherwise=invalid)

                    return pops.choose(s.rr > 0, then=positive, otherwise=breakdown)

                return pops.choose(estimate | extreme, then=confirm, otherwise=carry)

            return pops.choose(k == max_iterations, then=at_cap, otherwise=inside)

        return pops.choose(finite(norm), then=judged, otherwise=invalid)

    result = region.iterate(start=start, step=step, stop=stop, max_steps=max_iterations)
    return problem.candidate(result.x, iterations=result.iterations,
                             residual_norm=result.confirmed)


# What a relation solve of a traced method reads of cg() (N15 lot « solves », §5.5): the property
# of the operator that it needs, checked against the claims of the components before the solve is
# lowered, and its description; the solve runs this body as the region of its node (N9 lot « La
# référence sous les solves de relation », §3.8).
cg.declare_relation(requires=("symmetric_positive_definite",),
                    description="the reference conjugate gradient of pops.lib.solve (SV9)")
