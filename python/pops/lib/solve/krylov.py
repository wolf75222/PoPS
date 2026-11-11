"""The reference conjugate gradient of ``pops.lib.solve`` (spec 0.3 SV9, 2.6.9 row ``cg``).

The body mirrors the native ``solve_cg`` and its wrapper branch by branch (N9 lot 4, §8 C1 and
§9 D1-D3), so that the native is a substitution of the Trace class (SV13):

* it stops at once when the residual of the guess meets the request, and fails at once when that
  residual is not finite;
* each step tests the curvature and ``r.r`` for finiteness, the curvature for definiteness, then
  alpha, the updated residual norm and beta for finiteness, and ``r.r > 0`` before beta;
* the recursion is rescaled by its cycle scale: the norm of the first residual, then of the last
  confirmed one; a step confirms the true residual when its estimate meets the request or when
  the scaled norm reaches the subnormal zone, and restarts the recurrence from it otherwise;
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
    rr0 = dot(r0, r0)
    norm0 = sqrt(rr0)
    start = pops.record(x=x0, r=r0, d=r0, rr=rr0, scale=norm0, confirmed=norm0, k=0, done=0)

    def stop(s):
        # The guess that meets the request stops before the first step (0 iterations); after a
        # step, only an accepted candidate stops, so that the count is the native's (D3).
        return request.met(s.confirmed) | (s.done > 0)

    def step(s):
        Ad = A(s.d)
        curvature = dot(s.d, Ad)

        def checked():
            def definite():
                alpha = s.rr / curvature

                def advance():
                    return advanced(s, Ad, alpha)

                return pops.choose(finite(alpha), then=advance, otherwise=invalid)

            return pops.choose(curvature > 0, then=definite, otherwise=breakdown)

        return pops.choose(finite(curvature) & finite(s.rr), then=checked, otherwise=invalid)

    def advanced(s, Ad, alpha):
        x = s.x + alpha * s.d
        r = s.r - alpha * Ad
        rr = dot(r, r)
        norm = sqrt(rr)
        k = s.k + 1

        def judged():
            reached = request.met(norm)

            def at_cap():
                # The last step does not confirm in the recurrence (l. 938-945); the wrapper
                # measures the candidate's true residual once (l. 2118).
                def candidate():
                    true = problem.rhs - A(x)
                    measured = request.norm(true)

                    def checked():
                        def accepted():
                            return pops.record(x=x, r=true, d=s.d, rr=rr, scale=s.scale,
                                               confirmed=measured, k=k, done=1)

                        return pops.choose(request.met(measured), then=accepted,
                                           otherwise=unverified)

                    return pops.choose(finite(measured), then=checked, otherwise=invalid)

                return pops.choose(reached, then=candidate, otherwise=budget)

            def inside():
                normalized = norm / s.scale
                extreme = (normalized > 0) & (normalized < SQRT_MIN_NORMAL)

                def confirm():
                    true = problem.rhs - A(x)
                    measured = request.norm(true)

                    def checked():
                        def solved():
                            return pops.record(x=x, r=true, d=s.d, rr=rr, scale=s.scale,
                                               confirmed=measured, k=k, done=1)

                        def restart():
                            # rebase_cycle_residual: the confirmed residual is the new cycle
                            # scale, and the whole recurrence restarts from it.
                            squared = measured * measured

                            def restarted():
                                return pops.record(x=x, r=true, d=true, rr=squared,
                                                   scale=measured, confirmed=measured, k=k,
                                                   done=0)

                            return pops.choose(finite(squared), then=restarted, otherwise=invalid)

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

                return pops.choose(reached | extreme, then=confirm, otherwise=carry)

            return pops.choose(k == max_iterations, then=at_cap, otherwise=inside)

        return pops.choose(finite(norm), then=judged, otherwise=invalid)

    result = region.iterate(start=start, step=step, stop=stop, max_steps=max_iterations)
    return problem.candidate(result.x, iterations=result.iterations)
