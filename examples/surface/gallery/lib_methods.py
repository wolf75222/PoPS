"""Time methods, step rules, controllers and checks of pops.lib.time, in the surface of section 2.6 (version 2).

A method is a function decorated with `@pops.method` or `@pops.multistep(...)`, an acceptance one decorated with
`@pops.assess`. PoPS supplies the parameters named after an entity (`step`, `candidate`, `region`); every other
parameter is an option, given by keyword at the call: `ssprk2(rate=model)` (SU5). `step.state`, `step.time`,
`step.memory` and `step.previous(k)` come from the accepted continuation, and `step.dt`, `step.tolerance`,
`step.inputs` and `step.choices` from the proposal [N(65)]. In a method a rate times a duration is an
increment and a state plus an increment is a state (SU10), so each formula is written as on paper; the increment
of a candidate and its exchanges are derived. A state is a table keyed by quantities, read as `S[q]`, whose
quantities `S.quantities` lists (SD42). A tableau is Python data, unrolled once at authoring time (E4).

`subcycle`, `composite_balance` and `finite` reach index sets through the parameter `region` (SU15, level 3); the
words of spec 0.3 that they use are spelled as 2.6.8 spells them (SD46, SD52). The schedules `pops.every(time=)`
and `pops.every(steps=)` are core words (SD43), not words of this module. A line marked `GAP C45 (spec 0.3)` uses
one of the three readings that spec 0.3 lacks for subcycling, recorded as gap C45 of §18.1 and owned by T08
(ROM-67, SD53, SD65): the advance of one level (`on=`, `coarser=` and `in_time=` of pops.substep), the
coarse-fine faces of DM2 (`coarse_fine_faces`) and the restriction onto the covered cells (`pops.restrict_onto`).
Its spelling is the proposed one, and 2.6.8 lists each.
"""

import pops
from pops.math import *
from pops.regions import (box, level, level_count, level_ratio, composite, faces, coarse_fine_faces, Parent, AnyOrder,
                          ReproducibleSum)


# Explicit Runge-Kutta methods

@pops.method
def forward_euler(step, rate):
    """Forward Euler: u_new = u + h L(u, t)."""
    return pops.Candidate(step.state + step.dt * rate(step.state, step.time))


@pops.method
def ssprk2(step, rate):
    """Heun's method, SSPRK2: u1 = u + h L(u, t); u_new = (u + u1 + h L(u1, t + h)) / 2."""
    u = step.state
    t = step.time
    h = step.dt
    first = u + h * rate(u, t)
    return pops.Candidate((u + first + h * rate(first, t + h)) / 2)


@pops.method
def ssprk3(step, rate):
    """SSPRK3 of Shu and Osher: u1 = u + h L(u, t); u2 = (3 u + u1 + h L(u1, t + h)) / 4;
    u_new = (u + 2 u2 + 2 h L(u2, t + h / 2)) / 3."""
    u = step.state
    t = step.time
    h = step.dt
    first = u + h * rate(u, t)
    second = (3 * u + first + h * rate(first, t + h)) / 4
    return pops.Candidate((u + 2 * (second + h * rate(second, t + h / 2))) / 3)


@pops.method
def rk4(step, rate):
    """Classical RK4: k1 = L(u, t), k2 = L(u + h k1 / 2, t + h / 2), k3 = L(u + h k2 / 2, t + h / 2),
    k4 = L(u + h k3, t + h); u_new = u + h (k1 + 2 k2 + 2 k3 + k4) / 6."""
    u = step.state
    t = step.time
    h = step.dt
    k1 = rate(u, t)
    k2 = rate(u + h / 2 * k1, t + h / 2)
    k3 = rate(u + h / 2 * k2, t + h / 2)
    k4 = rate(u + h * k3, t + h)
    return pops.Candidate(u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


@pops.method
def explicit_rk(step, tableau, rate):
    """k_i = L(u + h sum_(j<i) a_ij k_j, t + c_i h); u_new = u + h sum_i b_i k_i. With an embedded row b_low, the
    candidate carries evidence=pops.record(error=||u_new - u_low||, order=p_low), u_low = u + h sum_i b_low_i k_i,
    the norm being that of the tolerance and p_low the order that the tableau derives for b_low."""
    u = step.state
    t = step.time
    h = step.dt
    rates = []
    for i in range(len(tableau.b)):             # tableau data, unrolled once at authoring time (E4)
        stage = _plus_increments(u, _increments(h, tableau.a[i], rates))
        rates.append(rate(stage, t + tableau.c[i] * h))
    end = _plus_increments(u, _increments(h, tableau.b, rates))
    if tableau.b_low is None:
        return pops.Candidate(end)
    low = _plus_increments(u, _increments(h, tableau.b_low, rates))
    return pops.Candidate(end, evidence=pops.record(error=step.tolerance.norm(end - low),
                                                    order=tableau.embedded_order))


def _increments(h, coefficients, rates):
    """The increments h c_j k_j of the nonzero coefficients c_j, Python data at authoring time (E4)."""
    return [h * coefficients[j] * rates[j] for j in range(len(rates))
            if j < len(coefficients) and coefficients[j] != 0]


def _plus_increments(state, increments):
    """The state plus the sum of the increments, or the state itself when there is none."""
    if len(increments) == 0:
        return state
    return state + sum(increments)


# Tableaus (Python data)

class Tableau:
    """A Butcher tableau, Python data: the rows `a` (row i holds a_i1 ... a_i(i-1), or the whole row), the weights
    `b`, the nodes `c` and the embedded weights `b_low`. The order of b and that of b_low are derived from the
    order conditions sum_i b_i Phi_i(t) = 1 / gamma(t) over the rooted trees t, exactly for integers and
    Fractions and within 1e-12 for floats (5.11, SU10, SD45); they are never stated."""

    def __init__(self, a, b, c, b_low=None):
        self.a = a
        self.b = b
        self.c = c
        self.b_low = b_low
        self.order = _order(a, b)
        self.embedded_order = None if b_low is None else _order(a, b_low)


def _order(a, weights):
    """The largest p, at most s + 1 for s stages, such that sum_i w_i Phi_i(t) = 1 / gamma(t) for every rooted
    tree t of at most p vertices, with Phi_i(t) = prod over the subtrees t_k of t of sum_j a_ij Phi_j(t_k)."""
    stages = len(weights)
    order = 0
    for p in range(1, stages + 2):
        for tree in _trees(p):
            phi = _stage_values(a, stages, tree)
            if not _holds(sum([weights[i] * phi[i] for i in range(stages)]), Fraction(1, _density(tree))):
                return order
        order = p
    return order


def _trees(size):
    """Every rooted tree of `size` vertices, written as the list of its subtrees. Some trees appear more than
    once, which repeats a condition and changes no order."""
    if size == 1:
        return [[]]
    trees = []
    for first in range(1, size):
        for subtree in _trees(first):
            for rest in _trees(size - first):
                trees.append([subtree] + rest)
    return trees


def _stage_values(a, stages, tree):
    """Phi_i(t) for each stage i, Phi_i of a single vertex being 1."""
    values = [Fraction(1)] * stages
    for subtree in tree:
        inner = _stage_values(a, stages, subtree)
        values = [values[i] * sum([_entry(a, i, j) * inner[j] for j in range(stages)]) for i in range(stages)]
    return values


def _entry(a, i, j):
    """a_ij, zero beyond the stored part of row i."""
    if j < len(a[i]):
        return a[i][j]
    return 0


def _density(tree):
    """gamma(t) = |t| prod_k gamma(t_k)."""
    result = _size(tree)
    for subtree in tree:
        result = result * _density(subtree)
    return result


def _size(tree):
    """|t|, the number of vertices of t."""
    return 1 + sum([_size(subtree) for subtree in tree])


def _holds(value, target):
    """value = target, exactly for integers and Fractions and within 1e-12 for floats (SU10)."""
    if isinstance(value, float):
        return abs(value - target) <= 1e-12
    return value == target


# The 3(2) pair of Bogacki and Shampine: order 3, embedded order 2, the last stage equal to the first of the next.
bogacki_shampine = Tableau(
    a=[[],
       [Fraction(1, 2)],
       [0, Fraction(3, 4)],
       [Fraction(2, 9), Fraction(1, 3), Fraction(4, 9)]],
    b=[Fraction(2, 9), Fraction(1, 3), Fraction(4, 9), 0],
    c=[0, Fraction(1, 2), Fraction(3, 4), 1],
    b_low=[Fraction(7, 24), Fraction(1, 4), Fraction(1, 3), Fraction(1, 8)],
)

# The 5(4) pair of Dormand and Prince: order 5, embedded order 4, the last stage equal to the first of the next.
dormand_prince = Tableau(
    a=[[],
       [Fraction(1, 5)],
       [Fraction(3, 40), Fraction(9, 40)],
       [Fraction(44, 45), Fraction(-56, 15), Fraction(32, 9)],
       [Fraction(19372, 6561), Fraction(-25360, 2187), Fraction(64448, 6561), Fraction(-212, 729)],
       [Fraction(9017, 3168), Fraction(-355, 33), Fraction(46732, 5247), Fraction(49, 176), Fraction(-5103, 18656)],
       [Fraction(35, 384), 0, Fraction(500, 1113), Fraction(125, 192), Fraction(-2187, 6784), Fraction(11, 84)]],
    b=[Fraction(35, 384), 0, Fraction(500, 1113), Fraction(125, 192), Fraction(-2187, 6784), Fraction(11, 84), 0],
    c=[0, Fraction(1, 5), Fraction(3, 10), Fraction(4, 5), Fraction(8, 9), 1, 1],
    b_low=[Fraction(5179, 57600), 0, Fraction(7571, 16695), Fraction(393, 640), Fraction(-92097, 339200),
           Fraction(187, 2100), Fraction(1, 40)],
)


class ImexTableau:
    """A pair of Butcher tableaus with the same number of stages, Python data (SD45): `explicit` for the explicit
    terms and `implicit`, diagonally implicit, for the implicit terms. The pair is neither a tuple (SU14) nor a
    pops.record, which holds PoPS values; each tableau derives its own orders."""

    def __init__(self, explicit, implicit):
        self.explicit = explicit
        self.implicit = implicit


# Implicit and IMEX methods

@pops.method
def backward_euler(step, rate):
    """u_new = u + h L(u_new, t + h): one solve, unnamed, which solvers= keys by the method (2.6.2, SD67)."""
    X = pops.unknown(step.state.space)
    return pops.Candidate(pops.solve(X == step.state + step.dt * rate(X, step.time + step.dt), X))


@pops.multistep(depth=1, startup=backward_euler, on_regrid=pops.invalidate())
def bdf2(step, rate):
    """alpha u_new - (1 + r) u_n + beta u_(n-1) = h L(u_new, t + h), with r = h / h_prev, alpha = (1 + 2 r) / (1 + r)
    and beta = r^2 / (1 + r) [N(83)]: a relation D == J whose weights sum to zero, solved for u_new (SU10). The
    first step runs the startup word, which the decorator takes uncalled and calls with the options of the call,
    backward_euler(rate=...) unless startup= at the call states another (SD45). Under a regrid the memory is
    invalidated and the next step runs the startup again (CN7), unless on_regrid= at the call states another
    policy, bdf2(rate=model, on_regrid=pops.remap(T)) (CN5, SD54). Its one solve is unnamed and keyed by the
    method (SD67)."""
    previous = step.previous(1)
    h = step.dt
    r = h / previous.dt
    X = pops.unknown(step.state.space)
    relation = ((1 + 2 * r) / (1 + r) * X - (1 + r) * step.state + r * r / (1 + r) * previous.state
                == h * rate(X, step.time + h))
    return pops.Candidate(pops.solve(relation, X))


@pops.method
def imex_euler(step, explicit, implicit):
    """u_new = u + h E(u, t) + h I(u_new, t + h): forward Euler on the explicit terms and backward Euler on the
    implicit terms; one solve, unnamed and keyed by the method (SD67)."""
    u = step.state
    t = step.time
    h = step.dt
    X = pops.unknown(step.state.space)
    return pops.Candidate(pops.solve(X == u + h * explicit(u, t) + h * implicit(X, t + h), X))


@pops.method
def ars222(step, explicit, implicit):
    """ARS(2,2,2) of Ascher, Ruuth and Spiteri, with gamma = 1 - 1 / sqrt(2) and delta = 1 - 1 / (2 gamma):
    U2 = u + gamma h E(u, t) + gamma h I(U2, t + gamma h);
    U3 = u + delta h E(u, t) + (1 - delta) h E(U2, t + gamma h) + (1 - gamma) h I(U2, t + gamma h)
         + gamma h I(U3, t + h);
    u_new = U3. The two solves are named stage2 and stage3."""
    gamma = 1 - 1 / sqrt(2)
    delta = 1 - 1 / (2 * gamma)
    u = step.state
    t = step.time
    h = step.dt
    first = explicit(u, t)
    X2 = pops.unknown(step.state.space)
    U2 = pops.solve(X2 == u + gamma * h * first + gamma * h * implicit(X2, t + gamma * h), X2, name="stage2")
    X3 = pops.unknown(step.state.space)
    U3 = pops.solve(X3 == u + delta * h * first + (1 - delta) * h * explicit(U2, t + gamma * h)
                    + (1 - gamma) * h * implicit(U2, t + gamma * h) + gamma * h * implicit(X3, t + h),
                    X3, name="stage3")
    return pops.Candidate(U3)


@pops.method
def imex_rk(step, tableau, explicit, implicit):
    """U_i = u + h sum_(j<i) (ae_ij E_j + ai_ij I_j) + h ai_ii I(U_i, t + ci_i h), solved for U_i where ai_ii is not
    zero (the solve named stage{i}), with E_j = E(U_j, t + ce_j h) and I_j = I(U_j, t + ci_j h);
    u_new = u + h sum_i (be_i E_i + bi_i I_i). `tableau` is an ImexTableau: (ae, be, ce) and (ai, bi, ci)."""
    u = step.state
    t = step.time
    h = step.dt
    ex = tableau.explicit
    im = tableau.implicit
    explicit_rates = []
    implicit_rates = []
    for i in range(len(ex.b)):                  # tableau data, unrolled once at authoring time (E4)
        known = _plus_increments(u, _increments(h, ex.a[i], explicit_rates)
                                 + _increments(h, im.a[i], implicit_rates))
        diagonal = _entry(im.a, i, i)
        if diagonal == 0:
            stage = known
        else:
            X = pops.unknown(step.state.space)
            stage = pops.solve(X == known + h * diagonal * implicit(X, t + im.c[i] * h), X, name=f"stage{i + 1}")
        explicit_rates.append(explicit(stage, t + ex.c[i] * h))
        implicit_rates.append(implicit(stage, t + im.c[i] * h))
    increments = _increments(h, ex.b, explicit_rates) + _increments(h, im.b, implicit_rates)
    return pops.Candidate(_plus_increments(u, increments))


# Splitting

@pops.method
def strang(step, first, second, complete=None):
    """S(h) = A(h / 2) B(h) A(h / 2): `first` over half the step, `second` over the whole step, `first` over the
    other half, each child on its own clock (T20, T36). `complete`, an advance on the clock of `first`, completes
    each half of `first` up to its end, t + h / 2 or t + h, by a child over the time that the half left, as
    after a relaxed method that reaches only gamma h / 2 (5.10, T26, T36 case 1, SD54)."""
    a = _half(first, complete, step, step.time + step.dt / 2, step.dt / 2)
    b = pops.substep(second, start=a, dt=step.dt)
    c = _half(first, complete, b, step.time + step.dt, step.dt / 2)
    return pops.Candidate(c)


def _half(first, complete, start, end, dt):
    """A half of `first` over dt from `start`, then, when `complete` is given, a child of `complete` from the
    time that the half reached up to `end` (5.10)."""
    half = pops.substep(first, start=start, dt=dt)
    if complete is None:                        # Python data, decided once at authoring time (E4)
        return half
    return pops.substep(complete, start=half, dt=end - half.time)


# Levels of a hierarchy (SU15, level 3)

def interpolate(history, at):
    """U(t) = (1 - theta) U_0 + theta U_1, theta = (t - t_0) / (t_1 - t_0): linear in time between the two entries
    of history = [earlier, later], steps or children each read as .state and .time (T7, SD60). It is called
    interpolate([earlier, later], at=t), or passed without a call, in_time=interpolate."""
    earlier = history[0]
    later = history[1]
    theta = (at - earlier.time) / (later.time - earlier.time)
    return (1 - theta) * earlier.state + theta * later.state


@pops.method
def subcycle(step, region, method, coarse_fine, in_time=interpolate, ratio=None, restriction=None):
    """One step of a hierarchy, level by level from the coarsest (5.10, DM12, DM14, X13, X14). Level l advances
    over h_l with `method`. Level l + 1 then folds r advances over h_l / r from the provisional state, r being
    `ratio` when it is stated and otherwise the ratio of the layout between the two levels, level_ratio(Ω, l);
    ratio=1 steps the levels together (SD65). Each stage of these advances reads its coarse-fine ghosts at its own
    time t as coarse_fine applied to in_time([start, end], at=t), start and end the coarser level before and after
    its advance (T23, SD60). The coarse contribution J_c of each coarse face F on which fine faces lie is replaced
    by J_f(F), the sum of the fine contributions on F over the substeps, so that the coarse cells receive
    B_c (J_f - J_c) once. pops.restrict_onto then gives each covered cell of level l the restriction of level
    l + 1 by `restriction`, the restriction of the representation when it is left out, mean_of_children() for a
    CellAverage, and the uncovered cells keep their values (DM14, SD65). Children stay provisional until the root
    accepts. The quantities of the state share one support, whose layout is the hierarchy."""
    domain = step.state.quantities[0].support
    # level_count(Ω) and level_ratio(Ω, l) read the descriptor Hierarchy(levels, ratio) of IR3 (SD65), Python data
    # at tracing time.
    levels = level_count(domain)

    def advance_from(lvl, start, h, coarser):
        """Level lvl over h from `start`, then the finer levels, then the reflux and the restriction onto level
        lvl; `coarser` is None on the coarsest level and [start, end] of the coarser level on the others. A
        Python recursion over the levels, unrolled once at tracing time (E4); the region that it builds holds no
        recursion (IR6)."""
        advanced = _level_advance(method, start, h, level(domain, lvl), coarse_fine, in_time, coarser)
        if lvl + 1 == levels:
            return advanced

        space = level_ratio(domain, lvl)        # the ratio of the layout, which Parent reads
        steps = _substeps(ratio, space)

        def fine_substep(child, k):
            """Substep k of level lvl + 1, from start.time + k h / steps over h / steps, which reads level lvl
            between `start` and `advanced`."""
            return advance_from(lvl + 1, child, h / steps, [start, advanced])

        # fine_substep receives at index k the value read at the time where substep k starts on the clock of level
        # lvl + 1: at k = 0 the time of `start`, not the time where `advanced` ends on the clock of level lvl (the
        # fold row of 2.6.8, 5.2, T35, SD77), so that level lvl + 1 reads level lvl between `start` and `advanced`.
        fine = region.fold(box(steps), start=advanced, step=fine_substep)   # a size of one axis is its number (SU14)
        refluxed = _reflux(region, domain, lvl, advanced, fine, space)
        # GAP C45 (spec 0.3): DM14 restricts the fine solution onto the covered cells, but no construct gives the
        # child in which they hold it; here pops.restrict_onto(child, level(Ω, l), transfer=T), the uncovered cells
        # keeping their values and transfer=None being the restriction of the representation (SD65).
        return pops.restrict_onto(refluxed, level(domain, lvl), transfer=restriction)

    return pops.Candidate(advance_from(0, step, step.dt, None))


def _substeps(ratio, space):
    """The number of substeps of a finer level: `ratio` when it is stated, else the ratio of the layout."""
    if ratio is None:                           # Python data, decided once at tracing time (E4)
        return space
    return ratio


def _level_advance(method, start, h, cells, coarse_fine, in_time, coarser):
    """`method` on the cells of one level over h, the other levels unchanged. On a finer level, a stage at time t
    reads its coarse-fine ghosts as coarse_fine applied to in_time(coarser, at=t), coarser = [start, end] of the
    coarser level (DM14, T23). pops.substep takes an advance (2.6.2), so the method is wrapped in pops.Advance,
    whose step pops.substep gives (SD60)."""
    # GAP C45 (spec 0.3): 5.10 calls advance on the level method, which spec 0.3 does not spell: neither
    # restricting a child advance to the cells of one level, on the clock of that level (here on=), nor giving it
    # the Transfer of its coarse-fine ghosts and the coarser level that they read at the time of each stage (here
    # coarse_fine=, coarser= and in_time=, proposed by SD60).
    if coarser is None:
        return pops.substep(pops.Advance(method), start=start, dt=h, on=cells)
    return pops.substep(pops.Advance(method), start=start, dt=h, on=cells, coarse_fine=coarse_fine,
                        coarser=coarser, in_time=in_time)


def _reflux(region, domain, lvl, coarse, fine, ratio):
    """J_f(F) = sum of J(f) over the fine faces f that lie on the coarse face F, J the contribution of a term to the
    increment of level lvl + 1 over its substeps, with their time weights, which the fine child holds; the coarse
    contribution J_c(F) of each face F on which fine faces lie is then replaced by J_f(F) (X13, X14, DM12), in a
    reproducible order (PF9)."""
    # fine.increment is the increment that T35 gives the child, and fine.increment.terms the terms whose
    # contributions it holds (SD65). pops.replace_contribution(child, old, new, faces=) returns a child equal to
    # `child` in which old is replaced by new on the faces, keeping the replacements already in `child`, so the
    # loop threads it through the axes and the terms from fine, the last child of level lvl + 1 (X13).
    # The core owns the replacement (DM12, SD46), so replace_contribution is called as pops.replace_contribution,
    # although X13 lists it in pops.lib.time.
    corrected = fine
    for axis in domain.axes:                    # Python data at tracing time (E4)
        # GAP C45 (spec 0.3): DM2 lists the coarse-fine faces of a hierarchy with their Incidence maps, but no
        # reading names them; here coarse_fine_faces(Ω, l, axis), the faces of level l on which faces of level
        # l + 1 lie, a set that faces= of pops.replace_contribution takes (SD60). faces(level(Ω, l), axis) is the
        # faces(κ, a) of IR2 on the lattice of a level (SD65).
        fine_faces = faces(level(domain, lvl + 1), axis)
        coarse_faces = faces(level(domain, lvl), axis)
        for term in fine.increment.terms:
            fine_sum = region.scatter_reduce(_on_coarse_faces(region, fine.increment[term], fine_faces, axis, ratio),
                                             targets={Parent(ratio=ratio): 1}, into=coarse_faces, combine=_plus,
                                             order=ReproducibleSum())
            corrected = pops.replace_contribution(corrected, coarse.increment[term], fine_sum,
                                                  faces=coarse_fine_faces(domain, lvl, axis))
    return corrected


def _on_coarse_faces(region, values, fine_faces, axis, ratio):
    """The entries of `values` on the fine faces that lie on a coarse face, whose coordinate along the axis is ratio
    times that of their parent, and 0 on the fine faces inside a coarse cell; a sum along Parent then gathers onto
    each coarse face F the fine faces that lie on F."""
    parent = Parent(ratio=ratio)

    def entry(i):
        lies_on = equal(region.coord(i, axis), ratio * region.coord(parent(i), axis))
        return where(lies_on, region.at(values, i), 0)

    return region.map(fine_faces, entry)


def _contents(region, values, cells):
    """|K| U(K) on each cell K of the set, the contents of a CellAverage U (D4); region.measure(cells) reads the
    geometry arrays of 4.3 (SD65)."""
    measure = region.measure(cells)

    def content(i):
        return region.at(measure, i) * region.at(values, i)

    return region.map(cells, content)


def _plus(x, y):
    """The combining operation of a sum, for region.reduce and region.scatter_reduce (SU15)."""
    return x + y


# Step rules and controllers (dt=)

def cfl_limit(flux, cfl, max_dt=None):
    """A step rule after master's AdaptiveCFL: dt(cell) = cfl min_a h_a / max_a s_a, h_a the width of the cell along
    the axis a and s_a = cell.speed(flux, a), the largest |speed| that the face fluxes of the scheme of `flux` record
    on the two faces of the cell along a, or else the spectral radius of `flux` at the cell value (R9, SD44);
    infinity where every s_a is zero, and max_dt caps the result when it is given. The step is the smallest value
    over the cells, which on a uniform layout is master's cfl h_min / max speed without diffusion (SD70). `flux` may
    be a list of fluxes, whose smallest limit is kept (SD67). flux.quantity, q.support and Ω.axes are handle
    reads."""
    fluxes = flux if isinstance(flux, list) else [flux]     # Python data, decided once at authoring time (E4)

    def limit(cell):
        bound = smallest([_flux_limit(cell, f, cfl) for f in fluxes])
        if max_dt is None:                      # Python data, decided once at authoring time (E4)
            return bound
        return smallest([bound, max_dt])

    return limit


def _flux_limit(cell, flux, cfl):
    """cfl min_a h_a / max_a s_a for one flux, infinity where every s_a is zero."""
    axes = flux.quantity.support.axes
    speed = largest([cell.speed(flux, axis) for axis in axes])
    width = smallest([cell.width(axis) for axis in axes])
    return where(speed > 0, cfl * width / speed, infinity)


@pops.assess(memory=pops.record(previous_error=1.0))
def pi_controller(step, candidate, safety=0.9, integral_gain=0.3, proportional_gain=0.4, smallest_factor=0.2,
                  largest_factor=5.0):
    """f = clamp(safety e^(-(integral_gain + proportional_gain) / q) e_prev^(proportional_gain / q),
    smallest_factor, largest_factor), e = candidate.evidence.error, scaled by the tolerance, and
    q = candidate.evidence.order; accepts when e <= 1 and proposes f dt after an acceptance and after a rejection
    (5.10, T29). The first step is first_dt= of pops.Advance (2.6.2, SD74). The gains are not named `integral`
    and `proportional`, so that no parameter shadows a word of pops.math."""
    error = candidate.evidence.error
    order = candidate.evidence.order
    factor = clamp(safety * error ** (-(integral_gain + proportional_gain) / order)
                   * step.memory.previous_error ** (proportional_gain / order),
                   smallest_factor, largest_factor)
    return where(error <= 1,
                 pops.Accept(next_dt=step.dt * factor, memory=pops.record(previous_error=error)),
                 pops.Reject(next_dt=step.dt * factor, reason=error))


# Checks (accept=)

@pops.assess
def finite(step, candidate, region):
    """Accepts when every value v of the candidate is finite: equal(v, v) fails on NaN and abs(v) < infinity on an
    infinity. One reduce per quantity counts the values that fail (IR4). It is the accept= of an Advance that
    states none (SD35)."""
    failures = sum([_count_not_finite(region, candidate.state[q]) for q in candidate.state.quantities])
    return where(equal(failures, 0), pops.Accept(), pops.Reject(reason=failures))


def _count_not_finite(region, values):
    """The number of entries of an array that are NaN or infinite; an integer sum is exact in any order."""
    indices = region.domain(values)

    def not_finite(i):
        v = region.at(values, i)
        return where(all_of([equal(v, v), abs(v) < infinity]), 0, 1)

    return region.reduce(indices, region.map(indices, not_finite), combine=_plus, order=AnyOrder())


@pops.assess
def composite_balance(step, candidate, region, atol):
    """Accepts when |I_q(candidate) - I_q(step) - E_q| <= atol for each quantity q, where I_q(U) is the sum over the
    cells K of composite(Ω) of |K| U_q(K), the composite inventory, which leaves out the covered coarse cells
    (X14), and E_q the net delivery of the candidate's increment to q through the physical boundary over the step,
    with its productions (8.6, DM12). The sums run in a reproducible order (PF9)."""
    gaps = []
    for q in step.state.quantities:             # Python data at tracing time (SD42)
        cover = composite(q.support)
        before = _inventory(region, cover, step.state[q])
        after = _inventory(region, cover, candidate.state[q])
        # increment.boundary_exchange(q) reads the external exchange J_b^ext of X1 with the productions of q,
        # each physical face counted once, at its finest level (SD65).
        gaps.append(abs(after - before - candidate.increment.boundary_exchange(q)))
    return where(all_of([gap <= atol for gap in gaps]), pops.Accept(), pops.Reject(reason=largest(gaps)))


def _inventory(region, cover, values):
    """The sum over the cells K of the cover of |K| U(K), U the values of one quantity."""
    return region.reduce(cover, _contents(region, values, cover), combine=_plus, order=ReproducibleSum())
