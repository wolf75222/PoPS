"""Schemes of pops.lib.numerics and pops.lib.numerics.kinetic, in the surface of section 2.6 (version 2).

Each scheme is a function decorated with `@pops.scheme`. PoPS supplies its entity parameters (`face`, `cell`,
`term`, `axis`, `depth`); every other parameter is an option, given by keyword at the call:
`rusanov(reconstruction=muscl(limiter=minmod))` (SU5). `term` is the key that the scheme is placed under in
`schemes=`, a joint key `pops.jointly(...)`, for a boundary policy each relation of `model.on` that the policy
realizes, or, for a gradient scheme, the expression whose gradient it gives (kind Gradient, SD66). A scheme reads
the data of its key through `term` and never restates the model (2.6.4). The kind
of the key, `term.kind`, and the form of a boundary relation, `term.form`, are Python data, words of
`pops.discrete.kinds` and `pops.discrete.forms` compared with `is`; a face scheme runs once per axis of its
support, so `face.axis` is Python data too (SD42). Limiters and summands are plain functions, passed without
call. No body holds a lambda, a tuple used as a value or a record written as a dictionary; a dictionary here is a
table keyed by handles (SU14).

A scheme that needs a form of its key states it with `term.expect(form)`, which returns nothing and refuses the
scheme (`RealizationMismatch`) when the key does not have it. The form is written as the key was declared, an
expression or a table keyed by quantity, with the readings of `term` and the words of pops.math; it matches when
the normalized IR of the key equals that of the form (CC2, SD36, SD47).

A face flux records the normal speed that it used, `FaceFlux(..., speed=s)`, which `cell.speed` and `cfl_limit`
read (SD44). `discrete_maxwellian` belongs to pops.lib.numerics.kinetic.
"""

import pops
from pops.math import *
from pops.discrete import (FaceFlux, FaceValue, CellRate, CellValue, CellResidual, Reconstruct, Ghost, BoundaryFlux,
                           Transfer, sum_faces, sum_children, sum_fibre, spectral_radius, slowest_speed, fastest_speed,
                           kinds, forms)


# Fluxes of a transport term

@pops.scheme
def rusanov(face, term, reconstruction=None):
    """F* = (F(uL).n + F(uR).n) / 2 - s (uR - uL) / 2, s the larger spectral radius of the two
    sides (R9), which the flux records as its speed."""
    q = term.quantity
    n = face.normal
    left = face.left(q)
    right = face.right(q)
    speed = largest([spectral_radius(term, left, n), spectral_radius(term, right, n)])
    flux_left = dot(term.law(states={q: left}), n)
    flux_right = dot(term.law(states={q: right}), n)
    return FaceFlux(0.5 * (flux_left + flux_right) - 0.5 * speed * (right - left),
                    reconstruction=reconstruction, speed=speed)


@pops.scheme
def upwind(face, term, speed=None, reconstruction=None):
    """F* = s u_L where s > 0 and s u_R elsewhere, s being recorded as the speed of the flux.
    Without `speed`, the flux must be F(u) = F(1) u, linear in the scalar state u: term.expect
    refuses another flux (RealizationMismatch, whose message names rusanov() and hll()), and the
    face is the one of rusanov(), whose spectral radius is |F(1).n| on both sides, so that it is
    the upwind flux in master's arithmetic. With `speed=f`, s = f(face) is the normal speed, the
    velocity of the term dotted with face.normal, whose sign follows the orientation of the face,
    and nothing is checked: u is upwinded at that speed, which pops.inspect reports as stated. On
    a product support `speed=` is a table keyed by factor, `speed={velocity: f}`, and a factor
    that the table omits takes the derived speed (SD73)."""
    q = term.quantity
    n = face.normal
    left = face.left(q)
    right = face.right(q)
    stated = _stated_speed(face, speed)
    if stated is None:                      # Python data, decided once at authoring time (E4)
        term.expect(term.law(states={q: 1}) * q,
                    write="rusanov() or hll(), which take the wave speeds of the flux")
        radius = largest([spectral_radius(term, left, n), spectral_radius(term, right, n)])
        flux_left = dot(term.law(states={q: left}), n)
        flux_right = dot(term.law(states={q: right}), n)
        return FaceFlux(0.5 * (flux_left + flux_right) - 0.5 * radius * (right - left),
                        reconstruction=reconstruction, speed=radius)
    s = stated(face)
    return FaceFlux(where(s > 0, s * left, s * right), reconstruction=reconstruction, speed=s)


def _stated_speed(face, speed):
    """The stated speed at the face: `speed` itself, or the entry of a table {factor: f} whose factor holds
    face.axis, the handle of the axis that this run of the scheme covers (SD42), compared with `is` to the axes
    of each factor (Ω.factors, Ω.axes); None where no speed is stated there, so that the speed is derived."""
    if not isinstance(speed, dict):
        return speed
    for factor in speed:
        for axis in factor.axes:
            if axis is face.axis:
                return speed[factor]
    return None


@pops.scheme
def hll(face, term, reconstruction=None):
    """F* = F_L where s_L >= 0, F_R where s_R <= 0, and
    (s_R F_L - s_L F_R + s_L s_R (u_R - u_L)) / (s_R - s_L) elsewhere, with F_L = F(u_L).n,
    F_R = F(u_R).n, s_L the smaller of the two slowest speeds and s_R the larger of the two
    fastest speeds (R9), as master's HLL. The flux records the larger of |s_L| and |s_R| as its
    speed."""
    q = term.quantity
    n = face.normal
    left = face.left(q)
    right = face.right(q)
    slow = smallest([slowest_speed(term, left, n), slowest_speed(term, right, n)])
    fast = largest([fastest_speed(term, left, n), fastest_speed(term, right, n)])
    flux_left = dot(term.law(states={q: left}), n)
    flux_right = dot(term.law(states={q: right}), n)
    star = (fast * flux_left - slow * flux_right + slow * fast * (right - left)) \
        * (1.0 / (fast - slow))
    return FaceFlux(where(slow >= 0, flux_left, where(fast <= 0, flux_right, star)),
                    reconstruction=reconstruction, speed=largest([abs(slow), abs(fast)]))


# Reconstruction and limiters

@pops.scheme
def muscl(cell, term, axis, limiter):
    """u_(c+1/2) = u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2, the trace on the face on the positive side of
    `axis`; the trace on the negative side mirrors it (2.6.4). A record state is limited field by field, since
    record arithmetic acts field by field (SD71). As master's MUSCL face value."""
    q = term.quantity
    centre = cell.value(q)
    backward = centre - cell.shift(axis, -1).value(q)
    forward = cell.shift(axis, 1).value(q) - centre
    return Reconstruct(centre + limiter(backward, forward) / 2)


def minmod(a, b):
    """minmod(a, b) = 0 where a b <= 0; elsewhere a where |a| < |b|, and b otherwise (master's Minmod)."""
    return where(a * b > 0, where(abs(a) < abs(b), a, b), 0)


def vanleer(a, b):
    """vanleer(a, b) = 0 where a b <= 0; elsewhere 2 a b / (a + b) (master's VanLeer)."""
    return where(a * b > 0, 2 * a * b / (a + b), 0)


# Diffusive fluxes, sources, definitions and constraints

@pops.scheme
def two_point(face, term):
    """F* = F(u_f, g_f).n, with u_f = (u_L + u_R) / 2 the face mean of each quantity and
    g_f = (w_R - w_L) / face.distance n the two-point normal gradient of each w that the law
    differentiates, a gradient variable or a first-order input (E2, SD51)."""
    states = {r: face.average(r) for r in term.quantities}
    gradients = {w: face.normal_gradient(w) * face.normal for w in _differentiated(term)}
    return FaceFlux(dot(term.law(states=states, gradients=gradients), face.normal))


def _differentiated(term):
    """Every w that gradients= must give: the gradient variables of the key, then its first-order inputs, the
    fields whose gradient the law reads as a first-order transport input, in the order of the law (E2, SD51)."""
    return term.gradient_variables + term.first_order_inputs


@pops.scheme
def pointwise(cell, term, gradient=None):
    """The law S of the key at the cell values u_c of the quantities that it reads: a source or an interaction as
    the CellRate S(u_c), S being for an interaction the table of its deliveries; a definition or an inventory as
    the CellValue S(u_c), S being for an inventory the table of its densities (SD47); a model relation K == 0, such
    as a closure, as the CellResidual K(u_c), its unknowns being among the quantities that it reads and determined
    by the solve of its handle. An integral over a support that is only integrated over uses the nodes and weights
    of the quadrature rule that is the layout of that support (SD31). A law that reads grad(w) of an analytic
    definition, one that reads only coordinates, constants and parameters, takes it exactly; a law that reads the
    gradient of a discrete field takes it from the cell gradient scheme `gradient`, such as centred_gradient(),
    and is refused without it (AmbiguousElaboration, naming gradient=) (SD72)."""
    states = {r: cell.value(r) for r in term.quantities}
    if gradient is None:                    # Python data: only exact gradients of analytic definitions (SD72)
        value = term.law(states=states)
    else:
        value = term.law(states=states,
                         gradients={w: cell.gradient(w, scheme=gradient) for w in _differentiated(term)})
    kind = term.kind
    if kind is kinds.Definition or kind is kinds.Inventory:
        return CellValue(value)
    if kind is kinds.Relation:
        return CellResidual(value)
    return CellRate(value)


@pops.scheme
def scharfetter_gummel(face, term):
    """F*.n = (D / dx) (B(-P) n_L - B(P) n_R), B(z) = z / (exp(z) - 1), P = (v.n) dx / D the Peclet number of the
    face and dx = face.distance: the joint flux of F = v n - D grad(n), the drift and the diffusion of the density
    n under the key pops.jointly(drift, diffusion) (D8). v and D are read from the joint law, which is affine in n
    and grad(n): v = F(n = 1, grad(n) = 0) and D = F(n = 0, grad(n) = -n_f).n_f, with the other quantities at
    their face means and the other gradients two-point: the potential psi of the drift is read among
    term.first_order_inputs, and grad(psi) enters by its two-point gradient (E2, SD51). Each branch keeps its
    exponential below 1."""
    density = term.quantity
    n = face.normal
    dx = face.distance

    def states(value):
        return {r: value if r is density else face.average(r) for r in term.quantities}

    def gradients(value):
        return {w: value if w is density else face.normal_gradient(w) * n for w in _differentiated(term)}

    velocity = term.law(states=states(1), gradients=gradients(0 * n))
    diffusivity = dot(term.law(states=states(0), gradients=gradients(-n)), n)
    peclet = dot(velocity, n) * dx / diffusivity
    left = face.left(density)
    right = face.right(density)
    return FaceFlux(diffusivity / dx * where(peclet >= 0,
                                             _bernoulli(-peclet) * (left - exp(-peclet) * right),
                                             _bernoulli(peclet) * (exp(peclet) * left - right)))


def _bernoulli(z):
    """B(z) = z / (exp(z) - 1), B(0) = 1, by its series 1 - z / 2 + z^2 / 12 where |z| < 1e-4."""
    return where(abs(z) < 1e-4, 1 - z / 2 + z * z / 12, z / expm1(z))


@pops.scheme
def poisson_residual(cell, term):
    """R_c = -sum_f G_f.n_f |f| - int_c load, for a constraint -div(G) == load: G_f = G(u_f, g_f) at the face mean
    of the quantities that G reads and the two-point normal gradient of each unknown, each face seen from the cell
    with its normal pointing out (SD40); the load is read from the constraint and integrated by
    cell.load(term.load), with the quadrature of the representation of the constraint's unknown; a load that
    integrates over another factor of a product, such as int f dv, is read as the sum over the fibre (sum_fibre,
    4.3, SD72). term.law gives the flux G of a constraint whose operator is -div(G), and refuses another operator
    (SD42)."""
    G = term.law(states={r: r for r in term.quantities}, gradients={w: grad(w) for w in term.unknowns})
    term.expect(-div(G) == term.load)

    def face_flux(face):
        states = {r: face.average(r) for r in term.quantities}
        gradients = {w: face.normal_gradient(w) * face.normal for w in term.unknowns}
        return dot(term.law(states=states, gradients=gradients), face.normal) * face.measure

    return CellResidual(-sum_faces(cell, face_flux) - cell.load(term.load))


@pops.scheme
def centred_gradient(cell, term):
    """g_K, the centred gradient of e on the cell K: g_K.e_a = (e(K + 1_a) - e(K - 1_a)) / (x_a(K + 1_a) -
    x_a(K - 1_a)), a running over the axes of e.support (SD42, SD72). The key is the expression e of
    cell.gradient(e, scheme=centred_gradient()), of kind Gradient; pointwise(gradient=centred_gradient()) reads it
    for each discrete field whose gradient its law reads."""
    e = term.quantity
    return CellValue(sum([(cell.shift(a, 1).value(e) - cell.shift(a, -1).value(e))
                          / (cell.shift(a, 1).center(a) - cell.shift(a, -1).center(a)) * unit(a)
                          for a in e.support.axes]))


@pops.scheme
def adjacent_mean(face, term):
    """grad_f e = (g_L + g_R) / 2, g_L and g_R the centred_gradient of e on face.left_cell and face.right_cell, so
    that the formula of the cell gradient is written once (SD72). The key is the expression e of
    face.gradient(e, scheme=adjacent_mean())."""
    e = term.quantity
    left = face.left_cell.gradient(e, scheme=centred_gradient())
    right = face.right_cell.gradient(e, scheme=centred_gradient())
    return FaceValue((left + right) / 2)


# Boundary policies and ghosts (D9); face.normal is outward on a boundary face

@pops.scheme
def ghosts(face, term, depth):
    """Boundary policy: each relation of model.on is realized by the ghost that its form (term.form, 2.6.2) gives,
    at each depth k, its data g being read at the centre of the face, where the relation holds (D9, SD78):
    a value g: 2 g - u_k, in closed form (D9); a normal flux (Robin included) or a general
    relation: the ghost that satisfies it two-point, a solve that PoPS eliminates on each point when it is affine
    (SU9, SD63); no data: u_1; a characteristic relation: u_k + 2 P_in(u_k) (g - u_k); a zero normal velocity: the
    reflected state reflect(u_k, normal=n). Where u_k > 2 g the ghost of a value changes sign; constant_ghost()
    under the handle of the relation keeps g."""
    return Ghost(_ghost_value(face, term, depth))


@pops.scheme
def weak_fluxes(face, term):
    """Boundary policy: each relation of model.on is realized by the boundary flux F(u_b, (u_b - u_1) / (dx / 2) n).n,
    F = term.flux, the total flux of the block that the relation closes (SD42), and u_b the state that its form
    gives on the face: the root of the relation for a value, a normal flux (Robin included) or a general relation;
    u_1 for no data; u_1 + P_in(u_1) (g - u_1) for a characteristic relation; (u_1 + reflect(u_1, normal=n)) / 2
    for a zero normal velocity. The inner cell receives it with the outward sign."""
    q = term.quantity
    inner = _inner(face, term, 1)
    boundary = _boundary_state(face, term)
    gradient = (boundary - inner) / (face.distance / 2) * face.normal
    flux = term.flux(states={q: boundary}, gradients={q: gradient})
    return BoundaryFlux(dot(flux, face.normal), deliver={q: -1})


@pops.scheme
def constant_ghost(face, term):
    """G_k = g at every depth, g the data of the value relation q == g under whose handle the scheme is placed,
    for a value whose ghost 2 g - u_k could change sign (SD64)."""
    term.expect(term.quantity == term.data)
    return Ghost(term.data)


@pops.scheme
def extrapolation_ghost(face, term):
    """G_k = u_1 at every depth, the first inner cell, so that no value is imposed (master's Extrapolate)."""
    return Ghost(_inner(face, term, 1))


@pops.scheme
def reflection_ghost(face, term, depth):
    """G_k = reflect(u_k, normal=n), u_k the state of the k-th inner cell: each vector part v becomes v - 2 (v.n) n,
    each tensor part T becomes N T N with N = I - 2 n n^T, and the scalar parts are copied (master's
    ReflectiveSlip)."""
    return Ghost(_reflected(face, term, depth))


def _inner(face, term, depth):
    """u_k, the value of the k-th cell inward from the face (D9)."""
    return face.inner(depth).value(term.quantity)


def _ghost_value(face, term, depth):
    """The ghost at depth k that the form of the relation gives. The ghost of a value is the closed form of D9,
    so it runs no solve; a normal-flux relation (Robin included) is affine in the face mean and the normal
    derivative, so PoPS eliminates its solve on each point (SU9, SD63); a general relation is solved as it
    stands."""
    form = term.form
    if form is forms.Value:
        return 2 * term.data - _inner(face, term, depth)
    if form is forms.NoData:
        return _inner(face, term, 1)
    if form is forms.Characteristic:
        inner = _inner(face, term, depth)
        return inner + 2 * _incoming(face, term.data - inner, inner)
    if form is forms.NormalComponent:
        return _reflected(face, term, depth)
    return _solved_ghost(face, term, depth)


def _solved_ghost(face, term, depth):
    """G_k, the root of the residual of the relation (term.law, SD42) at the face mean (G_k + u_k) / 2 and the
    normal derivative (G_k - u_k) / ((2 k - 1) dx), dx = face.distance: u_k - (2 k - 1) dx h / nu for
    -nu dq/dn == h, and the Robin ghost for a Robin relation."""
    q = term.quantity
    inner = _inner(face, term, depth)
    distance = (2 * depth - 1) * face.distance
    ghost = pops.unknown(q.space)
    residual = term.law(states={q: (ghost + inner) / 2}, gradients={q: (ghost - inner) / distance * face.normal})
    return pops.solve(residual == 0, ghost)


def _boundary_state(face, term):
    """u_b, the state on the face that the form of the relation gives (weak_fluxes)."""
    q = term.quantity
    inner = _inner(face, term, 1)
    form = term.form
    if form is forms.Value:
        return term.data
    if form is forms.NoData:
        return inner
    if form is forms.Characteristic:
        return inner + _incoming(face, term.data - inner, inner)
    if form is forms.NormalComponent:
        return (inner + _reflected(face, term, 1)) / 2
    boundary = pops.unknown(q.space)
    gradient = (boundary - inner) / (face.distance / 2) * face.normal
    return pops.solve(term.law(states={q: boundary}, gradients={q: gradient}) == 0, boundary)


def _incoming(face, difference, state):
    """P_in(u) d, the part of d on the incoming characteristics: incoming_projector of pops.math (SD43) at the
    outward normal of the face, read at the discrete state u of the quantity of the relation, which stands for U.
    Where the scheme of the flux states its normal speed s with speed= (upwind), P_in is 1 where s < 0 and 0
    elsewhere, so that the speed is stated once; elsewhere R9 gives P_in at u (SD75)."""
    return dot(incoming_projector(state, normal=face.normal), difference)


def _reflected(face, term, depth):
    """reflect(u_k, normal=n), the k-th inner state reflected across the face (pops.math, SD43)."""
    return reflect(_inner(face, term, depth), normal=face.normal)


# Transfers between levels (4.6)

@pops.scheme
def linear_prolongation(cell, term, limiter):
    """u_k = u_P + sum_a limiter(u_P - u_(P-1_a), u_(P+1_a) - u_P) (x_a(k) - x_a(P)) / h_a(P) on each child k of the
    parent P, h_a(P) the width of P along a. The children lie symmetrically about the centre of P, so their mean
    is u_P."""
    q = term.quantity
    parent = cell.parent
    centre = parent.value(q)
    slopes = [limiter(centre - parent.shift(a, -1).value(q), parent.shift(a, 1).value(q) - centre)
              * (cell.center(a) - parent.center(a)) / parent.width(a) for a in q.support.axes]
    return Transfer(centre + sum(slopes))


@pops.scheme
def mean_of_children(cell, term):
    """u_P = sum_k |k| u_k / |P| over the children k of P: the restriction of a CellAverage (4.6), which the
    representation derives when pops.Regrid or subcycle states no restriction=."""
    q = term.quantity

    def content(child):
        return child.measure * child.value(q)

    return Transfer(sum_children(cell, content) / cell.measure)


# Moving faces and kinetic definitions

@pops.scheme
def relative_flux(face, term, scheme):
    """F*_rel = F* - w u_s on a face that moves with the normal velocity w = face.velocity: F* = face.flux(term,
    scheme=scheme), the face flux of `scheme` for the key, and u_s the state of the cell that the face sweeps, u_R
    where w > 0 and u_L elsewhere (D18, SD42). The flux and the volume update use the same swept faces, so a
    constant state with no source keeps its value (D21)."""
    q = term.quantity
    w = face.velocity
    swept = where(w > 0, face.right(q), face.left(q))
    return FaceFlux(face.flux(term, scheme=scheme) - w * swept)


@pops.scheme
def discrete_maxwellian(cell, term, tolerance=None):
    """M_h(v_j) = exp(a + sum_k b_k v_jk + c |v_j|^2) on each fibre, (a, b, c) the root of the discrete moment
    equations sum_j |V_j| (M_h(v_j) - f_j) phi(v_j) = 0 for phi = 1, v_1, ..., v_d and |v|^2 / 2, the sums running
    over the velocity cells of the fibre (moment_h of 4.3): the Maxwellian whose discrete moments equal those
    of f. The key is a definition whose value is maxwellian(f, over=velocity). The unknowns a, b and c are read at
    a cell with cell.value, like quantities, and the solve has the scope that these readings give (SD42, SV1).
    `tolerance` is the request of the solve, a pops.Tolerance; without it, the solve takes the request of the
    step."""
    f = term.quantities[0]
    factors = f.support.factors                 # [space, velocity] for pops.product(space, velocity)
    space = factors[0]
    velocity = factors[1]
    a = pops.unknown(pops.Scalar(space))
    b = [pops.unknown(pops.Scalar(space)) for velocity_axis in velocity.axes]
    c = pops.unknown(pops.Scalar(space))

    def ansatz(at, alpha, beta, gamma):
        """exp(alpha + sum_k beta_k v_k + gamma |v|^2) at the velocity of the cell `at`."""
        v = [at.center(velocity_axis) for velocity_axis in velocity.axes]
        drift = sum([at.value(beta[k]) * v[k] for k in range(len(v))])
        return exp(at.value(alpha) + drift + at.value(gamma) * sum([vk * vk for vk in v]))

    def moment_equation(weight):
        """sum over the fibre of |K| (M_h - f) weight == 0; |K| = |X| |V_j|, and |X| is the same on the fibre."""
        def summand(other):
            return other.measure * (ansatz(other, a, b, c) - other.value(f)) * weight(other)
        return sum_fibre(cell, summand) == 0

    def one(other):
        return 1

    def coordinate(velocity_axis):
        def weight(other):
            return other.center(velocity_axis)
        return weight

    def half_square(other):
        return sum([other.center(velocity_axis) ** 2 for velocity_axis in velocity.axes]) / 2

    weights = [one] + [coordinate(velocity_axis) for velocity_axis in velocity.axes] + [half_square]
    solution = pops.solve([moment_equation(weight) for weight in weights], [a] + b + [c], tolerance=tolerance)
    return CellValue(ansatz(cell, solution[a], [solution[bk] for bk in b], solution[c]))
