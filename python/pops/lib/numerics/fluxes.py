"""The scheme words of a flux and of a diffusive flux, written with ``@pops.scheme`` (spec 0.3,
2.6.9; A4, SU5, CF4). N3 lot 1 (ROM-87).

``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()`` are what ``pops.Numerics(schemes={F:
s})`` takes, each called with its options. Each is a scheme of the surface, the same object a case
writes by hand: a copy of a body written in a case elaborates into the same program, the same
semantic identity and the same C++ (CF4). The bodies are those of the gallery
(``examples/surface/gallery/lib_schemes.py``, D8), in the arithmetic of the b6 bodies that they
replace (``examples/api_v02/b6_numerics.py``, the parity oracles, D5): ``0.5 * x``, never
``x / 2``, and ``x * (1.0 / y)`` for the quotient of hll, so that the generated C++ is the one of
b6 operation by operation (P1).

* ``rusanov()``: the mean of the two normal fluxes less s (u_R - u_L) / 2, s the larger spectral
  radius of the two sides (R9), recorded as its speed.
* ``upwind()``: without ``speed``, the flux must be F(1) u, linear in its scalar quantity, which
  ``term.expect`` checks, naming ``rusanov()`` and ``hll()`` in its refusal (D9); the face is then
  the one of ``rusanov()``, whose spectral radius is |F(1)·n| on both sides: the upwind flux of
  F(1) u, with master's arithmetic (D1). With ``speed=f``, s = f(face) is the signed normal speed
  and nothing is checked: s u_L where s > 0, s u_R elsewhere, |s| recorded (SD44).
* ``hll()``: master's HLL with the slowest and the fastest wave speeds united over the two traces.
* ``two_point()``: F(u_f, g_f)·n with the face mean u_f and the two-point normal derivative g_f of
  each gradient variable (SD51); b6 realizes it by a ``StencilFlux`` whose frequency it derives
  (D7, D10).
"""
import pops.surface as pops
from pops.surface.math import dot, largest, smallest, where
from pops.surface.discrete import FaceFlux, spectral_radius, slowest_speed, fastest_speed


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
    """The stated speed at the face: `speed` itself, or the entry of a table {factor: f} whose
    factor holds face.axis, the handle of the axis that this run of the scheme covers (SD42),
    compared with `is` to the axes of each factor; None where no speed is stated there, so that
    the speed is derived."""
    if not isinstance(speed, dict):
        return _function_of_face(speed)
    for factor in speed:
        axes = getattr(factor, "axes", None)
        if axes is None:
            raise TypeError("upwind(speed={factor: f}) keys its table by the factors of a product "
                            "support; %r is not one" % (factor,))
        for axis in axes:
            if axis is face.axis:
                return _function_of_face(speed[factor])
    return None


def _function_of_face(speed):
    """`speed` itself, a function of face passed uncalled (SU5), or None."""
    if speed is not None and (not callable(speed) or isinstance(speed, type)):
        raise TypeError("upwind(speed=f) takes a function of face, def f(face): ..., got %r"
                        % (speed,))
    return speed


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


# The flux of a diffusive term

@pops.scheme
def two_point(face, term):
    """F* = F(u_f, g_f).n, with u_f = (u_L + u_R) / 2 the face mean of each quantity and
    g_f = (w_R - w_L) / face.distance n the two-point normal gradient of each w that the law
    differentiates, a gradient variable or a first-order input (E2, SD51)."""
    states = {r: face.average(r) for r in term.quantities}
    gradients = {w: face.normal_gradient(w) * face.normal for w in _differentiated(term)}
    return FaceFlux(dot(term.law(states=states, gradients=gradients), face.normal))


def _differentiated(term):
    """Every w that gradients= must give: the gradient variables of the key, then its first-order
    inputs, the fields whose gradient the law reads as a first-order transport input, in the
    order of the law (E2, SD51)."""
    return term.gradient_variables + term.first_order_inputs


__all__ = ["hll", "rusanov", "two_point", "upwind"]
