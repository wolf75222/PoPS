"""The boundary policy ``ghosts()`` of ``pops.lib.numerics``, written with the surface (spec 0.3,
2.6.2, 2.6.9; D9, R20; lot B of N15).

``pops.Numerics(boundary=ghosts())`` realizes each relation of ``model.on`` by the ghost G_k that
its form (``term.form``) gives at each depth k, its data g read at the centre of the face: a value
g by 2 g - u_k in closed form, and no data (``outflow(q)``) by u_1. The other forms of 2.6.2 are
written as 2.6.9 states them; their reads (``incoming_projector``, ``reflect``, ``pops.solve``)
belong to later lots and refuse with ``NotYetElaborated``, and ``model.on`` refuses their relations
first in this lot.
"""
import pops.surface as pops
from pops.surface.discrete import Ghost, forms
from pops.surface.math import dot, incoming_projector, reflect


@pops.scheme
def ghosts(face, term, depth):
    """Boundary policy: each relation of model.on is realized by the ghost that its form
    (term.form, 2.6.2) gives, at each depth k, its data g being read at the centre of the face,
    where the relation holds (D9, SD78): a value g: 2 g - u_k, in closed form (D9); a normal flux
    (Robin included) or a general relation: the ghost that satisfies it two-point, a solve that
    PoPS eliminates on each point when it is affine (SU9, SD63); no data: u_1; a characteristic
    relation: u_k + 2 P_in(u_k) (g - u_k); a zero normal velocity: the reflected state
    reflect(u_k, normal=n). Where u_k > 2 g the ghost of a value changes sign; constant_ghost()
    under the handle of the relation keeps g."""
    return Ghost(_ghost_value(face, term, depth))


def _inner(face, term, depth):
    """u_k, the value of the k-th cell inward from the face (D9)."""
    return face.inner(depth).value(term.quantity)


def _ghost_value(face, term, depth):
    """The ghost at depth k that the form of the relation gives."""
    form = term.form
    if form is forms.Value:
        return 2 * term.data - _inner(face, term, depth)
    if form is forms.NoData:
        return _inner(face, term, 1)
    if form is forms.Characteristic:
        inner = _inner(face, term, depth)
        return inner + 2 * dot(incoming_projector(inner, normal=face.normal), term.data - inner)
    if form is forms.NormalComponent:
        return reflect(_inner(face, term, depth), normal=face.normal)
    return _solved_ghost(face, term, depth)


def _solved_ghost(face, term, depth):
    """G_k, the root of the residual of the relation at the face mean (G_k + u_k) / 2 and the
    normal derivative (G_k - u_k) / ((2 k - 1) dx), dx = face.distance."""
    q = term.quantity
    inner = _inner(face, term, depth)
    distance = (2 * depth - 1) * face.distance
    ghost = pops.unknown(q.space)
    residual = term.law(states={q: (ghost + inner) / 2},
                        gradients={q: (ghost - inner) / distance * face.normal})
    return pops.solve(residual == 0, ghost)


__all__ = ["ghosts"]
