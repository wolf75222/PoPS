"""M07 public authoring counterexample: coordinated hydrostatic faces are missing.

The physical topographic product is retained explicitly. Execution intentionally
stops at the unsupported numerical construction; the NumPy reference is separate.
"""
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.math import ddt, div, maximum, minimum, sqrt, where
from pops.numerics import PathConservativeFiniteVolume, SymbolicPath
from pops.numerics import reconstruction, riemann, variables


def declarations(order=("h", "q", "z")):
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


if __name__ == "__main__":
    # A native M07 run cannot be authored until the generic coordinated-face
    # contract exists. Do not silently run the homogeneous or NumPy problem.
    unavailable_hydrostatic_method()
