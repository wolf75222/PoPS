"""pops.moments.transforms -- the binomial / standardization transform descriptors (inert).

Document the two transforms the generator performs:

* :class:`CenteredTransform` -- the binomial transform between raw moments M_pq and
  central moments C_pq (engine ``build_moment_model`` central-moments loop).
* :class:`StandardizedTransform` -- the ``sx``/``sy`` normalization between central
  moments C_pq and standardized moments S_pq (engine standardization step).

Both are inert records; the arithmetic lives in the generator and lowers to C++.
"""


class CenteredTransform:
    """The binomial transform raw <-> central: ``C_pq = sum comb(p,i)comb(q,j)(-u)^.. m_ij``.

    Inert descriptor of the engine's central-moments loop. It records the order it
    applies to; it performs no arithmetic in Python.
    """

    def __init__(self, order):
        self.order = int(order)

    def __repr__(self):
        return "CenteredTransform(order=%d)" % (self.order,)


class StandardizedTransform:
    """The ``sx``/``sy`` standardization: ``S_pq = C_pq / (sx^p sy^q)`` with ``S20 = S02 = 1``.

    Inert descriptor of the engine's standardization step (``sx = sqrt(C20)``,
    ``sy = sqrt(C02)``). It records the order; it performs no arithmetic in Python.
    """

    def __init__(self, order):
        self.order = int(order)

    def __repr__(self):
        return "StandardizedTransform(order=%d)" % (self.order,)


__all__ = ["CenteredTransform", "StandardizedTransform"]
