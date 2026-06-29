"""pops.linalg.operator -- typed linear-operator descriptors (Spec 5 sec.5.6).

The linear-algebra layer NAMES the operator ``A`` in ``A x = b``; it does not apply it.
:class:`LinearOperator` references an assembled / matrix-backed operator (optionally a real
native symbol), while :class:`MatrixFreeOperator` names an operator known only by its action
``x -> A x`` (no stored matrix). Both are inert descriptors: they carry the operator name and
a matrix-free capability flag, and compute nothing -- the C++ runtime applies the operator.
"""
from pops.descriptors import Descriptor


_OPERATOR_KINDS = frozenset({"scalar", "vector", "state"})


def _check_name(name, where):
    if not isinstance(name, str) or not name:
        raise TypeError("%s: name must be a non-empty string" % where)
    return str(name)


def _check_kind(kind, where):
    if kind not in _OPERATOR_KINDS:
        raise ValueError("%s: kind must be one of %s; got %r"
                         % (where, sorted(_OPERATOR_KINDS), kind))
    return kind


def _check_ncomp(domain, ncomp, where):
    if domain == "scalar":
        if ncomp not in (None, 1):
            raise ValueError("%s: scalar operators have ncomp=1; got %r" % (where, ncomp))
        return 1
    if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
        raise ValueError("%s: %s operators require ncomp as an int >= 1; got %r"
                         % (where, domain, ncomp))
    return int(ncomp)


class LinearOperator(Descriptor):
    """A typed linear operator ``A`` (the matrix in ``A x = b``).

    ``LinearOperator("laplacian", native_id="pops::DivEpsGrad")`` names the operator and,
    optionally, the native C++ symbol that materialises it. It is inert: it declares the
    operator name and that it is NOT matrix-free (an assembled / matrix-backed operator); the
    runtime applies it. Use :class:`MatrixFreeOperator` when only the action ``x -> A x`` is
    available.
    """

    category = "linear_operator"

    def __init__(self, name, native_id=None):
        self._name = _check_name(name, "LinearOperator")
        self.native_id = native_id if native_id is None else str(native_id)

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name}

    def capabilities(self):
        return {"matrix_free": False}

    def _key(self):
        return (type(self), self._name, self.native_id)

    def __eq__(self, other):
        return isinstance(other, LinearOperator) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class MatrixFreeOperator(Descriptor):
    """A matrix-free linear operator: known only by its action ``x -> A x``.

    ``MatrixFreeOperator("stencil_apply")`` names an operator that is never assembled into a
    stored matrix (the common case for a stencil / FFT / Schur action). The optional
    :meth:`apply` decorator stores an IR builder used at Program-authoring time to generate a
    C++ ``pops::ApplyFn``. The builder never runs during ``sim.step``; it only records Program
    IR, and the generated C++ runtime applies the operator.
    """

    category = "linear_operator"

    def __init__(self, name, *, domain="scalar", range_="scalar", ncomp=None):
        self._name = _check_name(name, "MatrixFreeOperator")
        self.domain = _check_kind(domain, "MatrixFreeOperator.domain")
        self.range = _check_kind(range_, "MatrixFreeOperator.range_")
        if self.domain != self.range:
            raise ValueError(
                "MatrixFreeOperator: domain and range_ must match for a Krylov solve; "
                "got domain=%r range_=%r" % (domain, range_))
        self.ncomp = _check_ncomp(self.domain, ncomp, "MatrixFreeOperator")
        self._apply_builder = None

    @property
    def name(self):
        return self._name

    @property
    def apply_builder(self):
        return self._apply_builder

    def apply(self, builder):
        """Attach the matrix-free action builder and return the decorated builder.

        The builder has the same shape as :meth:`pops.time.Program.set_apply`:
        ``builder(program, out, x) -> result``. It is an authoring callback only. It records
        an apply sub-block that codegen lowers to C++; it is never called from the runtime
        Krylov iteration.
        """
        if not callable(builder):
            raise TypeError("MatrixFreeOperator.apply expects a callable IR builder")
        if self._apply_builder is not None:
            raise ValueError("MatrixFreeOperator %r already has an apply builder" % self._name)
        self._apply_builder = builder
        return builder

    def options(self):
        return {
            "name": self._name,
            "domain": self.domain,
            "range": self.range,
            "ncomp": self.ncomp,
            "has_apply": self._apply_builder is not None,
        }

    def capabilities(self):
        return {"matrix_free": True, "domain": self.domain, "range": self.range,
                "ncomp": self.ncomp}

    def validate(self, context=None):
        if self._apply_builder is None:
            raise ValueError(
                "MatrixFreeOperator %r has no apply builder; use @A.apply before solving"
                % self._name)
        return True

    def _key(self):
        return (type(self), self._name, self.domain, self.range, self.ncomp,
                self._apply_builder is not None)

    def __eq__(self, other):
        return isinstance(other, MatrixFreeOperator) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


__all__ = ["LinearOperator", "MatrixFreeOperator"]
