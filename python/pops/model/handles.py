"""Typed operator handles (Spec 5 sec.14.2.3).

An :class:`OperatorHandle` is a lightweight, typed reference to a declared operator: it carries
the operator ``name``, its ``kind`` (the operator-first kind, when the declarer supplies it), the
declared :class:`pops.model.signatures.Signature` and the mathematical ``category`` (ADC-559) -- and
nothing else: no numerics, no IR, no Program array data. A user-facing operator declarer
(``m.rate`` / ``m.field_solve`` / ``m.local_linear_map`` / ``m.source_term`` / ``m.linear_source``)
returns one so a named operator is referenced as a typed, INSPECTABLE object, NOT a bare string::

    R = m.rate("explicit_rhs", flux=True, sources=["electric"])
    R.category        # "rate"
    R.signature       # Signature((U, Fields) -> Rate(U))
    rate = P.call(R, U, fields)      # the handle is the one public P.call selector

The handle is the public :meth:`pops.time.Program.call` selector: the public ``P.call`` REQUIRES an
``OperatorHandle`` (a bare string operator name is refused). The handle resolves through the registry
lookup + lowering identical to the internal name path, so ``P.call(handle, ...)`` builds the
byte-identical IR (same ``_ir_hash``) as the internal ``P._call(name, ...)`` the lib.time macros /
lowering use. The handle holds no Program reference, so the same handle works in any Program bound to
a registry that declares its name.

This module imports only the standard library (and the sibling ``pops.model`` types lazily inside
methods) so it stays codegen-free and ``_pops``-free and keeps the ``pops.time`` import graph acyclic
(``pops.time`` already imports ``pops.model``; ``pops.model`` never imports ``pops.time``).
"""


class OperatorHandle:
    """A typed, INSPECTABLE reference to a declared operator (Spec 5 sec.14.2.3, ADC-559).

    Carries the operator ``name``, an optional ``kind`` (the operator-first kind, e.g.
    ``"local_rate"`` / ``"local_source"`` / ``"local_linear_operator"`` / ``"field_operator"``),
    the declared :class:`pops.model.signatures.Signature` (or ``None`` when the declarer did not
    supply one) and the mathematical ``category`` -- the readable family the ``kind`` folds into
    (``rate`` / ``field_solve`` / ``local_linear_map`` / ``matrix_free_map`` / ``projection`` /
    ``coupled_rate`` / ...). It is value-like: two handles compare equal iff their ``(name, kind)``
    match (the signature/category are DERIVED metadata, not part of identity), so a handle stays
    usable as a dict key or in tests. It carries no Program, no registry and no IR; the public
    ``P.call`` (and ADC-560's ``handle(...)`` facade) resolve ``handle.name`` against the Program's
    bound registry (the internal name path is the undocumented ``P._call``).
    """

    __slots__ = ("name", "kind", "signature", "category")

    def __init__(self, name, kind=None, signature=None, category=None):
        if not isinstance(name, str) or not name:
            raise ValueError("OperatorHandle: name must be a non-empty string")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "signature", signature)
        if category is None and kind is not None:
            from pops.model.operators import operator_family
            category = operator_family(kind)
        object.__setattr__(self, "category", category)

    def inspect(self):
        """A structured, inert view of the operator this handle names (ADC-559).

        Returns a plain dict ``{name, kind, category, signature}`` -- the readable identity a user
        or a report reads without touching the registry or the codegen. ``signature`` is the repr of
        the declared :class:`Signature` (``None`` when the declarer supplied none). Read-only: it
        touches no numerics, IR or Program.
        """
        return {"name": self.name, "kind": self.kind, "category": self.category,
                "signature": repr(self.signature) if self.signature is not None else None}

    def __eq__(self, other):
        return (isinstance(other, OperatorHandle)
                and self.name == other.name and self.kind == other.kind)

    def __hash__(self):
        return hash((self.name, self.kind))

    def __repr__(self):
        if self.kind is None:
            return "OperatorHandle(%r)" % (self.name,)
        return "OperatorHandle(%r, kind=%r)" % (self.name, self.kind)
