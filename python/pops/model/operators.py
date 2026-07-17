"""Operator-valued types and the named, typed :class:`Operator` (Spec 2).

Defines the ten :data:`OPERATOR_KINDS`, the operator-valued types
``LocalLinearOperator`` / ``MatrixFreeOperator`` (a ``Space -> Space`` map usable
by a local-linear or Krylov solve), and the named :class:`Operator` that pairs a
kind with a :class:`pops.model.signatures.Signature`. Carries no numerics; the body
lives in the model / codegen.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from pops.provenance import ProvenanceRecord

from .signatures import Signature
from .spaces import FieldSpace, RateSpace, Space, StateSpace, _as_signature_inputs
from ._module_freeze import deep_freeze_model_value

# The operator kinds of Spec 2. A kind is metadata only; the Signature carries
# the actual type contract that Program validation checks.
OPERATOR_KINDS = (
    "local_rate",
    "local_source",
    "local_linear_operator",
    "field_operator",
    "grid_operator",
    "projection",
    "diagnostic",
    "matrix_free_operator",
    "local_nonlinear_residual",
    "global_residual",
    "coupled_rate",
)

# The MATHEMATICAL families a user reads (ADC-559): every operator kind folds into one of the
# seven families the model exposes -- rate, field_solve, local_linear_map, matrix_free_map,
# projection, diagnostic, coupled_rate. The family is a READABLE label for `OperatorHandle.category`
# / `OperatorHandle.inspect()`; the kind stays the codegen-facing selector. A kind is never absent
# here (a KeyError would be a bug), so `operator_family` is total over OPERATOR_KINDS.
OPERATOR_FAMILIES = {
    "local_rate": "rate",
    "grid_operator": "rate",
    "local_source": "rate",
    "coupled_rate": "coupled_rate",
    "field_operator": "field_solve",
    "local_linear_operator": "local_linear_map",
    "matrix_free_operator": "matrix_free_map",
    "projection": "projection",
    "diagnostic": "diagnostic",
    "local_nonlinear_residual": "residual",
    "global_residual": "residual",
}


def operator_family(kind: Any) -> str:
    """The mathematical family of an operator ``kind`` (ADC-559): a readable label
    (``rate`` / ``field_solve`` / ``local_linear_map`` / ``matrix_free_map`` / ``projection`` /
    ``coupled_rate`` / ``diagnostic`` / ``residual``) for introspection. An unknown kind maps to
    ``"other"`` rather than raising, so introspection never fails on a foreign kind."""
    return OPERATOR_FAMILIES.get(kind, "other")


# The documented axes of an operator's ``requirements`` dict (ADC-528). Requirements are what an
# operator NEEDS from the runtime context, declared by the operator's author -- NEVER inferred from
# the operator's name. The axes are: ``ghosts`` (halo depth an int), ``fields`` (named elliptic
# fields solved before the call), ``params`` (named model parameters read), ``aux`` (named aux
# channels read), ``solvers`` (named local/global solvers needed), ``layout`` (a required storage
# layout, e.g. "cell"), ``backend`` (a required backend capability, e.g. "device"). The set is a
# documented VOCABULARY, not a hard schema: an author may still record an extra key, but
# ``Module.operator_requirements`` warns on one outside this set so a typo surfaces early.
OPERATOR_REQUIREMENT_KEYS = frozenset({
    "ghosts", "fields", "params", "aux", "solvers", "layout", "backend",
})


class LocalLinearOperator:
    """Operator-valued type ``State -> State`` (an ``L`` such that ``L: U -> U``).

    Domain and range retain their complete immutable :class:`Space` descriptors.
    A space name is presentation metadata, never evidence of type compatibility.
    Used to type ``linear_source`` operators and to check
    ``Program.solve(LocalLinear(I - dt*L, rhs), solver=DenseLU())``.
    """

    __pops_ir_immutable__ = True

    domain: Space
    range: Space

    def __init__(self, domain: Any, range_: Any) -> None:
        if not isinstance(domain, Space) or not isinstance(range_, Space):
            raise TypeError(
                "LocalLinearOperator domain and range must be typed Space descriptors; "
                "got %r -> %r" % (domain, range_))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "range", range_)

    @property
    def domain_name(self) -> str:
        """Readable domain identifier; not part of compatibility checks on its own."""
        return self.domain.name

    @property
    def range_name(self) -> str:
        """Readable range name; not part of compatibility checks on its own."""
        return self.range.name

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LocalLinearOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LocalLinearOperator is immutable")

    def _key(self) -> Any:
        return ("local_linear_operator", self.domain, self.range)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, LocalLinearOperator)
                and self._key() == other._key())

    def __hash__(self) -> int:
        return hash(self._key())

    def __rrshift__(self, inputs: Any) -> Any:
        """``(fields,) >> LocalLinearOperator(U, U)`` signature sugar."""
        return Signature(_as_signature_inputs(inputs), self)

    def __repr__(self) -> str:
        return "LocalLinearOperator(%r, %r)" % (self.domain, self.range)

    def to_data(self) -> dict[str, Any]:
        return {"kind": "local_linear_operator", "domain": self.domain.to_data(),
                "range": self.range.to_data()}


class MatrixFreeOperator:
    """Operator-valued type ``VectorSpace -> VectorSpace`` usable by a Krylov solve
    (``solve_linear``). Domain and range retain their complete typed spaces."""

    __pops_ir_immutable__ = True

    domain: Space
    range: Space

    def __init__(self, domain: Any, range_: Any) -> None:
        if not isinstance(domain, Space) or not isinstance(range_, Space):
            raise TypeError(
                "MatrixFreeOperator domain and range must be typed Space descriptors; "
                "got %r -> %r" % (domain, range_))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "range", range_)

    @property
    def domain_name(self) -> str:
        """Readable domain identifier; not part of compatibility checks on its own."""
        return self.domain.name

    @property
    def range_name(self) -> str:
        """Readable range name; not part of compatibility checks on its own."""
        return self.range.name

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("MatrixFreeOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MatrixFreeOperator is immutable")

    def _key(self) -> Any:
        return ("matrix_free_operator", self.domain, self.range)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, MatrixFreeOperator)
                and self._key() == other._key())

    def __hash__(self) -> int:
        return hash(self._key())

    def __rrshift__(self, inputs: Any) -> Any:
        """``(v,) >> MatrixFreeOperator(V, V)`` signature sugar."""
        return Signature(_as_signature_inputs(inputs), self)

    def __repr__(self) -> str:
        return "MatrixFreeOperator(%r, %r)" % (self.domain, self.range)

    def to_data(self) -> dict[str, Any]:
        return {"kind": "matrix_free_operator", "domain": self.domain.to_data(),
                "range": self.range.to_data()}


class SignatureContract:
    """One declarative operator-kind signature grammar.

    The validation table below is deliberately independent of the ``Operator``
    class.  Adding a family means supplying one small protocol validator, not
    extending a central class hierarchy or a lowering ``if/elif`` chain.
    """

    __slots__ = ("description", "_validator")

    def __init__(self, description: str, validator: Any) -> None:
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "_validator", validator)

    def validate(self, signature: Signature) -> None:
        self._validator(signature)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SignatureContract is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SignatureContract is immutable")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise TypeError(message)


def _state_rate_signature(signature: Signature) -> None:
    inputs = signature.inputs
    _require(
        len(inputs) in (1, 2) and isinstance(inputs[0], StateSpace)
        and (len(inputs) == 1 or isinstance(inputs[1], FieldSpace)),
        "expected (StateSpace[, FieldSpace]) -> Rate(StateSpace)",
    )
    _require(
        isinstance(signature.output, RateSpace)
        and signature.output.base_space == inputs[0],
        "output must be Rate() of the first StateSpace input",
    )


def _field_signature(signature: Signature) -> None:
    _require(
        bool(signature.inputs)
        and all(isinstance(item, StateSpace) for item in signature.inputs),
        "expected one or more StateSpace inputs",
    )
    _require(isinstance(signature.output, FieldSpace), "output must be a FieldSpace")


def _local_linear_signature(signature: Signature) -> None:
    _require(
        len(signature.inputs) <= 1
        and all(isinstance(item, FieldSpace) for item in signature.inputs),
        "expected () or (FieldSpace,) inputs",
    )
    output = signature.output
    _require(
        isinstance(output, LocalLinearOperator)
        and isinstance(output.domain, StateSpace)
        and isinstance(output.range, StateSpace),
        "output must be LocalLinearOperator(StateSpace, StateSpace)",
    )
    _require(output.domain == output.range, "local linear operator must be square")


def _projection_signature(signature: Signature) -> None:
    _require(
        len(signature.inputs) == 1 and isinstance(signature.inputs[0], StateSpace),
        "expected (StateSpace,) -> the same StateSpace",
    )
    _require(signature.output == signature.inputs[0],
             "projection output must equal its StateSpace input")


def _coupled_rate_signature(signature: Signature) -> None:
    from .bundles import RateBundle

    _require(
        len(signature.inputs) >= 2
        and all(isinstance(item, StateSpace) for item in signature.inputs),
        "expected at least two StateSpace inputs",
    )
    bundle = signature.output
    _require(isinstance(bundle, RateBundle) and len(bundle) > 0,
             "output must be a non-empty RateBundle")
    output_bases = [rate.base_space for _, rate in bundle.items()]
    _require(all(base in signature.inputs for base in output_bases),
             "every RateBundle output must be tangent to one input StateSpace")
    _require(len(set(output_bases)) == len(output_bases),
             "a coupled RateBundle cannot expose the same StateSpace twice")


def _matrix_free_signature(signature: Signature) -> None:
    _require(
        all(isinstance(item, (StateSpace, FieldSpace)) for item in signature.inputs),
        "matrix-free context inputs must be StateSpace or FieldSpace descriptors",
    )
    output = signature.output
    _require(isinstance(output, MatrixFreeOperator),
             "output must be a MatrixFreeOperator")
    _require(output.domain == output.range, "matrix-free operator must be square")


def _unavailable_signature(_signature: Signature) -> None:
    raise TypeError(
        "this operator kind has no public Module signature/lowering protocol yet; "
        "declare it through its dedicated Program API until a typed output space exists")


# One source of truth for the public Module/registry signature grammar.  Kinds
# with no honest public output type are explicit refusal rows, never permissive
# fall-throughs that later drop inputs or manufacture a malformed ProgramValue.
OPERATOR_SIGNATURE_CONTRACTS = MappingProxyType({
    "local_rate": SignatureContract(
        "(State[, Fields]) -> Rate(State)", _state_rate_signature),
    "local_source": SignatureContract(
        "(State[, Fields]) -> Rate(State)", _state_rate_signature),
    "grid_operator": SignatureContract(
        "(State[, Fields]) -> Rate(State)", _state_rate_signature),
    "field_operator": SignatureContract(
        "(State, ...) -> Fields", _field_signature),
    "local_linear_operator": SignatureContract(
        "(Fields?) -> LocalLinearOperator(State, State)", _local_linear_signature),
    "projection": SignatureContract(
        "(State,) -> State", _projection_signature),
    "coupled_rate": SignatureContract(
        "(State, State, ...) -> RateBundle", _coupled_rate_signature),
    "matrix_free_operator": SignatureContract(
        "(State|Fields, ...) -> MatrixFreeOperator", _matrix_free_signature),
    "diagnostic": SignatureContract("dedicated typed scalar output required", _unavailable_signature),
    "local_nonlinear_residual": SignatureContract(
        "dedicated typed residual output required", _unavailable_signature),
    "global_residual": SignatureContract(
        "dedicated typed residual output required", _unavailable_signature),
})
assert set(OPERATOR_SIGNATURE_CONTRACTS) == set(OPERATOR_KINDS)


def validate_operator_signature(kind: Any, signature: Any, *, operator_name: Any = None) -> None:
    """Validate one signature against the declarative grammar for ``kind``."""
    if kind not in OPERATOR_KINDS:
        raise ValueError("operator %r: unknown kind %r (expected one of %s)"
                         % (operator_name, kind, ", ".join(OPERATOR_KINDS)))
    if not isinstance(signature, Signature):
        raise TypeError("operator %r: signature must be a Signature" % (operator_name,))
    contract = OPERATOR_SIGNATURE_CONTRACTS[kind]
    try:
        contract.validate(signature)
    except TypeError as exc:
        raise TypeError(
            "operator %r (%s) has incompatible signature %r; contract is %s: %s"
            % (operator_name, kind, signature, contract.description, exc)) from None


class Operator:
    """A named, typed operator: ``name``, ``kind`` (one of :data:`OPERATOR_KINDS`),
    ``signature``, plus ``capabilities`` and ``requirements`` dicts and a ``source``
    tag naming the API that created it (for debug / introspection). Carries no
    numerics; the body lives in the model / codegen."""

    def __init__(self, name: Any, kind: Any, signature: Any, capabilities: Any = None,
                 requirements: Any = None, source: Any = None, lowering: Any = None,
                 body: Any = None) -> None:
        self._frozen = False
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string")
        validate_operator_signature(kind, signature, operator_name=name)
        self.name = name
        self.kind = kind
        self.signature = signature
        self.capabilities = dict(capabilities) if capabilities else {}
        self.requirements = dict(requirements) if requirements else {}
        if source is not None and not isinstance(source, ProvenanceRecord):
            raise TypeError("Operator.source must be a ProvenanceRecord or None before registration")
        self.source = source
        # Codegen hint consumed by the lowering of a typed P.call (e.g. a composite
        # rate operator carries {"flux", "sources", "fluxes"}); empty for primitives.
        self.lowering = dict(lowering) if lowering else {}
        # OPTIONAL body: the callable / expression that builds the operator IR when the
        # operator is declared via Module.operator; None for a derived dsl operator.
        self.body = body

    @property
    def frozen(self) -> bool:
        return bool(getattr(self, "_frozen", False))

    def freeze(self) -> Operator:
        """Deep-freeze this registry record while detaching every stale container alias."""
        if self.frozen:
            return self
        if not isinstance(self.source, ProvenanceRecord):
            raise TypeError("Operator.source must be a ProvenanceRecord before freeze")
        for name in ("capabilities", "requirements", "lowering", "body"):
            value = getattr(self, name)
            if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                object.__setattr__(self, name, deep_freeze_model_value(value))
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "operator %r is frozen with its Module; cannot set %r after Problem.freeze(). "
                "Author a fresh Module and recompile." % (getattr(self, "name", "?"), name)
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.frozen:
            raise RuntimeError(
                "operator %r is frozen with its Module; cannot delete %r after Problem.freeze()"
                % (getattr(self, "name", "?"), name)
            )
        object.__delattr__(self, name)

    def __repr__(self) -> str:
        return "Operator(%r, kind=%r, %r)" % (
            self.name, self.kind, self.signature)


__all__ = [
    "LocalLinearOperator", "MatrixFreeOperator", "OPERATOR_FAMILIES", "OPERATOR_KINDS",
    "OPERATOR_REQUIREMENT_KEYS", "OPERATOR_SIGNATURE_CONTRACTS", "Operator",
    "SignatureContract", "operator_family", "validate_operator_signature",
]
