"""Immutable, canonical residual-system descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from pops.ir.literals import ScalarLiteral, scalar_literal
from pops.time.residual_common import (
    CanonicalDescriptor, coverage_errors, residual_name as _name, residual_names as _names,
)
from pops.time.residual_reports import ResidualReport, SupportReport, SupportStatus
from pops.time.value_metadata import positive_scalar_literal


@dataclass(frozen=True, slots=True)
class ProductSpace(CanonicalDescriptor):
    """A named product space whose components are fully qualified semantic ids."""

    kind: ClassVar[str] = "product_space"
    name: str
    components: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "ProductSpace.name"))
        object.__setattr__(self, "components", _names(
            self.components, "ProductSpace.components", nonempty=True))
        for component in self.components:
            if "::" not in component:
                raise ValueError(
                    "ProductSpace components must be qualified ids containing '::' (got %r)"
                    % component)


@dataclass(frozen=True, slots=True)
class UnknownSpace(ProductSpace):
    kind: ClassVar[str] = "unknown_space"


@dataclass(frozen=True, slots=True)
class EquationSpace(ProductSpace):
    kind: ClassVar[str] = "equation_space"


@dataclass(frozen=True, slots=True)
class IdentityTerm(CanonicalDescriptor):
    kind: ClassVar[str] = "identity"
    equation: str
    unknown: str
    coefficient: ScalarLiteral | Any = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "equation", _name(self.equation, "IdentityTerm.equation"))
        object.__setattr__(self, "unknown", _name(self.unknown, "IdentityTerm.unknown"))
        object.__setattr__(self, "coefficient", scalar_literal(self.coefficient))


@dataclass(frozen=True, slots=True)
class MassTerm(CanonicalDescriptor):
    kind: ClassVar[str] = "mass"
    equation: str
    unknown: str
    operator: str

    def __post_init__(self) -> None:
        for attr in ("equation", "unknown", "operator"):
            object.__setattr__(self, attr, _name(getattr(self, attr), "MassTerm.%s" % attr))


@dataclass(frozen=True, slots=True)
class AlgebraicTerm(CanonicalDescriptor):
    kind: ClassVar[str] = "algebraic"
    equation: str
    operator: str
    unknowns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "equation", _name(self.equation, "AlgebraicTerm.equation"))
        object.__setattr__(self, "operator", _name(self.operator, "AlgebraicTerm.operator"))
        object.__setattr__(self, "unknowns", _names(
            self.unknowns, "AlgebraicTerm.unknowns", nonempty=True))


@dataclass(frozen=True, slots=True)
class Dt(CanonicalDescriptor):
    kind: ClassVar[str] = "dt"
    symbol: str = "dt"
    positive: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _name(self.symbol, "Dt.symbol"))
        if not isinstance(self.positive, bool):
            raise TypeError("Dt.positive must be bool")


@dataclass(frozen=True, slots=True)
class Coupling(CanonicalDescriptor):
    kind: ClassVar[str] = "coupling"
    source: str
    target: str
    operator: str

    def __post_init__(self) -> None:
        for attr in ("source", "target", "operator"):
            object.__setattr__(self, attr, _name(getattr(self, attr), "Coupling.%s" % attr))


@dataclass(frozen=True, slots=True)
class FieldDependency(CanonicalDescriptor):
    kind: ClassVar[str] = "field_dependency"
    field: str
    equations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _name(self.field, "FieldDependency.field"))
        object.__setattr__(self, "equations", _names(
            self.equations, "FieldDependency.equations", nonempty=True))


@dataclass(frozen=True, slots=True)
class Boundary(CanonicalDescriptor):
    kind: ClassVar[str] = "boundary"
    equation: str
    boundary: str
    condition: str

    def __post_init__(self) -> None:
        for attr in ("equation", "boundary", "condition"):
            object.__setattr__(self, attr, _name(getattr(self, attr), "Boundary.%s" % attr))


@dataclass(frozen=True, slots=True)
class Constraint(CanonicalDescriptor):
    kind: ClassVar[str] = "constraint"
    name: str
    equations: tuple[str, ...]
    operator: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "Constraint.name"))
        object.__setattr__(self, "equations", _names(
            self.equations, "Constraint.equations", nonempty=True))
        object.__setattr__(self, "operator", _name(self.operator, "Constraint.operator"))


@dataclass(frozen=True, slots=True)
class ExactJacobian(CanonicalDescriptor):
    kind: ClassVar[str] = "exact_jacobian"
    operator: str
    domain: UnknownSpace
    codomain: EquationSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _name(self.operator, "ExactJacobian.operator"))
        if not isinstance(self.domain, UnknownSpace) or not isinstance(self.codomain, EquationSpace):
            raise TypeError("ExactJacobian requires domain UnknownSpace and codomain EquationSpace")


@dataclass(frozen=True, slots=True)
class AutomaticJVP(CanonicalDescriptor):
    kind: ClassVar[str] = "automatic_jvp"
    engine: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine", _name(self.engine, "AutomaticJVP.engine"))


@dataclass(frozen=True, slots=True)
class FiniteDifferenceJVP(CanonicalDescriptor):
    kind: ClassVar[str] = "finite_difference_jvp"
    relative_step: ScalarLiteral | Any = 1.0e-7

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_step", positive_scalar_literal(
            self.relative_step, where="FiniteDifferenceJVP.relative_step"))


@dataclass(frozen=True, slots=True)
class ApproximateLinearization(CanonicalDescriptor):
    kind: ClassVar[str] = "approximate_linearization"
    operator: str
    approximation: str
    domain: UnknownSpace
    codomain: EquationSpace

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _name(
            self.operator, "ApproximateLinearization.operator"))
        object.__setattr__(self, "approximation", _name(
            self.approximation, "ApproximateLinearization.approximation"))
        if self.approximation.casefold() == "exact":
            raise ValueError(
                "ApproximateLinearization cannot claim exact fidelity; use ExactJacobian")
        if not isinstance(self.domain, UnknownSpace) or not isinstance(self.codomain, EquationSpace):
            raise TypeError(
                "ApproximateLinearization requires domain UnknownSpace and codomain EquationSpace")


@dataclass(frozen=True, slots=True)
class PreconditionerDomain(CanonicalDescriptor):
    """Direction of a residual preconditioner: equation vectors to unknown corrections."""

    kind: ClassVar[str] = "preconditioner_domain"
    input_space: EquationSpace
    output_space: UnknownSpace

    def __post_init__(self) -> None:
        if not isinstance(self.input_space, EquationSpace):
            raise TypeError("PreconditionerDomain.input_space must be EquationSpace")
        if not isinstance(self.output_space, UnknownSpace):
            raise TypeError("PreconditionerDomain.output_space must be UnknownSpace")


@dataclass(frozen=True, slots=True)
class PreconditionerContract(CanonicalDescriptor):
    kind: ClassVar[str] = "preconditioner_contract"
    operator: str
    domain: PreconditionerDomain

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", _name(
            self.operator, "PreconditionerContract.operator"))
        if not isinstance(self.domain, PreconditionerDomain):
            raise TypeError("PreconditionerContract.domain must be PreconditionerDomain")

    def compatibility_errors(self, residual: Any) -> tuple[str, ...]:
        if not isinstance(residual, ResidualOperator):
            raise TypeError("PreconditionerContract expects a ResidualOperator")
        errors = []
        if self.domain.input_space.components != residual.equation_space.components:
            errors.append(
                "preconditioner input must cover equation_space exactly and in block order")
        if self.domain.output_space.components != residual.unknown_space.components:
            errors.append(
                "preconditioner output must cover unknown_space exactly and in block order")
        return tuple(errors)

    def validate(self, residual: Any) -> bool:
        errors = self.compatibility_errors(residual)
        if errors:
            raise ValueError("invalid preconditioner contract: " + "; ".join(errors))
        return True

    def validate_for(self, residual: Any) -> bool:
        """Validate this contract for a residual (explicit Program integration spelling)."""
        return self.validate(residual)


class LinearizationFidelity(str, Enum):
    EXACT = "exact"
    AUTOMATIC = "automatic"
    FINITE_DIFFERENCE = "finite_difference"
    APPROXIMATE = "approximate"


def linearization_fidelity(value: Any) -> LinearizationFidelity:
    if isinstance(value, ExactJacobian):
        return LinearizationFidelity.EXACT
    if isinstance(value, AutomaticJVP):
        return LinearizationFidelity.AUTOMATIC
    if isinstance(value, FiniteDifferenceJVP):
        return LinearizationFidelity.FINITE_DIFFERENCE
    if isinstance(value, ApproximateLinearization):
        return LinearizationFidelity.APPROXIMATE
    raise TypeError("unsupported residual linearization %s" % type(value).__name__)


@dataclass(frozen=True, slots=True)
class ConsistentInitialization(CanonicalDescriptor):
    kind: ClassVar[str] = "consistent_initialization"
    solver: str
    tolerance: ScalarLiteral | Any
    max_iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "solver", _name(self.solver, "ConsistentInitialization.solver"))
        object.__setattr__(self, "tolerance", positive_scalar_literal(
            self.tolerance, where="ConsistentInitialization.tolerance"))
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) \
                or self.max_iterations <= 0:
            raise ValueError("ConsistentInitialization.max_iterations must be a positive int")


@dataclass(frozen=True, slots=True)
class RequireConsistentInitialState(CanonicalDescriptor):
    kind: ClassVar[str] = "require_consistent_initial_state"


@dataclass(frozen=True, slots=True)
class PostStepValidation(CanonicalDescriptor):
    kind: ClassVar[str] = "post_step_validation"
    tolerance: ScalarLiteral | Any
    reject_on_failure: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance", positive_scalar_literal(
            self.tolerance, where="PostStepValidation.tolerance"))
        if not isinstance(self.reject_on_failure, bool):
            raise TypeError("PostStepValidation.reject_on_failure must be bool")


@dataclass(frozen=True, slots=True)
class NoPostStepValidation(CanonicalDescriptor):
    kind: ClassVar[str] = "no_post_step_validation"


@dataclass(frozen=True, slots=True)
class Index1DAE(CanonicalDescriptor):
    """Declared index-1 partition; this is a claim checked by validation, not inferred."""

    kind: ClassVar[str] = "index_1_dae"
    differential_unknowns: tuple[str, ...]
    algebraic_unknowns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "differential_unknowns", _names(
            self.differential_unknowns, "Index1DAE.differential_unknowns", nonempty=True))
        object.__setattr__(self, "algebraic_unknowns", _names(
            self.algebraic_unknowns, "Index1DAE.algebraic_unknowns", nonempty=True))
        if set(self.differential_unknowns) & set(self.algebraic_unknowns):
            raise ValueError("Index1DAE differential and algebraic partitions overlap")


_TERM_TYPES = (IdentityTerm, MassTerm, AlgebraicTerm)
_LINEARIZATION_TYPES = (ExactJacobian, AutomaticJVP, FiniteDifferenceJVP,
                        ApproximateLinearization)
_INIT_TYPES = (ConsistentInitialization, RequireConsistentInitialState)
_POST_TYPES = (PostStepValidation, NoPostStepValidation)


@dataclass(frozen=True, slots=True)
class ResidualOperator(CanonicalDescriptor):
    """Complete canonical residual contract, independent of any backend implementation."""

    kind: ClassVar[str] = "residual_operator"
    name: str
    unknown_space: UnknownSpace
    equation_space: EquationSpace
    terms: tuple[CanonicalDescriptor, ...]
    dt: Dt
    linearization: CanonicalDescriptor
    preconditioner: PreconditionerContract | None = None
    couplings: tuple[Coupling, ...] = ()
    field_dependencies: tuple[FieldDependency, ...] = ()
    boundaries: tuple[Boundary, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    dae: Index1DAE | None = None
    consistent_initialization: CanonicalDescriptor | None = None
    post_step: CanonicalDescriptor = NoPostStepValidation()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "ResidualOperator.name"))
        if not isinstance(self.unknown_space, UnknownSpace):
            raise TypeError("ResidualOperator.unknown_space must be UnknownSpace")
        if not isinstance(self.equation_space, EquationSpace):
            raise TypeError("ResidualOperator.equation_space must be EquationSpace")
        for attr, expected in (("terms", _TERM_TYPES), ("couplings", (Coupling,)),
                               ("field_dependencies", (FieldDependency,)),
                               ("boundaries", (Boundary,)), ("constraints", (Constraint,))):
            values = tuple(getattr(self, attr))
            if any(not isinstance(value, expected) for value in values):
                raise TypeError("ResidualOperator.%s contains an unsupported descriptor" % attr)
            object.__setattr__(self, attr, values)
        if not self.terms:
            raise ValueError("ResidualOperator.terms must not be empty")
        if not isinstance(self.dt, Dt):
            raise TypeError("ResidualOperator.dt must be Dt")
        if not isinstance(self.linearization, _LINEARIZATION_TYPES):
            raise TypeError("ResidualOperator.linearization must be a typed linearization")
        if self.preconditioner is not None and not isinstance(
                self.preconditioner, PreconditionerContract):
            raise TypeError("ResidualOperator.preconditioner must be PreconditionerContract or None")
        if self.dae is not None and not isinstance(self.dae, Index1DAE):
            raise TypeError("ResidualOperator.dae must be Index1DAE or None")
        if self.consistent_initialization is not None and not isinstance(
                self.consistent_initialization, _INIT_TYPES):
            raise TypeError("ResidualOperator.consistent_initialization has unsupported policy")
        if not isinstance(self.post_step, _POST_TYPES):
            raise TypeError("ResidualOperator.post_step has unsupported policy")

    def report(self) -> ResidualReport:
        errors: list[str] = []
        unknowns = set(self.unknown_space.components)
        equations = set(self.equation_space.components)
        if len(unknowns) != len(equations):
            errors.append("unknown and equation product spaces must have equal arity")
        if isinstance(self.linearization, (ExactJacobian, ApproximateLinearization)):
            if self.linearization.domain.components != self.unknown_space.components:
                errors.append(
                    "linearization domain must cover unknown_space exactly and in block order")
            if self.linearization.codomain.components != self.equation_space.components:
                errors.append(
                    "linearization codomain must cover equation_space exactly and in block order")
        if self.preconditioner is not None:
            errors.extend(self.preconditioner.compatibility_errors(self))
        for term in self.terms:
            if term.equation not in equations:
                errors.append("term references equation outside equation_space: %s" % term.equation)
            for unknown in ((term.unknown,) if isinstance(term, (IdentityTerm, MassTerm))
                            else term.unknowns):
                if unknown not in unknowns:
                    errors.append("term references unknown outside unknown_space: %s" % unknown)
        covered_equations = {term.equation for term in self.terms}
        covered_unknowns = {
            unknown for term in self.terms
            for unknown in ((term.unknown,) if isinstance(term, (IdentityTerm, MassTerm))
                            else term.unknowns)
        }
        errors.extend(coverage_errors(equations, covered_equations, "equation"))
        errors.extend(coverage_errors(unknowns, covered_unknowns, "unknown"))
        for coupling in self.couplings:
            if coupling.source not in unknowns or coupling.target not in equations:
                errors.append("coupling endpoints are outside the declared product spaces")
        for dependency in self.field_dependencies:
            if any(equation not in equations for equation in dependency.equations):
                errors.append("field dependency references equation outside equation_space")
        for boundary in self.boundaries:
            if boundary.equation not in equations:
                errors.append("boundary references equation outside equation_space")
        for constraint in self.constraints:
            if any(equation not in equations for equation in constraint.equations):
                errors.append("constraint references equation outside equation_space")
        mass_unknowns = {term.unknown for term in self.terms
                         if isinstance(term, (IdentityTerm, MassTerm))}
        if self.dae is not None:
            partition = set(self.dae.differential_unknowns) | set(self.dae.algebraic_unknowns)
            if partition != unknowns:
                errors.append("Index1DAE partition must cover unknown_space exactly")
            if not set(self.dae.differential_unknowns) <= mass_unknowns:
                errors.append("every Index1DAE differential unknown needs an identity/mass term")
            if set(self.dae.algebraic_unknowns) & mass_unknowns:
                errors.append("Index1DAE algebraic unknowns cannot have identity/mass terms")
            if self.consistent_initialization is None:
                errors.append("Index1DAE requires an explicit consistent-initialization policy")
            if isinstance(self.post_step, NoPostStepValidation):
                errors.append("Index1DAE requires an explicit post-step validation policy")
        return ResidualReport(
            valid=not errors,
            errors=tuple(dict.fromkeys(errors)),
            facts={"n_unknowns": len(unknowns), "n_equations": len(equations),
                   "n_terms": len(self.terms),
                   "n_identity_terms": sum(isinstance(term, IdentityTerm) for term in self.terms),
                   "n_mass_terms": sum(isinstance(term, MassTerm) for term in self.terms),
                   "n_algebraic_terms": sum(isinstance(term, AlgebraicTerm) for term in self.terms),
                   "n_dt": 1, "n_couplings": len(self.couplings),
                   "n_boundaries": len(self.boundaries),
                   "n_fields": len({dependency.field for dependency in self.field_dependencies}),
                   "n_field_dependencies": len(self.field_dependencies),
                   "n_constraints": len(self.constraints),
                   "has_preconditioner": self.preconditioner is not None,
                   "dae_index": 1 if self.dae else None,
                   "linearization_fidelity": linearization_fidelity(self.linearization).value})

    def validate(self) -> bool:
        report = self.report()
        if not report.valid:
            raise ValueError("invalid residual operator: " + "; ".join(report.errors))
        return True

    def canonical_identity(self) -> dict[str, Any]:
        """Return validated canonical data suitable for hashing/manifests."""
        self.validate()
        return self.to_data()

    def support(self, backend: Any = None) -> SupportReport:
        """Return fail-closed support evidence; no backend support is implied by this model."""
        validation = self.report()
        if not validation.valid:
            return SupportReport(SupportStatus.UNAVAILABLE, None,
                                 reasons=("residual contract is invalid",))
        if backend is None:
            return SupportReport(SupportStatus.UNKNOWN, None, missing=("residual_backend",))
        backend_name = getattr(backend, "name", None)
        probe = getattr(backend, "residual_support", None)
        if not isinstance(backend_name, str) or not callable(probe):
            return SupportReport(SupportStatus.UNKNOWN, None,
                                 missing=("typed residual support probe",))
        result = probe(self)
        if not isinstance(result, SupportReport):
            return SupportReport(SupportStatus.UNKNOWN, backend_name,
                                 reasons=("backend returned no structured SupportReport",))
        if result.supported and result.backend != backend_name:
            return SupportReport(SupportStatus.UNKNOWN, backend_name,
                                 reasons=("support evidence names a different backend",))
        return result


__all__ = [
    "AlgebraicTerm", "ApproximateLinearization", "AutomaticJVP", "Boundary",
    "ConsistentInitialization", "Constraint", "Coupling", "Dt",
    "EquationSpace", "ExactJacobian", "FieldDependency", "FiniteDifferenceJVP", "IdentityTerm",
    "Index1DAE", "LinearizationFidelity", "MassTerm",
    "NoPostStepValidation", "PostStepValidation", "PreconditionerContract",
    "PreconditionerDomain", "ProductSpace", "RequireConsistentInitialState",
    "ResidualOperator", "ResidualReport", "SupportReport", "SupportStatus", "UnknownSpace",
    "linearization_fidelity",
]
