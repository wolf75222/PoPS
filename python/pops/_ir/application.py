"""Immutable joint applications and projections of captured expression bodies."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .expr import Expr, RateTerm, _wrap
from .quantity import QuantityRef, expression_handle_key


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Instantiation coordinates; differing stages/iterates/samples cannot share a call."""
    stage: Any = None
    iterate: Any = None
    location: Any = None
    sampling: str = "unspecified"
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, str) or not self.sampling:
            raise TypeError("application sampling must be explicit text")
        from .symbolic import freeze_symbolic_metadata
        for name in ("stage", "iterate", "location"):
            object.__setattr__(self, name, freeze_symbolic_metadata(getattr(self, name)))

    def declaration_references(self) -> tuple[Any, ...]:
        from .expr_references import collect_reference_value
        references: list[Any] = []
        collect_reference_value((self.stage, self.iterate, self.location), references, set())
        return tuple(references)

    def resolve_references(self, resolver: Any) -> ApplicationContext:
        from .expr_references import resolve_reference_value
        stage, iterate, location = resolve_reference_value(
            (self.stage, self.iterate, self.location), resolver, {})
        return ApplicationContext(stage, iterate, location, self.sampling)


class ApplicationEvaluation(Mapping):
    """One evaluation context with a private joint-result cache.

    Create a new context for changed inputs or another stage/iterate. Caching belongs
    to this invocation, never to an application object or a simulation run.
    """
    def __init__(self, values: Mapping) -> None:
        self._values = MappingProxyType(dict(values))
        self._applications: dict[int, Any] = {}

    def __getitem__(self, key: Any) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class OperatorApplication(Expr):
    """A captured joint computation; projection does not author another operation."""
    def __init__(self, operator: Any, inputs: Any, outputs: Any, *,
                 context: ApplicationContext | None = None, effects: Any = (),
                 occurrence: int = 0) -> None:
        from pops.model.handles import OperatorHandle
        if not isinstance(operator, OperatorHandle) or operator.signature is None:
            raise TypeError("application requires a typed OperatorHandle")
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError("application requires named output expression tuples")
        normalized = {}
        for name, values in outputs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("application outputs require nonempty projection names")
            normalized[name] = tuple(_wrap(value) for value in values)
            if not normalized[name]:
                raise ValueError("application output tuples cannot be empty")
        self.operator = operator
        self.inputs = tuple(tuple(values) for values in inputs)
        self.outputs = normalized
        self.context = context or ApplicationContext()
        if not isinstance(self.context, ApplicationContext):
            raise TypeError("application context must be ApplicationContext")
        self.effects = tuple(sorted(set(effects)))
        if any(not isinstance(effect, str) or not effect for effect in self.effects):
            raise TypeError("application effects must be named obligations")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
            raise TypeError("application occurrence must be a nonnegative integer")
        self.occurrence = occurrence

    def __getitem__(self, output: Any) -> Any:
        from pops.model.handles import Handle
        if isinstance(output, Handle):
            return self.rate(output)
        return tuple(ApplicationProjection(self, output, index)
                     for index in range(len(self.outputs[output])))

    def rate(self, target: Any, *, output: str | None = None) -> RateApplicationProjection:
        """Select a whole RateSpace output by its exact evolved quantity."""
        from pops.model.handles import Handle
        from pops.model.spaces import RateSpace
        if (not isinstance(target, Handle) or target.kind != "state"
                or target.owner_path != self.operator.owner_path):
            raise ValueError("a rate projection requires a state declared by its application owner")
        result_type = self.operator.signature.output
        entries = (("value", result_type),) if isinstance(result_type, RateSpace) else (
            result_type.items() if callable(getattr(result_type, "items", None)) else ())
        matches = [(name, space) for name, space in entries
                   if isinstance(space, RateSpace) and space.base_space.name == target.local_id
                   and (not hasattr(target, "space") or target.space == space.base_space)
                   and (output is None or name == output)]
        if len(matches) != 1:
            raise ValueError(
                "a rate projection requires one exact RateSpace output for its target; "
                "use application.rate(target, output=...) if that target has multiple outputs")
        name, space = matches[0]
        if len(self.outputs[name]) != len(space.components):
            raise ValueError("rate projection output shape differs from its declared RateSpace")
        return RateApplicationProjection(self, name, target, space)

    def eval(self, env: Any) -> Any:
        if isinstance(env, ApplicationEvaluation):
            if id(self) not in env._applications:
                env._applications[id(self)] = self._evaluate(env)
            return env._applications[id(self)]
        return self._evaluate(env)

    def deps(self) -> set[str]:
        from .visitors import _dependencies
        return _dependencies(self.__pops_ir_children__())

    def _evaluate(self, env: Any) -> Any:
        return {name: tuple(expr.eval(env) for expr in values)
                for name, values in self.outputs.items()}

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return tuple(expr for values in self.outputs.values() for expr in values)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        from pops.model.spaces import _metadata_key
        return ("operator_application", expression_handle_key(self.operator),
                tuple(tuple(recurse(value) for value in values) for values in self.inputs),
                tuple((name, tuple(recurse(value) for value in values))
                      for name, values in self.outputs.items()),
                _metadata_key((self.context.stage, self.context.iterate,
                               self.context.location, self.context.sampling)), self.effects,
                self.occurrence if self.effects else None)

    def to_cpp(self) -> str:
        raise TypeError("a joint application requires an explicit resolved native realization")

    def to_data(self) -> dict[str, Any]:
        from .visitors import _dag_key_data
        return _dag_key_data((self,))


class ApplicationProjection(Expr):
    def __init__(self, application: OperatorApplication, output: str, index: int) -> None:
        if not isinstance(application, OperatorApplication):
            raise TypeError("projection requires an OperatorApplication")
        if isinstance(index, bool) or not isinstance(index, int) \
                or not 0 <= index < len(application.outputs[output]):
            raise ValueError("application projection index is outside its output shape")
        self.application = application
        self.output = output
        self.index = index

    @property
    def expression(self) -> Expr:
        return self.application.outputs[self.output][self.index]

    def eval(self, env: Any) -> Any:
        return self.application.eval(env)[self.output][self.index]

    def deps(self) -> set[str]:
        return self.application.deps()

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.application,)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("application_projection", recurse(self.application), self.output, self.index)

    def to_cpp(self) -> str:
        raise TypeError("application projections must be lowered with their joint context")


class RateApplicationProjection(RateTerm):
    """A whole rate output, usable in balance algebra and component-indexable.

    The wrapper keeps the one joint application. It never copies its expression
    body into a separately registered source, even if multiple balances select it.
    """

    kind = "projection"

    def __init__(self, application: OperatorApplication, output: str,
                 target: Any, rate_space: Any) -> None:
        from pops.model.handles import Handle
        from pops.model.spaces import RateSpace
        if (not isinstance(application, OperatorApplication) or not isinstance(target, Handle)
                or target.kind != "state" or not isinstance(rate_space, RateSpace)
                or target.owner_path != application.operator.owner_path
                or target.local_id != rate_space.base_space.name
                or (hasattr(target, "space") and target.space != rate_space.base_space)):
            raise ValueError("rate projection target and declared RateSpace must agree exactly")
        result_type = application.operator.signature.output
        declared = (result_type if isinstance(result_type, RateSpace) and output == "value"
                    else dict(result_type.items()).get(output)
                    if callable(getattr(result_type, "items", None)) else None)
        if declared != rate_space or output not in application.outputs \
                or len(application.outputs[output]) != len(rate_space.components):
            raise ValueError("rate projection must select its application's declared output")
        self.application = application
        self.output = output
        self.target = target
        self.rate_space = rate_space

    @property
    def application_identity(self) -> tuple[Any, int]:
        return self.application.operator, self.application.occurrence

    def same_projection(self, other: Any) -> bool:
        return (isinstance(other, RateApplicationProjection)
                and self.application is other.application and self.output == other.output
                and self.target == other.target)

    def __len__(self) -> int:
        return len(self.rate_space.components)

    def __iter__(self):
        return iter(tuple(ApplicationProjection(self.application, self.output, index)
                          for index in range(len(self))))

    def __getitem__(self, index: int) -> ApplicationProjection:
        return ApplicationProjection(self.application, self.output, index)

    def _rate_terms(self) -> list[Any]:
        return [("projection", self, 1)]

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.application,)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("rate_application_projection", recurse(self.application), self.output,
                expression_handle_key(self.target), self.rate_space._key())

    def eval(self, env: Any) -> Any:
        return self.application.eval(env)[self.output]

    def to_cpp(self) -> str:
        raise TypeError("joint rate projections require a resolved native interaction realization")

    def to_data(self) -> dict[str, Any]:
        from pops.model.hash_data import canonical_hash_data
        from .balance import _handle_data
        return {"kind": self.kind, "application": canonical_hash_data(self.application),
                "application_occurrence": self.application.occurrence,
                "output": self.output, "target": _handle_data(self.target),
                "rate_space": self.rate_space.to_data()}


def substitute_quantities(body: Any, bindings: Mapping) -> Any:
    """Clone captured expressions while replacing exact input quantity leaves."""
    memo: dict[int, Any] = {}

    def clone(value: Any) -> Any:
        if isinstance(value, QuantityRef):
            return bindings.get((value.handle, value.index), value)
        if isinstance(value, Expr):
            if id(value) in memo:
                return memo[id(value)]
            result = object.__new__(type(value))
            memo[id(value)] = result
            for base in reversed(type(value).__mro__):
                slots = base.__dict__.get("__slots__", ())
                for key in (slots,) if isinstance(slots, str) else slots:
                    if key not in ("__dict__", "__weakref__", "_pops_symbolic_initializing") \
                            and hasattr(value, key):
                        object.__setattr__(result, key, clone(getattr(value, key)))
            for key, child in getattr(value, "__dict__", {}).items():
                object.__setattr__(result, key, clone(child))
            object.__setattr__(result, "_pops_symbolic_initializing", False)
            return result
        if isinstance(value, Mapping):
            return MappingProxyType({key: clone(child) for key, child in value.items()})
        if isinstance(value, (tuple, list)):
            return tuple(clone(child) for child in value)
        return value

    return clone(body)


__all__ = ["ApplicationContext", "ApplicationEvaluation", "OperatorApplication",
           "ApplicationProjection", "RateApplicationProjection"]
