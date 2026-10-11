"""Data-only, ordered native field initialization without advancing a clock."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FieldSolve:
    """Solve the single resolved Case field owned by the initialized block.

    Missing or ambiguous qualified plans are refused before the seed is published.
    This operation never selects the runtime's legacy default field solver.
    """

    __pops_ir_immutable__ = True

    def to_data(self) -> dict[str, Any]:
        return {"operation": "solve_fields"}


@dataclass(frozen=True, slots=True)
class ProviderDerivative:
    """A cell-centred provider derivative on the complete Cartesian initial grid.

    The second-order one-sided stencil is used only at physical domain edges.
    Patch interfaces, including remote rank interfaces, use centred differences.
    """

    key: Any
    axis: int
    edge_order: int = 2
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops.model.provider_pack import ComponentKey

        if type(self.key) is not ComponentKey:
            raise TypeError("ProviderDerivative.key must be one exact ComponentKey")
        if type(self.axis) is not int or self.axis not in (0, 1, 2):
            raise ValueError("ProviderDerivative.axis must be a ranked Cartesian axis")
        if type(self.edge_order) is not int or self.edge_order != 2:
            raise ValueError("ProviderDerivative requires edge_order=2")

    def to_data(self) -> dict[str, Any]:
        return {"source": "provider_derivative", "key": self.key.to_data(),
                "axis": self.axis, "edge_order": self.edge_order}


@dataclass(frozen=True, slots=True, init=False)
class StateMap:
    """An ordered pointwise conservative-state map using exact provider inputs."""

    components: tuple[Any, ...]
    inputs: tuple[tuple[int, Any], ...]
    consumer_qid: str
    __pops_ir_immutable__ = True

    def __init__(self, *, components: Any, inputs: Any, consumer_qid: str) -> None:
        from pops.analytic import ScalarExpr
        from pops.model.provider_pack import ComponentKey

        expressions = tuple(components)
        if not expressions or any(type(expression) is not ScalarExpr
                                  for expression in expressions):
            raise TypeError("StateMap.components requires an ordered ScalarExpr sequence")
        if not isinstance(inputs, Mapping):
            raise TypeError("StateMap.inputs must map contiguous integer ids to exact inputs")
        if not isinstance(consumer_qid, str) or not consumer_qid \
                or consumer_qid.strip() != consumer_qid:
            raise TypeError("StateMap.consumer_qid must be a non-empty exact string")
        normalized = []
        for value_id, source in inputs.items():
            if type(value_id) is not int or value_id < 0:
                raise TypeError("StateMap input ids must be non-negative integers")
            if type(source) is ProviderDerivative:
                normalized.append((value_id, source))
                continue
            if not isinstance(source, (tuple, list)) or len(source) != 2:
                raise TypeError("StateMap inputs require ('state', component), "
                                "('provider', ComponentKey), or ProviderDerivative")
            kind, value = source
            if kind == "state":
                if type(value) is not int or value < 0:
                    raise TypeError("StateMap state component must be a non-negative integer")
            elif kind == "provider":
                if type(value) is not ComponentKey:
                    raise TypeError("StateMap provider input must be one exact ComponentKey")
            else:
                raise ValueError("StateMap input source must be 'state' or 'provider'")
            normalized.append((value_id, (kind, value)))
        if sorted(inputs) != list(range(len(inputs))):
            raise ValueError("StateMap input ids must be contiguous from zero")
        used = set()
        for expression in expressions:
            expression.validate()
            if expression.time_clocks():
                raise ValueError("initial StateMap cannot read a physical clock")
            used.update(value_id for value_id, _ in expression.input_references())
        if sorted(used) != list(range(len(inputs))):
            raise ValueError("StateMap must use exactly every declared input id")
        object.__setattr__(self, "components", expressions)
        object.__setattr__(self, "inputs", tuple(sorted(normalized)))
        object.__setattr__(self, "consumer_qid", consumer_qid)

    def to_data(self) -> dict[str, Any]:
        inputs = []
        for value_id, source in self.inputs:
            if type(source) is ProviderDerivative:
                row = source.to_data()
            elif source[0] == "state":
                row = {"source": "state", "component": source[1]}
            else:
                row = {"source": "provider", "key": source[1].to_data()}
            inputs.append({"value_id": value_id, **row})
        return {"operation": "map_state", "consumer_qid": self.consumer_qid,
                "components": [expression.to_data() for expression in self.components],
                "inputs": inputs}


@dataclass(frozen=True, slots=True, init=False)
class BootstrapProgram:
    """Cell-centre analytic seed and explicitly ordered native field/map stages.

    This is a bind-time program. Neither its solves nor its maps advance the
    physical clock or accepted-step counter. Adaptive initialization is refused
    until this ordered program has a qualified composite execution route.
    """

    seed: Any
    stages: tuple[Any, ...]
    consumer_qid: str
    native_route = "bootstrap_program"
    reprojectable = False
    __pops_ir_immutable__ = True

    def __init__(self, *, seed: Any, stages: Any, consumer_qid: str) -> None:
        from . import Analytic

        if type(seed) is not Analytic or seed.cell_integrals is not None:
            raise TypeError("BootstrapProgram.seed must be an Analytic point profile")
        if any(expression.time_clocks() for expression in seed.components):
            raise ValueError("BootstrapProgram seed cannot read a physical clock")
        rows = tuple(stages)
        if not rows or any(type(row) not in (FieldSolve, StateMap) for row in rows):
            raise TypeError("BootstrapProgram.stages requires FieldSolve/StateMap values")
        if type(rows[0]) is not FieldSolve or type(rows[-1]) is not FieldSolve:
            raise ValueError("BootstrapProgram stages must start and finish with FieldSolve")
        if not isinstance(consumer_qid, str) or not consumer_qid \
                or consumer_qid.strip() != consumer_qid:
            raise TypeError("BootstrapProgram.consumer_qid must be a non-empty exact string")
        for row in rows:
            if type(row) is not StateMap:
                continue
            if len(row.components) != len(seed.components):
                raise ValueError("BootstrapProgram maps must preserve the complete state shape")
            for expression in row.components:
                if expression.frame_id not in (None, seed.frame.canonical_id):
                    raise ValueError("BootstrapProgram map belongs to another physical frame")
            for _, source in row.inputs:
                if type(source) is ProviderDerivative and source.axis >= len(seed.frame.axes):
                    raise ValueError("BootstrapProgram derivative axis exceeds the physical rank")
                if type(source) is not ProviderDerivative and source[0] == "state" \
                        and source[1] >= len(seed.components):
                    raise ValueError("BootstrapProgram input component exceeds the target state")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "stages", rows)
        object.__setattr__(self, "consumer_qid", consumer_qid)

    def validate_for(self, state: Any) -> bool:
        return self.seed.validate_for(state)

    def initial_source_options(self) -> dict[str, Any]:
        return {"native_route": self.native_route, "frame_id": self.seed.frame.canonical_id,
                "seed_sampling": "cell_center", "consumer_qid": self.consumer_qid,
                "seed_components": [expression.to_data() for expression in self.seed.components],
                "stages": [stage.to_data() for stage in self.stages]}

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "profile": "bootstrap_program",
                **self.initial_source_options()}

    canonical_identity = to_data
    inspect = to_data

    def captured_reference_handles(self) -> tuple[Any, ...]:
        expressions = self.seed.components + tuple(expression for stage in self.stages
                     if type(stage) is StateMap for expression in stage.components)
        return tuple(dict.fromkeys(handle for expression in expressions
                                   for handle in expression.parameter_handles()))

    @classmethod
    def resolve_captured_references(cls, *, value_identity: Any, source_options: Any,
                                    references: Any, resolver: Any) -> dict[str, Any]:
        from . import _reference_key, _replace_captured_parameter_references
        from pops.model import ParamHandle

        if type(value_identity) is not dict or type(source_options) is not dict \
                or value_identity != {"schema_version": 1, "profile": "bootstrap_program",
                                      **source_options} \
                or source_options.get("native_route") != cls.native_route:
            raise ValueError("captured BootstrapProgram identity and source disagree")
        if type(references) is not tuple or not callable(resolver):
            raise TypeError("captured BootstrapProgram requires exact reference authorities")
        replacements = {}
        for handle in references:
            if type(handle) is not ParamHandle:
                raise TypeError("BootstrapProgram references must be exact ParamHandle values")
            authored = handle.canonical_identity() if handle.is_resolved else handle.inspect()
            key = _reference_key(authored, where="BootstrapProgram reference")
            resolved = resolver(handle)
            if key in replacements or type(resolved) is not ParamHandle \
                    or not resolved.is_resolved or resolved.param_kind != handle.param_kind:
                raise ValueError("BootstrapProgram parameter authority changed")
            replacements[key] = resolved.canonical_identity()
        used: set[str] = set()
        source = _replace_captured_parameter_references(source_options,
                 replacements=replacements, used=used, where="BootstrapProgram")
        if used != set(replacements):
            raise ValueError("BootstrapProgram references do not exactly cover its expressions")
        return {"source_options": source, "value_identity": {
                "schema_version": 1, "profile": "bootstrap_program", **source}}


__all__ = ["BootstrapProgram", "FieldSolve", "ProviderDerivative", "StateMap"]
