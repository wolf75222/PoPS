"""One fail-closed planner for simultaneous top-level finite-volume RHS evaluations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence


def uses_default_flux(value: Any) -> bool:
    """Whether *value* reads the block's default finite-volume flux."""
    if getattr(value, "op", None) != "rhs" or not value.attrs.get("flux", True):
        return False
    fluxes = value.attrs.get("fluxes")
    return not fluxes or tuple(fluxes) == ("default",)


def groupable_default_rhs(value: Any) -> bool:
    """Whether one RHS can enter a native simultaneous-interface transaction."""
    if not uses_default_flux(value):
        return False
    requested = value.attrs.get("sources")
    return not any(source != "default" for source in (requested or ()))


@dataclass(frozen=True)
class RhsCoherenceRound:
    """One occurrence-aligned RHS round at one exact StagePoint."""

    point: Any
    occurrence: int
    indexed_values: tuple[tuple[int, Any], ...]
    barrier_index: int

    @property
    def values(self) -> tuple[Any, ...]:
        return tuple(value for _, value in self.indexed_values)


@dataclass(frozen=True)
class RhsCoherencePlan:
    """Immutable rounds plus the subset that must lower to ``rhs_group``."""

    rounds: tuple[RhsCoherenceRound, ...]

    @property
    def schedule(self) -> dict[int, tuple[Any, ...]]:
        return {
            round_.barrier_index: round_.values
            for round_ in self.rounds if len(round_.indexed_values) > 1
        }

    @property
    def grouped_ids(self) -> frozenset[int]:
        return frozenset(
            value.id
            for round_ in self.rounds if len(round_.indexed_values) > 1
            for _, value in round_.indexed_values
        )


@dataclass
class _StageRounds:
    point: Any
    counts: dict[Hashable, int]
    rounds: list[list[tuple[int, Any]]]
    cohort: frozenset[Any]


def _reorderable_ops(program: Any) -> frozenset[str]:
    # Unknown operations remain barriers. ``state``/``history``/``scalar_field`` are inert bindings;
    # ``reduce`` is value-only. The same allow-list is consumed by resolve validation and emission.
    return frozenset(program._PURE_OPS) | {"state", "history", "scalar_field", "reduce"}


def resolved_rhs_neighbours(blocks: Sequence[Any]) -> dict[str, frozenset[str]] | None:
    """Capture exact boundary state dependencies before authoring authorities detach.

    An empty edge table is proved independence. Unknown numerical/boundary contracts
    return None, retaining the conservative planner instead of guessing locality.
    """
    from pops.mesh.boundaries.ghost_plan import GhostProducerPlan
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
    from pops.identity import canonical_bytes

    if any(getattr(block, "state_identities", None) is None for block in blocks):
        return None
    names = {block.name for block in blocks}
    neighbours = {name: set() for name in names}
    state_owners = {identity: block.name for block in blocks for identity in block.state_identities}
    interfaces: dict[bytes, set[str]] = {}

    def add_states(owner: str, states: Any) -> bool:
        for state in states:
            identity = state if isinstance(state, str) else getattr(state, "qualified_id", None)
            peer = state_owners.get(identity)
            if peer is None:
                return False
            if peer != owner:
                neighbours[owner].add(peer)
                neighbours[peer].add(owner)
        return True

    for block in blocks:
        numerics = block.numerics
        if numerics is None:
            return None
        for boundary in numerics.boundaries:
            if type(boundary) is GhostProducerPlan:
                for production in boundary.productions:
                    for provider in production.producer.boundary_providers:
                        if provider.dependencies.fields or not add_states(block.name, provider.dependencies.states):
                            return None
                for contribution in boundary.residual_contributions:
                    if getattr(contribution, "fields", ()) or not add_states(block.name, contribution.states):
                        return None
                declared_interfaces = tuple(row.canonical_identity() for row in boundary.interfaces)
            elif type(boundary) is CompiledBoundaryPlan:
                data = boundary.canonical_identity()["compile_data"]
                for row in data["component_region_templates"]:
                    if row.get("fields") or not add_states(block.name, row.get("states", ())):
                        return None
                declared_interfaces = data.get("interfaces", ())
            else:
                return None
            for interface in declared_interfaces:
                interfaces.setdefault(canonical_bytes(interface), set()).add(block.name)
    for participants in interfaces.values():
        if len(participants) != 2:
            return None
        left, right = participants
        neighbours[left].add(right)
        neighbours[right].add(left)
    return {name: frozenset(peers) for name, peers in neighbours.items()}


def _rhs_cohorts(values: Sequence[Any], neighbours: Any) -> dict[Any, frozenset[Any]]:
    """Boundary edges plus actual SSA/field-state reads determine connected cohorts."""
    names = {value.block.local_id for value in values if groupable_default_rhs(value)}
    if neighbours is None:
        cohort = frozenset(names)
        return {name: cohort for name in names}
    if not isinstance(neighbours, Mapping):
        raise TypeError("RHS coherence requires an explicit block-neighbour mapping")
    graph = {name: set(peers) for name, peers in neighbours.items()}
    if not names <= set(graph):
        raise ValueError("RHS coherence connectivity omits participating blocks")
    if any(peer not in graph or name not in graph[peer]
           for name, peers in graph.items() for peer in peers):
        raise ValueError("RHS coherence block connectivity must be complete and symmetric")

    def connect(owner: str, peer: Any) -> None:
        name = getattr(peer, "local_id", None)
        if name is not None and name != owner:
            if name not in graph:
                raise ValueError("RHS coherence dependency references an unknown block")
            graph[owner].add(name)
            graph[name].add(owner)

    field_readers = {}

    def connect_context(owner: Any, context: Any) -> None:
        if context is None:
            return
        for block, _ in context.stage_sources:
            connect(owner.local_id, block)
        previous = field_readers.setdefault(context.field, owner)
        connect(owner.local_id, previous)

    for value in values:
        if not groupable_default_rhs(value):
            continue
        owner = value.block.local_id
        visited = set()
        pending = list(value.inputs)
        connect_context(value.block, getattr(value, "field_context", None))
        while pending:
            producer = pending.pop()
            if producer.id in visited:
                continue
            visited.add(producer.id)
            connect(owner, getattr(producer, "block", None))
            pending.extend(producer.inputs)
            connect_context(value.block, getattr(producer, "field_context", None))
    cohorts = {}
    for name in graph:
        if name in cohorts:
            continue
        reached, pending = set(), [name]
        while pending:
            peer = pending.pop()
            if peer in reached:
                continue
            reached.add(peer)
            pending.extend(graph[peer])
        cohort = frozenset(reached)
        cohorts.update((peer, cohort) for peer in reached)
    return cohorts


def plan_rhs_coherence(
        program: Any, values: Sequence[Any], *,
        block_key: Callable[[Any], Hashable] = lambda value: value.block,
        neighbours: Any = None, model: Any = None,
) -> RhsCoherencePlan:
    """Partition default RHS nodes into deterministic occurrence-aligned rounds.

    For each exact StagePoint and connected block cohort, the first RHS of each block forms
    round zero, the second forms round one, and so on. Only a resolved model graph or an explicit
    resolved boundary-neighbour table can prove blocks independent; absent proof retains the
    conservative whole-Program cohort. A round may cross only operations proven pure by the Program contract.
    Interleaved rounds, an early consumer, a missing producer, or two groups requiring the same
    emission barrier are rejected instead of being compiled with an accepted-state boundary fallback.
    """
    if model is not None:
        from pops.codegen.program_models import ProgramModelGraph
        if type(model) is ProgramModelGraph:
            if neighbours is not None:
                raise TypeError("RHS coherence received competing connectivity authorities")
            neighbours = model.rhs_coherence_neighbours
    cohorts = _rhs_cohorts(values, neighbours)
    indexed = list(enumerate(values))
    stages: list[_StageRounds] = []
    for index, value in indexed:
        if not groupable_default_rhs(value):
            continue
        cohort = cohorts[value.block.local_id]
        stage = next((candidate for candidate in stages
                      if candidate.point == value.point and candidate.cohort == cohort), None)
        if stage is None:
            stage = _StageRounds(value.point, {}, [], cohort)
            stages.append(stage)
        key = block_key(value)
        try:
            occurrence = stage.counts.get(key, 0)
            stage.counts[key] = occurrence + 1
        except TypeError as error:
            raise TypeError("RHS coherence block identity must be hashable") from error
        while len(stage.rounds) <= occurrence:
            stage.rounds.append([])
        stage.rounds[occurrence].append((index, value))

    producer_index = {value.id: index for index, value in indexed}
    reorderable = _reorderable_ops(program)
    rounds: list[RhsCoherenceRound] = []
    occupied_barriers: set[int] = set()
    for stage in stages:
        for occurrence, candidates in enumerate(stage.rounds):
            if not candidates:
                raise ValueError("RHS coherence occurrence partition contains an empty round")
            first_index = candidates[0][0]
            if len(candidates) == 1:
                rounds.append(RhsCoherenceRound(
                    stage.point, occurrence, tuple(candidates), first_index))
                continue

            last_index = candidates[-1][0]
            member_ids = {value.id for _, value in candidates}
            input_producers: list[int] = []
            for _, value in candidates:
                state_id = value.inputs[0].id
                if state_id not in producer_index:
                    raise ValueError(
                        "RHS coherence cannot locate state producer id %d for %r"
                        % (state_id, value.name))
                input_producers.append(producer_index[state_id])
            barrier_index = max(first_index, max(input_producers) + 1)
            if barrier_index > last_index:
                raise ValueError(
                    "RHS coherence state producer lies after its authored evaluation round")

            for index in range(first_index, last_index + 1):
                node = values[index]
                if node.id in member_ids:
                    continue
                if groupable_default_rhs(node) and node.point == stage.point:
                    if cohorts[node.block.local_id] != stage.cohort:
                        continue
                    raise ValueError(
                        "RHS coherence rounds at one StagePoint are interleaved around %r; "
                        "author each complete block round before starting the next" % node.name)
                if node.op not in reorderable:
                    raise ValueError(
                        "RHS coherence cannot reorder a sibling residual across ordering barrier "
                        "%r (op=%s); materialize every participating block state and place the "
                        "residual round before side effects" % (node.name, node.op))

            for index in range(first_index, barrier_index):
                consumer = values[index]
                if consumer.id in member_ids:
                    continue
                consumed = sorted(item.id for item in consumer.inputs if item.id in member_ids)
                if consumed:
                    raise ValueError(
                        "RHS coherence cannot delay residual node(s) %s past consumer %r; "
                        "materialize every participating block stage state before evaluating the "
                        "first cross-block residual" % (consumed, consumer.name))

            if barrier_index in occupied_barriers:
                raise ValueError("RHS coherence rounds have an ambiguous emission barrier")
            occupied_barriers.add(barrier_index)
            rounds.append(RhsCoherenceRound(
                stage.point, occurrence, tuple(candidates), barrier_index))

    rounds.sort(key=lambda round_: round_.indexed_values[0][0])
    return RhsCoherencePlan(tuple(rounds))


__all__ = [
    "RhsCoherencePlan",
    "RhsCoherenceRound",
    "groupable_default_rhs",
    "plan_rhs_coherence",
    "resolved_rhs_neighbours",
    "uses_default_flux",
]
