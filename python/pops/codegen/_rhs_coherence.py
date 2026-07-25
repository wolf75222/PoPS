"""One fail-closed planner for simultaneous top-level finite-volume RHS evaluations."""
from __future__ import annotations

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


def _reorderable_ops(program: Any) -> frozenset[str]:
    # Unknown operations remain barriers. ``state``/``history``/``scalar_field`` are inert bindings;
    # ``reduce`` is value-only. The same allow-list is consumed by resolve validation and emission.
    return frozenset(program._PURE_OPS) | {"state", "history", "scalar_field", "reduce"}


def plan_rhs_coherence(
        program: Any, values: Sequence[Any], *,
        block_key: Callable[[Any], Hashable] = lambda value: value.block,
) -> RhsCoherencePlan:
    """Partition default RHS nodes into deterministic occurrence-aligned rounds.

    For each exact StagePoint, the first RHS of every block forms round zero, the second forms round
    one, and so on. A round may cross only operations proven pure by the Program contract.
    Interleaved rounds, an early consumer, a missing producer, or two groups requiring the same
    emission barrier are rejected instead of being compiled with an accepted-state boundary fallback.
    """
    indexed = list(enumerate(values))
    stages: list[_StageRounds] = []
    for index, value in indexed:
        if not groupable_default_rhs(value):
            continue
        stage = next((candidate for candidate in stages if candidate.point == value.point), None)
        if stage is None:
            stage = _StageRounds(value.point, {}, [])
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
    "uses_default_flux",
]
