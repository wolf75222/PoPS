"""Hold-then-catch-up Program composition from per-block strides.

A block with stride ``M`` is held for ``M-1`` macro-steps and then advances once with the
accumulated window ``M*dt``. The Program cadence owns the window; child clocks + subcycle
advance faster blocks ``window/M`` times when the window closes. This is the same composition
already proved on uniform System (two-clock SampleAndHold).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, lcm
from typing import Any, Mapping, Sequence


def adaptive_strides(wave_speeds: Mapping[str, Any]) -> dict[str, int]:
    """Return the oracle stride map ``m_b = max(1, floor(w_max / w_b))``."""
    if not wave_speeds:
        raise ValueError("adaptive_strides requires at least one wave speed")
    speeds = {str(name): float(speed) for name, speed in wave_speeds.items()}
    if any(not (speed > 0.0) for speed in speeds.values()):
        raise ValueError("adaptive_strides requires strictly positive wave speeds")
    fastest = max(speeds.values())
    return {name: max(1, int(floor(fastest / speed))) for name, speed in speeds.items()}


@dataclass(frozen=True, slots=True)
class HoldCatchupBlock:
    """One block's cadence and optional IMEX mask for :func:`hold_catchup_program`."""

    state_handle: Any
    stride: int = 1
    linear_rate: float | None = None
    implicit_vars: tuple[str, ...] = ()
    implicit_roles: tuple[Any, ...] = ()
    kind: str = "explicit"
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stride, bool) or type(self.stride) is not int or self.stride < 1:
            raise ValueError("HoldCatchupBlock.stride must be a positive int")
        object.__setattr__(self, "implicit_vars", tuple(self.implicit_vars or ()))
        object.__setattr__(self, "implicit_roles", tuple(self.implicit_roles or ()))
        object.__setattr__(self, "kind", str(self.kind))


def _label(block: HoldCatchupBlock) -> str:
    if block.name:
        return str(block.name)
    handle = block.state_handle
    owner = getattr(handle, "block_ref", None)
    local = getattr(owner, "local_id", None) or getattr(owner, "name", None)
    if local:
        return str(local)
    return str(getattr(handle, "name", "block"))


def _window(strides: Sequence[int]) -> int:
    window = 1
    for stride in strides:
        window = lcm(window, int(stride))
    return window


def _is_imex(block: HoldCatchupBlock) -> bool:
    token = str(block.kind).lower()
    return "imex" in token or bool(block.implicit_vars) or bool(block.implicit_roles)


def _advance(program: Any, state: Any, block: HoldCatchupBlock, *, at: Any) -> Any:
    dt = program.dt
    if block.linear_rate is not None:
        return state + dt * (float(block.linear_rate) * state)
    if _is_imex(block):
        flux = program._rhs_primitive(state=state, flux=True, sources=[])
        explicit = program.explicit_source(
            state,
            implicit_vars=block.implicit_vars,
            implicit_roles=block.implicit_roles,
        )
        mid = program.value(
            "%s_mid" % _label(block),
            state + dt * flux + dt * explicit,
            at=at,
        )
        implicit = program.implicit_source(
            mid,
            implicit_vars=block.implicit_vars,
            implicit_roles=block.implicit_roles,
        )
        return mid + dt * implicit
    return state + dt * program._rhs_primitive(state=state, flux=True, sources=["default"])


def hold_catchup_program(
    blocks: Sequence[HoldCatchupBlock],
    *,
    name: str = "hold_catchup",
) -> Any:
    """Author the hold-then-catch-up Program for ``blocks``.

    The Program cadence stride is ``lcm`` of the block strides. A block with stride ``S``
    advances ``lcm/S`` times with child-tick duration ``S * dt_macro`` when the window
    closes. A user does not write ``Program.subcycle``.
    """
    from pops.time._program.api import Program
    from pops.time._schedule.synchronization import SampleAndHold
    from pops.time.points import Clock, TimePoint

    entries = tuple(blocks)
    if not entries:
        raise ValueError("hold_catchup_program requires at least one block")
    window = _window(block.stride for block in entries)
    program = Program(name)
    if window > 1:
        program.cadence(stride=window)
    parent = program.clock
    for block in entries:
        count = window // int(block.stride)
        label = _label(block)
        parent_state = program.state(block.state_handle)
        if count == 1:
            program.commit(
                parent_state.next,
                program.value(
                    "%s_catchup" % label,
                    _advance(program, parent_state.n, block, at=parent_state.next.point),
                    at=parent_state.next.point,
                ),
            )
            continue
        child = Clock("cadence_%s" % label, owner=program.owner_path)
        child_state = program.state(block.state_handle, clock=child)
        on_child = program.synchronize(
            parent_state.n,
            at=TimePoint(child),
            relation=SampleAndHold(),
            name="%s_to_child" % label,
        )

        def _tick(builder: Any, value: Any, _block: HoldCatchupBlock = block,
                  _child_state: Any = child_state, _label: str = label) -> Any:
            return builder.value(
                "%s_tick" % _label,
                _advance(builder, value, _block, at=_child_state.next.point),
                at=_child_state.next.point,
            )

        advanced = program.subcycle(
            on_child,
            clock=child,
            within=parent,
            count=count,
            body_fn=_tick,
            name="%s_ticks" % label,
        )
        program.commit(
            parent_state.next,
            program.synchronize(
                advanced,
                at=parent_state.next.point,
                relation=SampleAndHold(),
                name="%s_to_macro" % label,
            ),
        )
    return program


def step_adaptive_program(
    handles: Mapping[str, Any],
    wave_speeds: Mapping[str, Any],
    *,
    name: str = "step_adaptive",
    linear_rates: Mapping[str, float] | None = None,
    kinds: Mapping[str, str] | None = None,
    implicit_vars: Mapping[str, Sequence[str]] | None = None,
    implicit_roles: Mapping[str, Sequence[Any]] | None = None,
) -> Any:
    """Author the oracle adaptive-stride Program for ``handles`` / ``wave_speeds``."""
    strides = adaptive_strides(wave_speeds)
    missing = sorted(set(handles) - set(strides))
    extra = sorted(set(strides) - set(handles))
    if missing or extra:
        raise ValueError(
            "step_adaptive_program handles and wave_speeds must name the same blocks "
            "(missing=%s extra=%s)" % (missing, extra)
        )
    rates = dict(linear_rates or {})
    treatments = dict(kinds or {})
    vars_by_block = dict(implicit_vars or {})
    roles_by_block = dict(implicit_roles or {})
    blocks = [
        HoldCatchupBlock(
            handle,
            stride=strides[block_name],
            linear_rate=rates.get(block_name),
            implicit_vars=tuple(vars_by_block.get(block_name, ())),
            implicit_roles=tuple(roles_by_block.get(block_name, ())),
            kind=treatments.get(block_name, "explicit"),
            name=block_name,
        )
        for block_name, handle in handles.items()
    ]
    return hold_catchup_program(blocks, name=name)


__all__ = [
    "HoldCatchupBlock",
    "adaptive_strides",
    "hold_catchup_program",
    "step_adaptive_program",
]
