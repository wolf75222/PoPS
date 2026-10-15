"""Authored temporal point evaluation from a native macro-entry source bundle."""
from dataclasses import dataclass
from .transfer import HistoryTransfer


@dataclass(frozen=True, slots=True)
class TemporalSample:
    time: object
    interval: object
    available: object


@dataclass(frozen=True, slots=True, init=False)
class TemporalEvaluation:
    """Define ``body(samples, time) -> (valid, weights)`` in Python algebra.

    Sources are ordered ``[macro-entry State, ring slot 0, ..., ring slot N-1]``.
    Every source is frozen before the macro attempt. The first is an exact point
    with interval zero; retained slots expose their own time, interval and
    availability. A nonzero weight on an unavailable source is rejected natively.
    Only other levels are reconstructed: the active level uses the solve's exact
    explicitly supplied stage value. The graph defines the numerical method.
    """
    history: object
    transfer: HistoryTransfer

    def __init__(self, *, history, body):
        from pops.time.handles import HistoryHandle
        if type(history) is not HistoryHandle:
            raise TypeError("TemporalEvaluation requires an issued State keeper HistoryHandle")
        program = history._program
        _handle, state = program._require_history(history, "TemporalEvaluation")
        if state not in program._time_history_configs:
            raise ValueError("TemporalEvaluation requires explicit keep_history before authoring")
        slots = program._time_history_configs[state][0] + 2
        def graph(samples, target, ratio, *available):
            del ratio
            valid, weights = body(tuple(TemporalSample(item.time, item.interval, flag)
                for item, flag in zip(samples, available)), target)
            return valid, ((target, 0., weights),)
        transfer = HistoryTransfer(source_slots=slots, target_slots=1,
            parameters=tuple("available_%d" % i for i in range(slots)), body=graph)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "transfer", transfer)


__all__ = ["TemporalEvaluation", "TemporalSample"]
