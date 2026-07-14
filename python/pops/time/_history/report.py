"""Typed restart report for the ADC-626 history-persistence replay.

``System.restart`` recomputes the non-stored ring slots of a selective-persistence checkpoint by
deterministic replay. This value object (a sibling of
:class:`pops.runtime.program_report.ProgramRuntimeReport`, metadata-only, JSON-ready) states, per
history, how many slots were STORED verbatim vs RECOMPUTED and how many replay steps that cost --
so no heavy replay is silent (the ADC-591 metadata-only inspection house rule). It is attached to the
System after a restart (``System.last_restart_report()``).
"""
import json


class HistoryReplayReport:
    """Stored-vs-recomputed ring-slot accounting for a selective-persistence restart (ADC-626).

    Inert and JSON-ready: plain scalars and a list of per-history dicts, no field arrays. Built by
    ``System.restart`` from the counts ``rebuild_history_slots`` returns; a Dense (or v1) restart
    reports every slot stored and zero recomputed."""

    schema_version = 1
    report_type = "history_replay"

    def __init__(self, histories=None):
        #: list of {name, depth, policy_kind, stored_slots, recomputed_slots, replay_steps}.
        self.histories = [dict(row) for row in (histories or [])]

    def add(self, *, name, depth, policy_kind, stored_slots, recomputed_slots, replay_steps):
        """Record one history's accounting (chains)."""
        self.histories.append({
            "name": str(name),
            "depth": int(depth),
            "policy_kind": str(policy_kind),
            "stored_slots": int(stored_slots),
            "recomputed_slots": int(recomputed_slots),
            "replay_steps": int(replay_steps),
        })
        return self

    @property
    def total_stored(self):
        return sum(row["stored_slots"] for row in self.histories)

    @property
    def total_recomputed(self):
        return sum(row["recomputed_slots"] for row in self.histories)

    @property
    def total_replay_steps(self):
        return sum(row["replay_steps"] for row in self.histories)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "histories": [dict(row) for row in self.histories],
            "total_stored": self.total_stored,
            "total_recomputed": self.total_recomputed,
            "total_replay_steps": self.total_replay_steps,
        }

    def to_json(self, path=None, *, indent=2):
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __repr__(self):
        return ("HistoryReplayReport(histories=%d, recomputed=%d, replay_steps=%d)"
                % (len(self.histories), self.total_recomputed, self.total_replay_steps))

    def __str__(self):
        lines = ["history replay report (schema=%d)" % self.schema_version]
        lines.append("  histories       : %d ring(s)" % len(self.histories))
        lines.append("  total stored    : %d slot(s)" % self.total_stored)
        lines.append("  total recomputed: %d slot(s)" % self.total_recomputed)
        lines.append("  total replay    : %d step(s)" % self.total_replay_steps)
        for row in self.histories:
            lines.append("  - %s: %s stored=%d recomputed=%d replay=%d"
                         % (row["name"], row["policy_kind"], row["stored_slots"],
                            row["recomputed_slots"], row["replay_steps"]))
        return "\n".join(lines)


__all__ = ["HistoryReplayReport"]
