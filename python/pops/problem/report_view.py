"""pops.problem.report_view -- the typed inspection report of a Problem (ADC-564).

``Problem.inspect()`` returns a :class:`ProblemReport`: a typed :class:`pops.Report` carrying the
assembly's name / blocks / fields / params / aux / outputs / constraints / requirements /
capabilities as ATTRIBUTES, with :meth:`to_dict` the JSON bridge. It is inert -- built from the
registries' metadata, it triggers no validation and no compilation. This is distinct from the
per-registry ``inspect()`` dicts it composes and from ``Problem.to_dict()`` (the array-free
serialisation the snapshot / codegen consume, a superset that adds the stable handle ids).
"""
from pops._report import Report


class ProblemReport(Report):
    """The typed inspection report of a :class:`~pops.problem.problem.Problem` (ADC-564).

    Attributes (all read directly): ``name`` / ``category`` / ``native_id`` / ``options`` /
    ``requirements`` / ``capabilities`` / ``layout`` / ``blocks`` / ``fields`` / ``params`` /
    ``aux`` / ``outputs`` / ``constraints`` / ``time``. :meth:`to_dict` is the JSON bridge (its keys
    match the historical ``inspect()`` dict byte-for-byte so a ``to_dict()`` consumer is unchanged).
    """

    report_type = "problem"
    schema_version = 1

    def __init__(self, payload):
        # Store each field as an attribute AND keep the ordered payload for the byte-identical
        # to_dict (the historical inspect() dict shape).
        self._payload = dict(payload)
        for key, value in payload.items():
            setattr(self, key, value)

    def to_dict(self):
        # Byte-identical to the historical Problem.inspect() dict, stamped with the report identity.
        return self._stamp(dict(self._payload))

    def __str__(self):
        return ("problem %r [%s]: blocks=%s fields=%s params=%s aux=%s outputs=%s time=%s"
                % (self._payload.get("name"), self._payload.get("category"),
                   list(self._payload.get("blocks", {})), list(self._payload.get("fields", {})),
                   list(self._payload.get("params", {})), self._payload.get("aux"),
                   self._payload.get("outputs"), self._payload.get("time")))


__all__ = ["ProblemReport"]
