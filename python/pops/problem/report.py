"""Problem-facing home of the single immutable report tree.

Problem validation no longer has a second issue/report hierarchy or a mutable accumulator.  Each
registry returns a ``ReportTree`` and the aggregate Problem report composes those trees explicitly.
"""
from pops._report import DiagnosticError, ReportPhase, ReportSeverity, ReportTree

__all__ = ["DiagnosticError", "ReportPhase", "ReportSeverity", "ReportTree"]
