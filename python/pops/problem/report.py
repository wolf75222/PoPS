"""Case-facing home of the single immutable report tree.

Case validation no longer has a second issue/report hierarchy or a mutable accumulator.  Each
registry returns a ``ReportTree`` and the aggregate Case report composes those trees explicitly.
"""
from pops._report import DiagnosticError, ReportPhase, ReportSeverity, ReportTree

__all__ = ["DiagnosticError", "ReportPhase", "ReportSeverity", "ReportTree"]
