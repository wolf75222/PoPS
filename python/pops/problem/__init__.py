"""pops.problem -- the declarative top-level assembly root (Spec 5 sec.5.16 / sec.11).

:class:`Case` is the ONE public assembly a user authors before lowering: blocks, elliptic
fields, params, aux, outputs and a time scheme, split into typed internal registries
(:mod:`pops.problem.registries`) behind a compact facade. The package owns no runtime data, no
codegen and no ``_pops`` import; ``pops.compile(problem, layout=...)`` / ``pops.bind(...)`` do the
lowering. The stable authoring handles (:mod:`pops.problem.handles`) and the immutable
:class:`pops.ReportTree` validation explanation complete the surface.
"""
from pops._report import DiagnosticError, ReportTree
from pops.problem.handles import (
    BlockHandle, FieldHandle, OperatorHandle, StateHandle)
from pops.problem.problem import Case
from pops.problem.layout_subjects import LayoutSubjects
from pops.problem._snapshot import AuthoringSnapshot

__all__ = ["Case", "LayoutSubjects", "AuthoringSnapshot", "BlockHandle", "StateHandle", "FieldHandle",
           "OperatorHandle", "ReportTree", "DiagnosticError"]
