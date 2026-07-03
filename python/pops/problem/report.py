"""pops.problem.report -- the aggregated per-family validation report (ADC-553 / ADC-527).

A Problem's ``validate()`` accumulates STRUCTURED validation issues, one per detected problem,
grouped by the assembly FAMILY that raised them (``block`` / ``field`` / ``time`` / ``runtime`` /
``amr`` / ``params`` / ...). It is the per-registry return of ``validate(context)`` and the
aggregate return of ``Problem.validate_report()``: instead of a bare exception the caller gets an
inspectable object whose :meth:`by_family` lists the errors per subsystem (ADC-553 acceptance).

ADC-527 unifies this with the descriptor-side report: ``ProblemValidationReport`` /
``ProblemValidationIssue`` are the Problem-facing NAMES of the single
:class:`pops.descriptors_report.ValidationReport` / :class:`~pops.descriptors_report.ValidationIssue`
shape, so there is exactly ONE report class across the descriptor and Problem surfaces. Importing
:mod:`pops.descriptors_report` keeps :mod:`pops.problem` runtime / codegen / ``_pops`` free (that
module is pure stdlib). The accumulate / by_family / ok / raise_if_error surface is unchanged.
"""
from pops.descriptors_report import ValidationIssue, ValidationReport

# The Problem-facing aliases of the single validation-report shape (ADC-527: one class, two names).
ProblemValidationReport = ValidationReport
ProblemValidationIssue = ValidationIssue

__all__ = ["ProblemValidationReport", "ProblemValidationIssue",
           "ValidationReport", "ValidationIssue"]
