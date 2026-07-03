"""pops.problem.report -- the aggregated per-family validation report (ADC-553).

A :class:`ProblemValidationReport` accumulates STRUCTURED validation issues, one per
detected problem, grouped by the assembly FAMILY that raised them
(``block`` / ``field`` / ``time`` / ``runtime`` / ``amr`` / ``params`` / ...). It is the
per-registry return of ``validate(context)`` and the aggregate return of
``Problem.validate()``: instead of a bare exception the caller gets an inspectable object
whose :meth:`by_family` lists the errors per subsystem (ADC-553 acceptance).

This is a SELF-CONTAINED report for commit 1 (ADC-553). Commit 3 (ADC-527) promotes it to a
thin subject-bound view over the descriptor-side ``pops.descriptors_report.ValidationReport``
so there is exactly ONE report shape; the accumulate / by_family / ok / raise_if_error surface
here is the frozen contract both share. It runs no numeric loop and touches no runtime.
"""


class ProblemValidationIssue:
    """One structured validation error with the family it belongs to and user context.

    ``family`` names the assembly subsystem (``block`` / ``field`` / ``time`` / ``runtime`` /
    ``amr`` / ``params`` / ``layout`` / ``descriptor``); ``code`` is a short stable slug and
    ``message`` the human-facing explanation. ``context`` carries the offending names / values,
    and ``alternatives`` the typed remedies to point the user at. It is inert metadata.
    """

    def __init__(self, *, family, code, message, context=None, severity="error",
                 alternatives=()):
        self.family = str(family)
        self.code = str(code)
        self.message = str(message)
        self.context = dict(context or {})
        self.severity = str(severity)
        self.alternatives = list(alternatives or [])

    def to_dict(self):
        return {"family": self.family, "code": self.code, "message": self.message,
                "context": dict(self.context), "severity": self.severity,
                "alternatives": list(self.alternatives)}

    def __str__(self):
        head = "[%s/%s] %s" % (self.family, self.code, self.message)
        if self.alternatives:
            head += " (alternatives: %s)" % ", ".join(self.alternatives)
        return head

    def __repr__(self):
        return "ProblemValidationIssue(family=%r, code=%r)" % (self.family, self.code)


class ProblemValidationReport:
    """An accumulated, per-family structured report (ADC-553 / ADC-527).

    Chaining accumulators (:meth:`add` / :meth:`error` / :meth:`extend`) collect issues without
    raising, so a single ``validate`` pass reports EVERY problem at once. :meth:`by_family` groups
    them for the per-subsystem listing, :attr:`ok` / ``__bool__`` report the pass/fail verdict, and
    :meth:`raise_if_error` keeps the fail-loud behaviour the strict callers rely on.
    """

    def __init__(self, subject=None):
        self.subject = subject
        self._issues = []

    def add(self, issue):
        """Append a pre-built :class:`ProblemValidationIssue` (chains)."""
        self._issues.append(issue)
        return self

    def error(self, family, code, message, *, context=None, alternatives=()):
        """Accumulate an error-severity issue built from its parts (chains)."""
        return self.add(ProblemValidationIssue(
            family=family, code=code, message=message, context=context,
            severity="error", alternatives=alternatives))

    def extend(self, other):
        """Fold every issue of another report into this one (chains)."""
        if other is not None:
            self._issues.extend(other.issues)
        return self

    @property
    def issues(self):
        return list(self._issues)

    def by_family(self):
        """Group the accumulated issues by their family (ADC-553 per-subsystem listing)."""
        grouped = {}
        for issue in self._issues:
            grouped.setdefault(issue.family, []).append(issue)
        return grouped

    @property
    def ok(self):
        return not any(issue.severity == "error" for issue in self._issues)

    def __bool__(self):
        return self.ok

    def __iter__(self):
        return iter(self._issues)

    def __len__(self):
        return len(self._issues)

    def raise_if_error(self):
        """Raise a clear ``ValueError`` naming every accumulated error (fail-loud for strict callers)."""
        if not self.ok:
            raise ValueError(str(self))

    def to_dict(self):
        return {"subject": getattr(self.subject, "name", None),
                "ok": self.ok, "issues": [issue.to_dict() for issue in self._issues]}

    def __str__(self):
        if self.ok:
            return "validation ok"
        lines = ["validation failed:"]
        for family, issues in self.by_family().items():
            lines.append("  %s:" % family)
            for issue in issues:
                lines.append("    - %s" % issue)
        return "\n".join(lines)

    def __repr__(self):
        return "ProblemValidationReport(%d issue(s), ok=%s)" % (len(self._issues), self.ok)


__all__ = ["ProblemValidationIssue", "ProblemValidationReport"]
