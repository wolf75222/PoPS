"""Refusals of the user surface, reported in the user's terms (spec 0.3, section 2.6, SU11, §14.1).

A refusal names the file and the line of the user's statement that caused it, the rule that it
breaks in one sentence, and the construct to write in its place. The site is the first frame of
the stack outside the ``pops`` package and Python's own library; a frame of ``pops.lib`` between
the two is named after it, ``(in pops/lib/...:line)``, as the mock of R12c did. A refusal met when
a case is elaborated, after the statement that caused it has returned, takes the site that the
construct recorded when it was written.

Each class is a row of the refusal table of §14.1; its ``rules`` cite the rules that state it and
``phase`` is the phase of that table that raises it.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pops.time._program.traced_method import NotYetElaborated as _TracedNotYetElaborated

_POPS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_ROOT = os.path.join(_POPS_ROOT, "lib")
_PYTHON_ROOTS = tuple(sorted({os.path.abspath(path) for path in (
    sysconfig.get_paths().get("stdlib"), sysconfig.get_paths().get("platstdlib")) if path}))

_F = TypeVar("_F", bound=Callable[..., Any])


class Site:
    """The file and the line of a user statement, and the library line that met it, if any."""

    __slots__ = ("file", "line", "library")

    def __init__(self, file: str, line: int, library: tuple | None = None) -> None:
        self.file = file
        self.line = int(line)
        self.library = library

    def __str__(self) -> str:
        text = "%s:%d" % (self.file, self.line)
        if self.library is not None:
            text += " (in %s:%d)" % self.library
        return text

    def __repr__(self) -> str:
        return "Site(%s)" % self


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _is_python(path: str) -> bool:
    return path.startswith("<") or any(_inside(path, root) for root in _PYTHON_ROOTS)


def user_site(depth: int = 1) -> Site | None:
    """The site of the user statement that is running: the first frame outside pops and Python.

    ``depth`` skips the frames of the caller. A frame of ``pops.lib`` met on the way is kept as the
    library line of the site (SU11: the library frames are left out of the trace, not of the
    message).
    """
    frame = sys._getframe(depth)
    library = None
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if _inside(path, _LIB_ROOT):
            if library is None:
                library = (os.path.relpath(path, os.path.dirname(_POPS_ROOT)), frame.f_lineno)
        elif not _inside(path, _POPS_ROOT) and not _is_python(path):
            return Site(path, frame.f_lineno, library)
        frame = frame.f_back
    return None


def site_of_traceback(tb: Any) -> Site | None:
    """The innermost user frame of a traceback: where a refusal raised by the elaborated form met
    the user's text (a method body that the trace of N8 runs, for example)."""
    found = None
    library = None
    while tb is not None:
        path = os.path.abspath(tb.tb_frame.f_code.co_filename)
        if _inside(path, _LIB_ROOT):
            library = (os.path.relpath(path, os.path.dirname(_POPS_ROOT)), tb.tb_lineno)
        elif not _inside(path, _POPS_ROOT) and not _is_python(path):
            found, library = Site(path, tb.tb_lineno), None
        tb = tb.tb_next
    if found is not None and library is not None:
        found.library = library
    return found


class SurfaceRefusal(Exception):
    """A refusal of the surface: the user's site, the rule in one sentence, the construct to write.

    ``rule`` and ``phase`` follow the board's refusals: ``rule`` is the first id of ``rules``;
    the sentence is ``statement``.

    ``str()`` is the message of SU11: ``file:line: Name: rule (ids). Write construct.``
    """

    rules: tuple = ()
    rule: str | None = None
    phase = "tracing"
    code = "surface_refusal"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # ``rule`` is the id of the rule a row states, as on every refusal of the board
        # (``pops._ir.refusals``): the first of ``rules``. A row without ids keeps the rule of a
        # board refusal it also subclasses, or has none. The sentence is ``statement``.
        super().__init_subclass__(**kwargs)
        if cls.rules:
            cls.rule = cls.rules[0]
            return
        cls.rule = next((base.__dict__["rule"] for base in cls.__mro__[1:]
                         if not issubclass(base, SurfaceRefusal) and "rule" in base.__dict__),
                        None)

    def __init__(self, statement: str, *, write: str | None = None, site: Site | None = None,
                 also: Site | None = None) -> None:
        self.statement = statement
        self.write = write
        self.site = user_site(2) if site is None else site
        self.also = also
        super().__init__(self._message())

    def traced_at(self, site: Site | None) -> SurfaceRefusal:
        """The refusal, naming also the line where the case is traced when that is another line."""
        if self.also is None and site is not None and (
                self.site is None or (self.site.file, self.site.line) != (site.file, site.line)):
            self.also = site
            self.args = (self._message(),)
        return self

    def _message(self) -> str:
        where = "<unknown site>" if self.site is None else str(self.site)
        text = "%s: %s: %s" % (where, type(self).__name__, self.statement)
        if self.rules:
            text += " (%s)" % ", ".join(self.rules)
        text += "."
        if self.write:
            text += " Write %s." % self.write
        if self.also is not None:
            text += " The case is traced at %s." % self.also
        return text


class SymbolicTruthValue(SurfaceRefusal, TypeError):
    rules = ("E1", "SU11")
    code = "symbolic_truth_value"


class SymbolNotInSurface(SurfaceRefusal, ImportError):
    rules = ("SU3", "SU13", "§21.7")
    code = "symbol_not_in_surface"


class NotAPopsOperation(SurfaceRefusal, TypeError):
    rules = ("SU4", "SU5")
    code = "not_a_pops_operation"


class ReservedName(SurfaceRefusal, ValueError):
    rules = ("SU8",)
    code = "reserved_name"


class LevelMismatch(SurfaceRefusal, TypeError):
    rules = ("SU6",)
    code = "level_mismatch"


class OutOfScope(SurfaceRefusal, TypeError):
    rules = ("SU7",)
    code = "out_of_scope"


class IncrementTypeMismatch(SurfaceRefusal, TypeError):
    rules = ("SU10", "T34")
    code = "increment_type_mismatch"


class IncrementNotDerivable(SurfaceRefusal, ValueError):
    rules = ("SU10", "T34")
    phase = "validate"
    code = "increment_not_derivable"


class ClosedBodyViolation(SurfaceRefusal, TypeError):
    rules = ("T33",)
    phase = "validate"
    code = "closed_body_violation"


class TermCountedTwice(SurfaceRefusal, ValueError):
    rules = ("R7",)
    phase = "validate"
    code = "term_counted_twice"


class UnknownName(SurfaceRefusal, NameError):
    rules = ("CC1",)
    phase = "validate"
    code = "unknown_name"


class SupportMismatch(SurfaceRefusal, ValueError):
    rules = ("E2",)
    phase = "validate"
    code = "support_mismatch"


class ContradictsDerivation(SurfaceRefusal, ValueError):
    rules = ("SU3",)
    phase = "validate"
    code = "contradicts_derivation"


class AntiDiffusive(SurfaceRefusal, ValueError):
    rules = ("R2", "§2.6.2")
    phase = "validate"
    code = "anti_diffusive"


class NotDiffusive(SurfaceRefusal, ValueError):
    rules = ("R2", "§2.6.2")
    phase = "validate"
    code = "not_diffusive"


class CoverageNotExact(SurfaceRefusal, ValueError):
    rules = ("D5", "T5", "CC14")
    phase = "resolve"
    code = "coverage_not_exact"


class AmbiguousElaboration(SurfaceRefusal, ValueError):
    rules = ("D5d", "SU9")
    phase = "resolve"
    code = "ambiguous_elaboration"


class RealizationMismatch(SurfaceRefusal, ValueError):
    rules = ("D5", "§2.6.4")
    phase = "resolve"
    code = "realization_mismatch"


class MissingBoundaryRealization(SurfaceRefusal, ValueError):
    rules = ("SU9", "D9")
    phase = "resolve"
    code = "missing_boundary_realization"


class NotYetElaborated(SurfaceRefusal, _TracedNotYetElaborated):
    """A construct of the surface whose elaborated form this lot does not build yet.

    ``owner`` names the lot that elaborates it; without an owner, ``construct`` is a sentence that
    names its lot already, as a refusal of N8 does. The refusal keeps the site and the format of
    the other refusals, so that the user sees where the construct was written (P13: honest status).
    """

    rules = ()
    code = "not_yet_elaborated"

    def __init__(self, construct: str, owner: str | None, *, site: Site | None = None) -> None:
        self.construct = construct
        self.owner = owner
        statement = construct.rstrip(".") if owner is None else \
            "%s has no elaborated form yet; %s elaborates it" % (construct, owner)
        super().__init__(statement, site=user_site(2) if site is None else site)


class SpellingRefusal:
    """The refusal of a spelling of an earlier text: the construct to write in its place (§21.7).

    Each row is the message of ``SymbolNotInSurface`` for that spelling, so its text is error
    text.
    """

    __slots__ = ("write",)

    def __init__(self, write: str) -> None:
        self.write = write

    def refuse(self, spelling: str, where: str) -> SymbolNotInSurface:
        return SymbolNotInSurface("%s is a spelling of an earlier text; %s does not take it"
                                  % (spelling, where), write=self.write)


def surface_entry(function: _F) -> _F:
    """Trim the frames of the elaborator from a refusal raised through a public construct.

    Python keeps the frame of this wrapper; the frames below it, inside pops, are left out
    (SU11). Any other exception passes unchanged.
    """

    @wraps(function)
    def entry(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except SurfaceRefusal as refusal:
            raise refusal.with_traceback(None) from None

    return entry  # type: ignore[return-value]
