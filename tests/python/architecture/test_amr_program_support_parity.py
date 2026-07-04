"""ADC-634: source-only parity gate between the AMR Program support mirror and its C++ header.

The AMR Program op-support surface has ONE source of truth in C++:
``include/pops/runtime/program/amr_program_context.hpp``. Every op the AMR Program path does NOT yet
wire fails loud there -- via a ``deferred_op("<name>", ...)`` call, a ``history_deferred(...)``
method, or a direct inline ``throw std::runtime_error`` in the method body. The Python capability
query (``python/pops/runtime/amr_program_support.py``) mirrors that deferral surface in
``DEFERRED_GROUPS`` so the Spec 6 matrix + ``inspect`` can report which ops run on AMR WITHOUT a
build.

This test locks the mirror against the header, WITHOUT a build (the source-only architecture tier
has no ``_pops``): it loads ``amr_program_support`` standalone (the module is deliberately
import-free), parses the header's deferral sites with a tolerant, comment-aware scanner, and asserts
the header-derived deferred-identifier set EQUALS the mirror's ``header_deferred_methods()``.

This is the AUTO-GREEN LOCK. When ADC-631 removes the history throws and ADC-633 removes the Schur
``deferred_op`` sites (replacing them with real per-level implementations), the header-derived set
SHRINKS; this test then FAILS until the mirror is shrunk to match, so their PR cannot land without
greening the group -- and the capability query flips automatically, with no edit to any ADC-634 file.

Mirror of ``test_route_registry_parity.py`` (the routes.py <-> route_ids.hpp lock).
"""
import importlib.util
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SUPPORT_PY = REPO_ROOT / "python" / "pops" / "runtime" / "amr_program_support.py"
CONTEXT_HPP = (REPO_ROOT / "include" / "pops" / "runtime" / "program"
               / "amr_program_context.hpp")

# The ctor and the two private fail-loud HELPERS themselves throw, but they are not deferral SEAMS
# (deferred_op / history_deferred are the machinery the deferral methods call; the ctor throws on an
# unbuilt engine). They are excluded from the header-derived deferred set by NAME.
_NON_DEFERRAL_METHODS = frozenset({"AmrProgramContext", "deferred_op", "history_deferred"})


# --- amr_program_support.py (Python side): load standalone, no pops import --------------------
def _load_support_module():
    """Load amr_program_support.py by path, without importing the pops package or the compiled _pops."""
    spec = importlib.util.spec_from_file_location("_amr_program_support_parity", SUPPORT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- amr_program_context.hpp (C++ side): tolerant, build-free deferral parsing ----------------
def _strip_comments(text):
    """Drop // and /* */ comments while preserving string literals verbatim (mirror of the routes
    parity scanner): a commented-out throw must not count as a live deferral, and a `//` inside a
    string literal must survive."""
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


# A member method opening: `<ret-type...> <name>(<args>) [const] [[[noreturn]]] {`. The return type
# may span tokens / templates / references; we capture the LAST identifier before the '(' as the
# method name. Static / [[noreturn]] qualifiers are tolerated before it.
_METHOD_RE = re.compile(
    r"(?:\[\[noreturn\]\]\s*)?(?:static\s+)?[A-Za-z_][\w:<>,&*\s]*?"
    r"\b([A-Za-z_]\w*)\s*\([^;{]*?\)\s*(?:const\s*)?(?:\[\[noreturn\]\]\s*)?\{")
_DEFERRED_OP_RE = re.compile(r'\bdeferred_op\(\s*"([A-Za-z_]\w*)"')
_HISTORY_DEFERRED_RE = re.compile(r"\bhistory_deferred\s*\(")
_THROW_RE = re.compile(r"\bthrow\s+std::runtime_error\b")


def _method_spans(text):
    """Every member method as ``(name, body_start, body_end)`` over the (comment-stripped) text.

    ``body_start`` is the offset just past the opening ``{``; ``body_end`` is the matching ``}``
    (brace-and-string aware). Nested braces inside a body are tracked so a lambda / block does not end
    the method early. Used to attribute a throw / history_deferred site to its enclosing method.
    """
    spans = []
    for m in _METHOD_RE.finditer(text):
        name = m.group(1)
        open_brace = text.index("{", m.end() - 1)
        end = _match_brace(text, open_brace)
        spans.append((name, open_brace + 1, end))
    return spans


def _match_brace(text, open_index):
    """Offset of the ``}`` matching the ``{`` at @p open_index (string-aware)."""
    depth = 0
    i = open_index
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces from offset %d" % open_index)


def _enclosing_method(spans, pos):
    """The name of the INNERMOST method whose body contains offset @p pos, or ``None``."""
    best = None
    best_start = -1
    for name, start, end in spans:
        if start <= pos < end and start > best_start:
            best, best_start = name, start
    return best


def _parse_header_deferred_set(raw):
    """The set of deferral IDENTIFIERS the header declares (build-free).

    Three kinds, unioned:

      1. every string literal passed to ``deferred_op("<name>", ...)`` (the Schur / named-flux /
         scheduler-cache ops) -- 14 today;
      2. the enclosing method of every ``history_deferred(...)`` call (register_history / history /
         store_history);
      3. the enclosing method of every direct inline ``throw std::runtime_error`` that is a deferral
         SEAM -- excluding the ctor + the two private helpers (deferred_op / history_deferred), whose
         throws are machinery, not seams (apply_projection / the named solve_fields_from_state /
         solve_fields_from_blocks / scheduler_error).
    """
    text = _strip_comments(raw)
    spans = _method_spans(text)
    deferred = set()

    # (1) deferred_op("<name>") string literals.
    for m in _DEFERRED_OP_RE.finditer(text):
        deferred.add(m.group(1))

    # (2) history_deferred(...) -> the enclosing method name.
    for m in _HISTORY_DEFERRED_RE.finditer(text):
        method = _enclosing_method(spans, m.start())
        if method is not None and method not in _NON_DEFERRAL_METHODS:
            deferred.add(method)

    # (3) direct inline throw std::runtime_error -> the enclosing method name (deferral seams only). A
    # throw that lowers through deferred_op / history_deferred is already covered by (1)/(2); a direct
    # throw in the ctor / the helper defs is machinery, excluded by name.
    for m in _THROW_RE.finditer(text):
        method = _enclosing_method(spans, m.start())
        if method is not None and method not in _NON_DEFERRAL_METHODS:
            deferred.add(method)
    return deferred


# --- The parity gates -------------------------------------------------------------------------
def test_support_module_loads_standalone_and_stays_import_free():
    """amr_program_support.py must load by path, without importing pops, so this gate needs no build."""
    source = SUPPORT_PY.read_text(encoding="utf-8")
    offender = re.search(r"(?m)^\s*(?:import\s+pops|from\s+pops)\b", source)
    assert offender is None, (
        "python/pops/runtime/amr_program_support.py must stay import-free of the pops package at "
        "module scope: the source-only architecture gate loads it standalone via "
        "importlib.spec_from_file_location, BEFORE the compiled _pops extension exists; found %r"
        % (offender.group(0) if offender else None))

    module = _load_support_module()
    groups = module.deferred_groups()
    assert groups, "amr_program_support exposed no capability groups"
    assert set(groups.values()) <= {"green"} | {
        v for v in groups.values() if v.startswith("pending")}, groups


def test_header_deferred_set_matches_the_python_mirror():
    """The header-derived deferral set EQUALS amr_program_support.header_deferred_methods().

    This is the auto-green lock: when ADC-631 / ADC-633 remove their throws the header set shrinks and
    this assertion fails until the mirror shrinks with it, so the group cannot silently stay pending
    (or silently green) -- the two evolve as one set.
    """
    module = _load_support_module()
    mirror = set(module.header_deferred_methods())
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    assert header == mirror, (
        "AMR Program deferral surface drift between amr_program_context.hpp and the Python mirror:\n"
        "  only in the header (add to DEFERRED_GROUPS):        %s\n"
        "  only in the mirror (a header throw was removed?):   %s\n"
        "When ADC-631 / ADC-633 wire a deferred op, DROP its identifier from DEFERRED_GROUPS in the "
        "SAME PR so the group greens." % (sorted(header - mirror), sorted(mirror - header)))


def test_parser_finds_the_known_deferral_families():
    """Guard the parser itself: it must see the Schur, history, scheduler-cache and inline-throw
    families in the header (so a silently-empty parse cannot make the equality above vacuously pass)."""
    header = _parse_header_deferred_set(CONTEXT_HPP.read_text(encoding="utf-8"))
    # NB: the Schur deferral family (assemble_schur_coeffs / schur_reconstruct / ...) is GONE (ADC-633
    # wired it); the remaining families guard the parser: scheduler cache (deferred_op), named flux
    # (deferred_op), and the inline-throw seams (apply_projection / solve_fields_from_blocks).
    for needle in ("cache_should_update", "cache_effective_dt",      # scheduler cache (deferred_op)
                   "neg_div_flux_into",                              # named flux (deferred_op)
                   "apply_projection", "solve_fields_from_blocks"):  # inline-throw seams
        assert needle in header, (
            "the header parser missed the known deferral %r; the tolerant scanner is broken and the "
            "parity assertion would be vacuous" % needle)
    # The ctor + private helpers must NOT leak in as deferral seams.
    assert _NON_DEFERRAL_METHODS.isdisjoint(header), (
        "the parser mis-attributed a ctor / helper throw as a deferral seam: %s"
        % sorted(_NON_DEFERRAL_METHODS & header))


def test_ir_ops_mirror_the_codegen_op_group_sets():
    """The mirror's Schur ir_ops must EQUAL the codegen's own ``_SCHUR_PROGRAM_OPS`` (the single op-group
    source), so the capability query maps ops with the emit vocabulary, never a hand-drifted copy."""
    module = _load_support_module()
    kernels = (REPO_ROOT / "python" / "pops" / "codegen"
               / "program_emit_kernels.py").read_text(encoding="utf-8")
    m = re.search(r"_SCHUR_PROGRAM_OPS\s*=\s*frozenset\(\{([^}]*)\}\)", kernels, re.S)
    assert m is not None, "could not find _SCHUR_PROGRAM_OPS in program_emit_kernels.py"
    codegen_schur = set(re.findall(r'"([A-Za-z_]\w*)"', m.group(1)))
    assert set(module.DEFERRED_GROUPS["schur"]["ir_ops"]) == codegen_schur, (
        "amr_program_support schur ir_ops drifted from codegen _SCHUR_PROGRAM_OPS:\n"
        "  mirror : %s\n  codegen: %s"
        % (sorted(module.DEFERRED_GROUPS["schur"]["ir_ops"]), sorted(codegen_schur)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
