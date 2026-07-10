"""ADC-532 fence: pops.lib.time selects operators by typed handle, never by a free string.

The ready-scheme macros must not re-introduce a free-string operator selector. This source-only
test parses every ``python/pops/lib/time/*.py`` and fails if a macro passes a STRING LITERAL as the
operator selector into one of the operator-selecting seams:

  - ``_opcall(P, "<name>", ...)`` (the operator-first stage call),
  - ``P._call("<name>", ...)`` / ``_call("<name>", ...)`` (the internal lowering seam),
  - ``P.linear_source("<name>")`` (the linear-source reference),
  - ``P.apply(operator="<name>", ...)`` (a linear-source apply),
  - a string literal into ``_source_value`` / ``_linear_source_value`` (were they used here).

The flux/source DE-SUGARING is ALLOWED: ``_rhs_legacy(..., sources=[...])`` / ``_stage_rhs`` name
flux/source TERMS, not operator selectors, and a ``name=`` debug label is fine. The internal
``_call`` / ``_rhs_legacy`` seams themselves survive (they lower a resolved ``handle.name``); only a
FREE STRING passed as the operator argument is banned.

Also confirms the operator-first macros coerce their operator kwargs through ``_operator_handle`` (so
a stale string is refused, not silently taken).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB_TIME = REPO_ROOT / "python" / "pops" / "lib" / "time"

# The seams whose FIRST positional argument is an operator selector (a string literal there is banned).
_OPERATOR_POS0_SEAMS = {"_opcall", "_call", "_source_value", "_linear_source_value"}
# linear_source(<sel>) takes the selector as its first positional arg too.
_LINEAR_SOURCE_SEAMS = {"linear_source"}
# apply(operator=<sel>) / apply(<sel>, ...) takes the selector positionally or by operator=.
_APPLY_SEAMS = {"apply"}


def _call_name(node):
    """The bare function name of a Call node (``f`` or ``obj.f`` -> ``f``)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_str_literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _violations_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in _OPERATOR_POS0_SEAMS:
            # _opcall(P, sel, ...): the operator selector is the SECOND positional arg (after P);
            # _call(sel, ...): the FIRST. Flag a string literal in the selector slot.
            sel_index = 1 if name == "_opcall" else 0
            if len(node.args) > sel_index and _is_str_literal(node.args[sel_index]):
                out.append("%s:%d %s(...) takes a string operator selector"
                           % (path.name, node.lineno, name))
        elif name in _LINEAR_SOURCE_SEAMS:
            if node.args and _is_str_literal(node.args[0]):
                out.append("%s:%d linear_source(<str>) is a free-string selector"
                           % (path.name, node.lineno))
        elif name in _APPLY_SEAMS:
            if node.args and _is_str_literal(node.args[0]):
                out.append("%s:%d apply(<str>, ...) is a free-string selector" % (path.name, node.lineno))
            for kw in node.keywords:
                if kw.arg == "operator" and _is_str_literal(kw.value):
                    out.append("%s:%d apply(operator=<str>) is a free-string selector"
                               % (path.name, node.lineno))
    return out


def test_no_free_string_operator_selector_in_lib_time():
    violations = []
    for path in sorted(LIB_TIME.glob("*.py")):
        violations.extend(_violations_in(path))
    assert not violations, (
        "pops.lib.time must select operators by a typed OperatorHandle, not a free string "
        "(ADC-532):\n  " + "\n  ".join(violations))


def test_operator_first_macros_coerce_operator_kwargs_to_handles():
    # Each operator-first macro must funnel its operator kwargs through _operator_handle so a stale
    # string is refused. Assert the coercion call is present for the known operator kwargs.
    expect = {
        "imex.py": {"linear_source", "explicit_operator", "implicit_operator", "fields_operator"},
        "rk.py": {"rhs_operator", "fields_operator"},
        "predictor_corrector.py": {"fields_operator", "explicit_rate_operator", "implicit_operator"},
        "multistep.py": {"linear_source"},
        "strang.py": {"linear_operator"},
    }
    for fname, kwargs in expect.items():
        src = (LIB_TIME / fname).read_text(encoding="utf-8")
        assert "_operator_handle(" in src, "%s must coerce its operator kwargs via _operator_handle" % fname
        for kw in kwargs:
            assert '_operator_handle(%s,' % kw in src, (
                "%s: operator kwarg %r must be coerced through _operator_handle (ADC-532)"
                % (fname, kw))
