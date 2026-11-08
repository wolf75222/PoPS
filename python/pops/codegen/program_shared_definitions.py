"""Definitions that every call site of one generated translation unit shares.

A generated Program used to spell, at each call site, two fresh closures for the continuation
field callbacks and a fresh construction of each continuation schema. Each fresh closure type
instantiated the whole finite-structural packing template again, and each construction was
parsed and optimized again; the Hall MHD CT Program spent most of its compiler memory there
(B03 family F7, gap C49). Within one translation unit this registry emits each of them once:

- the two field callbacks as class templates over the program context type, whose call
  operators have the bodies of the closures they replace;
- each distinct schema construction as one accessor returning a copy of a constant built once
  from that same construction text;
- each distinct terminal failure of a distributed map outcome as one routine over the context and
  the outcome, whose body is the statement the exit emitted inline;
- each distinct staging of a distributed Array captured by a device map as one routine over the
  lane, the registry, the Array, the earlier domains of the map and the variables the staging
  writes, whose body is the staging statements of that capture.

The emitted values are unchanged: an accessor returns a schema equal to the one its construction
builds, chosen by the exact construction text, so no schema identity changes (G07, CC2). A routine
runs the same statements on the same objects. Outside a translation-unit scope, for instance in a
unit test of one emitter, the inline forms are returned.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
from typing import Any

_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("pops_shared_definitions", default=None)

_FIELD_COPY = ("[&](const auto& source, const auto& destination, const auto& lifetime, "
               "const auto&... metadata) { "
               "return ctx.continuation_bind_field_provenance(source, destination, lifetime, metadata...); }")
_FIELD_PREFLIGHT = ("[&](const auto& source, const auto& type, const auto&... metadata) { "
                    "ctx.continuation_preflight_field(source, type, metadata...); }")

_CALLBACK_DEFINITIONS = """\
// Continuation field callbacks: one type per program context, so the finite-structural packing
// templates are instantiated once per translation unit instead of once per call site. Like the
// closures they replace, they have internal linkage.
namespace {
template <class Context>
struct pops_continuation_field_copy {
  Context& ctx;
  template <class Source, class Destination, class Lifetime, class... Metadata>
  auto operator()(const Source& source, const Destination& destination, const Lifetime& lifetime,
                  const Metadata&... metadata) const {
    return ctx.continuation_bind_field_provenance(source, destination, lifetime, metadata...);
  }
};
template <class Context>
pops_continuation_field_copy(Context&) -> pops_continuation_field_copy<Context>;
template <class Context>
struct pops_continuation_field_preflight {
  Context& ctx;
  template <class Source, class Type, class... Metadata>
  void operator()(const Source& source, const Type& type, const Metadata&... metadata) const {
    ctx.continuation_preflight_field(source, type, metadata...);
  }
};
template <class Context>
pops_continuation_field_preflight(Context&) -> pops_continuation_field_preflight<Context>;
}  // namespace
"""


def shared_scope(function):
    """Give one translation-unit emission its own registry."""

    @wraps(function)
    def emit(*args, **kwargs):
        token = _ACTIVE.set({"callbacks": False, "schemas": {}, "routines": {}, "staging": {},
                             "sealed": False})
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return emit


def _open_registry() -> dict[str, Any] | None:
    registry = _ACTIVE.get()
    if registry is not None and registry["sealed"]:
        # A use emitted after the definitions were rendered would name a missing definition.
        raise RuntimeError("shared translation-unit definitions were already rendered")
    return registry


def field_callbacks() -> tuple[str, str]:
    """Return the copy and preflight callback arguments of one pack or unpack call."""
    registry = _open_registry()
    if registry is None:
        return _FIELD_COPY, _FIELD_PREFLIGHT
    registry["callbacks"] = True
    return "pops_continuation_field_copy{ctx}", "pops_continuation_field_preflight{ctx}"


def schema_reference(construction: str) -> str:
    """Return an expression with the value of @p construction, built once per translation unit."""
    registry = _open_registry()
    if registry is None:
        return construction
    schemas = registry["schemas"]
    name = "pops_continuation_schema_" + hashlib.sha256(construction.encode()).hexdigest()[:24]
    known = schemas.setdefault(name, construction)
    if known != construction:
        raise ValueError("continuation schema accessor identity collision")
    return name + "()"


# The parameter through which a shared terminal routine reads the outcome of its map.
OUTCOME_PARAMETER = "pops_outcome"


def terminal_failure_routine(lines: list[str]) -> str | None:
    """Return the routine running @p lines on (ctx, outcome), or None outside a scope."""
    registry = _open_registry()
    if registry is None:
        return None
    body = "\n".join(lines)
    name = "pops_terminal_failure_" + hashlib.sha256(body.encode()).hexdigest()[:24]
    known = registry["routines"].setdefault(name, body)
    if known != body:
        raise ValueError("terminal failure routine identity collision")
    return name


def staging_routine(parameters: list[str], lines: list[str]) -> str | None:
    """Return the routine running @p lines over @p parameters, or None outside a scope."""
    registry = _open_registry()
    if registry is None:
        return None
    signature = ", ".join(parameters)
    body = "\n".join(lines)
    digest = hashlib.sha256((signature + "\n" + body).encode()).hexdigest()[:24]
    name = "pops_stage_array_" + digest
    known = registry["staging"].setdefault(name, (signature, body))
    if known != (signature, body):
        raise ValueError("array staging routine identity collision")
    return name


def shared_definitions() -> str:
    """Render the definitions used so far; call after every region has been emitted."""
    registry = _open_registry()
    if registry is None:
        raise RuntimeError("shared definitions require a translation-unit scope")
    registry["sealed"] = True
    parts = [_CALLBACK_DEFINITIONS] if registry["callbacks"] else []
    for name, construction in registry["schemas"].items():
        parts.append(
            "static pops::runtime::program::ContinuationType %s() {\n"
            "  static const pops::runtime::program::ContinuationType schema = %s;\n"
            "  return schema;\n"
            "}\n" % (name, construction))
    if registry["routines"]:
        parts.append("// Terminal failures shared by every distributed map exit with the same statement.\n"
                     "namespace {\n")
        for name, body in registry["routines"].items():
            parts.append(
                "template <class Context>\n"
                "[[noreturn]] void %s(Context& ctx, const pops::runtime::program::distributed::"
                "Outcome<pops::kNativeDimension>& %s) {\n%s\n}\n"
                % (name, OUTCOME_PARAMETER, body))
        parts.append("}  // namespace\n")
    if registry["staging"]:
        # Not inlined back: each routine is compiled once however many captures call it.
        parts.append("// Captured Array staging shared by every device map capture with the same statements.\n"
                     "namespace {\n")
        for name, (signature, body) in registry["staging"].items():
            parts.append(
                "#if defined(__GNUC__) || defined(__clang__)\n"
                "__attribute__((noinline))\n"
                "#endif\n"
                "void %s(%s) {\n%s\n}\n" % (name, signature, body))
        parts.append("}  // namespace\n")
    return "".join(parts)
