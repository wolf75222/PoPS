"""pops.codegen.inspect_report -- INERT printable reports for a compiled artifact (Spec 5 sec.12.1).

The ``print(compiled)`` inspection surface (criteria #15 / #40-41, epic ADC-479): three value
classes and the pure builders that AGGREGATE the metadata a
:class:`pops.codegen.loader.CompiledProblem` ALREADY carries -- its lowered ``pops.time.Program``,
its physical model and its compile artifacts -- into deterministic, array-free printables.

  - :class:`CompiledReport` (sec.12.1) is the ``print(compiled)`` summary: name, backend, platform,
    layout, blocks, fields, program (commits), required runtime inputs (from ``arguments()``) and
    the on-disk artifacts (so_path / abi_key / cache_key) with the bind-status line. It is the human
    counterpart of :meth:`CompiledProblem.arguments` (the machine-readable bind contract).
  - :class:`RequirementsReport` (sec.12.1) lists the COMPILE-TIME constraints -- the model
    capabilities the lowered route needs (wave_speeds / hllc_star_state / roe_dissipation), the
    required descriptors and the layout / backend constraints -- DISTINCT from the runtime bind
    inputs ``arguments()`` enumerates. A piece unknowable from today's metadata is reported honestly
    (never fabricated).
  - :class:`BindReport` (sec.12.1) is the ``sim.explain_bind(compiled)`` view: per group, which
    blocks / params / aux / solvers a System ALREADY provides vs which the artifact still REQUIRES,
    computed by reusing ADC-463 ``collect_missing_arguments``.

Nothing here compiles, binds, dlopens or allocates: the builders read Python-side metadata only.
The module imports ``pops.time`` / the runtime lazily (in-function) to keep the codegen import
graph acyclic (cf. tests/python/architecture/test_import_graph.py).
"""

import json

# CompiledReport (the print(compiled) summary) + its builders and hash helpers are split into
# ``_inspect_compiled_report`` for the 500-line cap and re-exported so the historical
# ``from pops.codegen.inspect_report import CompiledReport, build_compiled_report`` paths are
# unchanged. RequirementsReport and BindReport stay below.
from pops.codegen._inspect_compiled_report import (  # noqa: F401  (re-exported at the historical path)
    CompiledReport,
    build_compiled_report,
)


# ---------------------------------------------------------------------------
# sec.12.1 -- RequirementsReport: the COMPILE-TIME constraints
# ---------------------------------------------------------------------------

# Map a spatial-flux capability flag a CompiledModel carries to the capability TOKEN the C++
# requires-gate names (the verbatim section-24 message), and the flux that needs it.
_CAPABILITY_FLAGS = (
    ("has_hllc", "hllc_star_state", "HLLC Riemann flux"),
    ("has_roe", "roe_dissipation", "Roe Riemann flux"),
    ("has_wave_speeds", "wave_speeds", "HLL / wave-speed bounded flux"),
)


class RequirementsReport:
    """The COMPILE-TIME constraints of a compiled artifact (Spec 5 sec.12.1).

    DISTINCT from :meth:`CompiledProblem.arguments` (the runtime bind inputs): this lists what the
    lowered ROUTE needs from the model + toolchain -- the model capabilities the emitted flux relies
    on, the required descriptors and the layout / backend constraints. An inert record:
    :meth:`to_dict` is JSON-ready and :meth:`__str__` a short table. A piece genuinely unknowable
    from today's metadata is recorded in :attr:`unknown` (honestly, never fabricated).
    """

    def __init__(self, *, capabilities, descriptors, constraints, unknown):
        self.capabilities = list(capabilities)  # [{capability, used_by, provided}]
        self.descriptors = list(descriptors)    # [{slot, name}]
        self.constraints = dict(constraints)    # {backend, layout, abi_key, ...}
        self.unknown = list(unknown)            # honestly-deferred pieces

    def to_dict(self):
        return {"capabilities": [dict(c) for c in self.capabilities],
                "descriptors": [dict(d) for d in self.descriptors],
                "constraints": dict(self.constraints), "unknown": list(self.unknown)}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self):
        lines = ["compile-time requirements:"]
        lines.append("  model capabilities (the lowered flux relies on):")
        if self.capabilities:
            for cap in self.capabilities:
                lines.append("    %-18s used_by=%s provided=%s"
                             % (cap.get("capability"), cap.get("used_by"), cap.get("provided")))
        else:
            lines.append("    (none beyond the base Rusanov flux)")
        lines.append("  descriptors:")
        if self.descriptors:
            for desc in self.descriptors:
                lines.append("    %-14s %s" % (desc.get("slot") + ":", desc.get("name")))
        else:
            lines.append("    (selected at bind: the spatial brick is a bind input)")
        lines.append("  constraints:")
        for key in sorted(self.constraints):
            lines.append("    %-14s %s" % (key + ":", self.constraints[key]))
        if self.unknown:
            lines.append("  not recorded in today's metadata (honestly deferred):")
            for note in self.unknown:
                lines.append("    - %s" % note)
        return "\n".join(lines)

    def __repr__(self):
        return ("RequirementsReport(capabilities=%d, descriptors=%d, unknown=%d)"
                % (len(self.capabilities), len(self.descriptors), len(self.unknown)))


def build_requirements(compiled):
    """Build the :class:`RequirementsReport` of a compiled artifact (sec.12.1).

    Reads the carried model + metadata only (no compile / bind):

      - capabilities: each Riemann capability the model emits (``CompiledModel.has_hllc /
        has_roe / has_wave_speeds``) becomes a row naming the capability TOKEN and the flux that
        needs it; a model that carries no such flag (a base Rusanov-only or a composed native
        ``pops.Model``) yields no rows;
      - descriptors: the spatial scheme is a BIND input (chosen on the Problem block's ``spatial=`` and
        flowed by ``pops.bind``), so it is reported as bind-time -- the artifact does not freeze a
        reconstruction / Riemann descriptor at compile;
      - constraints: backend (always ``production`` for a compiled Program), the target layout,
        whether MPI is supported, the ABI key the toolchain must match;
      - unknown: pieces genuinely not in today's metadata (e.g. the exact reconstruction stencil
        width), recorded honestly rather than guessed.
    """
    model = getattr(compiled, "model", None)
    args = compiled.arguments()
    layout_runtime = getattr(args, "layout_runtime", {})

    capabilities = []
    for flag, token, used_by in _CAPABILITY_FLAGS:
        if getattr(model, flag, False):
            row = {"capability": token, "used_by": used_by, "provided": True}
            # ADC-552: name the derivable wave-speed provider kind next to the wave_speeds row
            # (additive key). None when the source is not derivable from today's metadata (a bare
            # CompiledModel without a carried authoring model): honest, never fabricated.
            if flag == "has_wave_speeds":
                from pops.numerics.riemann.waves import provider_of
                try:
                    provider = provider_of(model) if model is not None else None
                except ValueError:
                    provider = None
                row["wave_speed_provider"] = provider.kind if provider is not None else None
            capabilities.append(row)

    constraints = {
        "backend": "production",
        "layout": layout_runtime.get("layout", "system"),
        "supports_mpi": bool(layout_runtime.get("supports_mpi", False)),
        "abi_key": getattr(compiled, "abi_key", None),
        "cxx_standard": getattr(compiled, "std", None),
    }
    from pops.runtime_environment import compiled_runtime_facts
    runtime = compiled_runtime_facts(supports_mpi=layout_runtime.get("supports_mpi"))
    constraints.update({
        "dimension": runtime["dimension"],
        "amr_refinement_ratio": runtime["amr_refinement_ratio"],
        "precision": runtime["precision"],
        "communicator": runtime["communicator"],
        "supports_custom_communicator": runtime["supports_custom_communicator"],
    })

    unknown = [
        "the spatial scheme (reconstruction / Riemann / variables) is a BIND input -- it is chosen "
        "on the Problem block (block(..., spatial=...)) and flowed by pops.bind, not frozen at compile, "
        "so no descriptor is named here.",
        "the reconstruction stencil width (ghost depth) is not recorded in today's metadata; the "
        "memory estimate assumes the conservative 2-cell MUSCL halo (cf. estimate_memory).",
    ]
    # A native composed pops.Model carries its capabilities in its bricks (the C++ requires-gate is
    # the backstop), not in queryable has_* flags -- say so honestly rather than report "none".
    from pops.codegen.loader import CompiledModel  # lazy: codegen <-> loader edge
    if model is not None and not isinstance(model, CompiledModel):
        unknown.append(
            "this artifact carries a composed (non-CompiledModel) model; its flux capabilities live "
            "in its bricks (validated by the C++ requires-gate at first use), not in queryable "
            "has_* flags -- the capability rows above may be incomplete.")

    return RequirementsReport(capabilities=capabilities, descriptors=[],
                              constraints=constraints, unknown=unknown)


# ---------------------------------------------------------------------------
# sec.12.1 -- BindReport: provided vs still-required, for a given sim
# ---------------------------------------------------------------------------

class BindReport:
    """The ``sim.explain_bind(compiled)`` view: provided vs still-required (Spec 5 sec.12.1).

    A plain, inert record of which bind inputs a System / AmrSystem ALREADY provides and which the
    artifact still REQUIRES, per group (instances / params / aux / solvers). :attr:`missing` is the
    actionable list ADC-463 :func:`collect_missing_arguments` produces (each line names exactly what
    to supply); :attr:`ready` is true when nothing required is missing. :meth:`__str__` is a short,
    deterministic table.
    """

    def __init__(self, *, program_name, provided, required, missing):
        self.program_name = program_name
        self.provided = dict(provided)   # group -> sorted [names]
        self.required = dict(required)   # group -> sorted [names]
        self.missing = list(missing)     # actionable lines (ADC-463)

    @property
    def ready(self):
        """True when every REQUIRED bind input is already provided (no missing line)."""
        return not self.missing

    def to_dict(self):
        return {"program": self.program_name,
                "provided": {k: list(v) for k, v in self.provided.items()},
                "required": {k: list(v) for k, v in self.required.items()},
                "missing": list(self.missing), "ready": self.ready}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self):
        lines = ["bind plan for compiled artifact %r" % (self.program_name or "problem")]
        for group in ("instances", "params", "aux", "solvers"):
            req = self.required.get(group, [])
            prov = self.provided.get(group, [])
            still = [name for name in req if name not in prov]
            lines.append("  %-10s required=%s provided=%s still-needed=%s"
                         % (group, req or "(none)", prov or "(none)", still or "(none)"))
        if self.missing:
            lines.append("  MISSING (supply before install):")
            for note in self.missing:
                lines.append("    - %s" % note)
        else:
            lines.append("  ready: every required bind input is provided")
        return "\n".join(lines)

    def __repr__(self):
        return ("BindReport(program=%r, ready=%s, missing=%d)"
                % (self.program_name, self.ready, len(self.missing)))


def build_bind_report(sim, compiled):
    """Build a :class:`BindReport` of @p compiled against @p sim (System or AmrSystem) -- sec.12.1.

    INERT: reads ``compiled.arguments()`` (the DECLARED bind inputs) and the sim's already-wired
    blocks (``sim.block_names()``) + named aux (``sim._aux_field_index``), then reuses ADC-463
    :func:`pops.runtime._system_unified_install.collect_missing_arguments` to compute the
    provided-vs-missing split -- the SAME contract ``install`` enforces. It binds nothing and
    mutates nothing.
    """
    from pops.runtime._system_unified_install import collect_missing_arguments  # lazy: runtime edge

    args = compiled.arguments()
    required = {
        "instances": sorted(name for name, spec in getattr(args, "instances", {}).items()
                            if spec.get("required")),
        "params": sorted(name for name, spec in getattr(args, "params", {}).items()
                        if spec.get("required")),
        "aux": sorted(name for name, spec in getattr(args, "aux", {}).items()
                     if spec.get("required")),
        "solvers": sorted(name for name, spec in getattr(args, "solvers", {}).items()
                         if spec.get("required")),
    }

    provided_blocks = set()
    try:
        provided_blocks = set(sim.block_names())
    except Exception:  # noqa: BLE001 -- block_names is a convenience; absence is not a failure
        pass
    provided_aux = set()
    for table in getattr(sim, "_aux_field_index", {}).values():
        provided_aux |= set(table)
    # A System carries no pre-supplied runtime params / solvers before install, so those provided
    # sets are empty here: explain_bind reports the contract a FRESH install must still satisfy.
    provided = {"instances": sorted(provided_blocks), "params": [],
                "aux": sorted(provided_aux), "solvers": []}

    missing = collect_missing_arguments(args, provided_blocks, set(), provided_aux, set())

    return BindReport(program_name=getattr(compiled, "program_name", None),
                      provided=provided, required=required, missing=missing)


__all__ = ["CompiledReport", "RequirementsReport", "BindReport",
           "build_compiled_report", "build_requirements", "build_bind_report"]
