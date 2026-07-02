"""CompiledReport: the print(compiled) summary (ADC-619 split).

The :class:`CompiledReport` value class and its pure builders (``build_compiled_report``
/ ``_compiled_options``) plus the hash helpers (``_short`` / ``_abi_token``). Split
out of ``pops.codegen.inspect_report`` for the 500-line cap;
``pops.codegen.inspect_report`` re-exports every name so the historical
``from pops.codegen.inspect_report import CompiledReport, build_compiled_report``
paths stay unchanged. Nothing here compiles, binds, dlopens or allocates: it reads
Python-side metadata only. ``pops.time`` / the runtime are imported lazily to keep
the codegen import graph acyclic (cf. tests/python/architecture/test_import_graph.py).
"""

import json


def _short(value, width=12):
    """A short prefix of a hash-like string, or ``"none"`` when absent."""
    return (value or "")[:width] or "none"


def _abi_token(abi_key, name):
    prefix = name + "="
    for part in str(abi_key or "").split(";"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


class CompiledReport:
    """The printable ``print(compiled)`` summary of a compiled artifact (Spec 5 sec.12.1).

    A plain, inert record AGGREGATING the metadata the :class:`CompiledProblem` carries -- it
    computes nothing of its own. :meth:`to_dict` is a JSON-ready view; :meth:`__str__` is the
    deterministic, array-free, multi-line report shaped like the Spec 5 sec.12.1 example. It never
    prints the ``.so`` contents, a field array, or a ``<...object at 0x...>`` repr.
    """

    def __init__(self, *, name, backend, platform, layout, blocks, fields, program, inputs,
                 artifacts, status, env=None, runtime=None, capabilities=None, options=None,
                 module_manifest=None):
        self.name = name
        self.backend = backend
        self.platform = platform
        self.layout = layout
        self.blocks = list(blocks)        # [{name, state, components, spatial}]
        self.fields = list(fields)        # [{name, solver}]
        self.program = dict(program)      # {name, commits, ops, hash}
        self.inputs = dict(inputs)        # {states, params, aux} -> [names]
        self.artifacts = dict(artifacts)  # {so_path, abi_key, cache_key}
        self.status = status
        # Active codegen POPS_* environment (Spec 5 sec.12.4, #47-48): the resolved CodegenEnv as a
        # plain dict (log_level / codegen_dir / keep_generated / dump_ir / dump_cpp / cache_dir /
        # profile / autotune / jit_backdoor), or {} when the handle carried no env snapshot. Surfaced
        # so the env state that governed the compile -- including the UNSAFE jit_backdoor gate -- is
        # inspectable, never hidden.
        self.env = dict(env) if env else {}
        self.runtime = dict(runtime) if runtime else {}
        self.capabilities = dict(capabilities) if capabilities else {}
        self.options = dict(options) if options else {}
        # The operator-first Module manifest (ADC-585): the JSON-ready dict of the resolved model's
        # Module (spaces / params / aux / typed operators / native routes), or None when the artifact
        # carries a bare dsl.Model with no backing Module -- absent, never fabricated.
        self.module_manifest = dict(module_manifest) if module_manifest else None

    def to_dict(self):
        """A plain-dict view of the whole report (JSON-ready)."""
        return {"name": self.name, "backend": self.backend, "platform": self.platform,
                "layout": self.layout, "blocks": [dict(b) for b in self.blocks],
                "fields": [dict(f) for f in self.fields], "program": dict(self.program),
                "inputs": {k: list(v) for k, v in self.inputs.items()},
                "artifacts": dict(self.artifacts), "status": self.status,
                "env": dict(self.env), "runtime": dict(self.runtime),
                "capabilities": dict(self.capabilities), "options": dict(self.options),
                "module_manifest": dict(self.module_manifest) if self.module_manifest else None}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self):
        lines = ["compiled problem %r" % self.name]
        lines.append("  backend  : %s" % self.backend)
        lines.append("  platform : %s" % self.platform)
        lines.append("  layout   : %s" % self.layout)
        lines.append("  blocks   :")
        for block in self.blocks:
            lines.append("    %-14s state=%s components=%s spatial=%s"
                         % (block.get("name"), block.get("state"), block.get("components"),
                            block.get("spatial")))
        lines.append("  fields   :")
        if self.fields:
            for field in self.fields:
                lines.append("    %-14s solver=%s" % (field.get("name"), field.get("solver")))
        else:
            lines.append("    (none)")
        prog = self.program
        lines.append("  program  : %s (%s ops, commits=%s)"
                     % (prog.get("name"), prog.get("ops"), prog.get("commits")))
        lines.append("  required runtime inputs:")
        lines.append("    states : %s" % (", ".join(self.inputs.get("states", [])) or "(none)"))
        lines.append("    params : %s" % (", ".join(self.inputs.get("params", [])) or "(none)"))
        lines.append("    aux    : %s" % (", ".join(self.inputs.get("aux", [])) or "(none)"))
        art = self.artifacts
        lines.append("  artifacts:")
        lines.append("    so_path  : %s" % art.get("so_path"))
        lines.append("    abi_key  : %s" % art.get("abi_key"))
        lines.append("    cache_key: %s" % art.get("cache_key"))
        if self.runtime:
            lines.append("  runtime:")
            lines.append("    dimension             : %s" % self.runtime.get("dimension"))
            lines.append("    amr_refinement_ratio  : %s"
                         % self.runtime.get("amr_refinement_ratio"))
            lines.append("    precision             : %s (%s bytes)"
                         % (self.runtime.get("precision"), self.runtime.get("real_bytes")))
            lines.append("    communicator          : %s"
                         % self.runtime.get("communicator"))
            lines.append("    custom_communicator   : %s"
                         % self.runtime.get("supports_custom_communicator"))
        if self.capabilities:
            routes = self.capabilities.get("routes", [])
            blocked = [r for r in routes if r.get("status") != "available"]
            lines.append("  capabilities:")
            lines.append("    schema_version : %s" % self.capabilities.get("schema_version"))
            lines.append("    abi_version    : %s" % self.capabilities.get("abi_version"))
            lines.append("    route_ids      : %d (%d partial/unavailable)"
                         % (len(routes), len(blocked)))
        if self.options:
            cache = self.options.get("cache_key", {})
            lines.append("  options:")
            lines.append("    defaults_schema : %s"
                         % self.options.get("defaults", {}).get("schema_version"))
            lines.append("    cache_key       : %s" % cache.get("cache_key"))
            lines.append("    const_params    : %s"
                         % (", ".join(cache.get("const_params", [])) or "(none)"))
            lines.append("    runtime_params  : %s"
                         % (", ".join(cache.get("runtime_params", [])) or "(none)"))
        if self.env:
            lines.append("  environment (active POPS_*):")
            lines.append("    log_level     : %s" % self.env.get("log_level"))
            lines.append("    codegen_dir   : %s" % self.env.get("codegen_dir"))
            lines.append("    keep_generated: %s" % self.env.get("keep_generated"))
            lines.append("    dump_ir       : %s" % self.env.get("dump_ir"))
            lines.append("    dump_cpp      : %s" % self.env.get("dump_cpp"))
            lines.append("    cache_dir     : %s" % self.env.get("cache_dir"))
            lines.append("    profile       : %s" % self.env.get("profile"))
            lines.append("    autotune      : %s%s"
                         % (self.env.get("autotune"),
                            "  (no-op stub: no autotune engine today)"
                            if self.env.get("autotune") not in (None, "off") else ""))
            backdoor = self.env.get("jit_backdoor")
            lines.append("    jit_backdoor  : %s%s"
                         % (backdoor, "  *** UNSAFE debug gate ENABLED ***" if backdoor else ""))
        if self.module_manifest:
            manifest = self.module_manifest
            ops = manifest.get("operators", [])
            lines.append("  module manifest (ADC-585):")
            lines.append("    schema_version : %s" % manifest.get("schema_version"))
            lines.append("    name           : %s" % manifest.get("name"))
            lines.append("    state_spaces   : %s"
                         % (", ".join(sorted(manifest.get("state_spaces", {}))) or "(none)"))
            lines.append("    operators      : %s"
                         % (", ".join(op.get("name") for op in ops) or "(none)"))
        lines.append("  status   : %s" % self.status)
        return "\n".join(lines)

    def __repr__(self):
        return ("CompiledReport(name=%r, backend=%r, blocks=%d, fields=%d)"
                % (self.name, self.backend, len(self.blocks), len(self.fields)))


def build_compiled_report(compiled):
    """Build the :class:`CompiledReport` of a compiled artifact (sec.12.1).

    AGGREGATES the metadata already carried (no compile / bind / runtime read):

      - name / program: ``program_name``, the committed blocks, the op count, a short IR hash;
      - backend: a compiled time Program is always the ``production`` codegen backend (the only
        backend ``compile_problem`` emits a Program for);
      - platform / layout: read from :meth:`CompiledProblem.arguments` (the layout the artifact
        targets and whether it supports MPI);
      - blocks: the committed blocks, each with the model's state space + component count;
      - fields: the elliptic / Krylov field solves in the IR (the solver brick is a bind input);
      - required runtime inputs: the REQUIRED ``arguments()`` entries (states / runtime params /
        aux), the human counterpart of the machine-readable :meth:`CompiledProblem.arguments`;
      - artifacts: ``so_path`` + short ``abi_key`` / ``cache_key``;
      - status: the always-true ``"compiled, waiting for pops.bind(...)"`` bind-pending line.
    """
    args = compiled.arguments()
    instances = getattr(args, "instances", {})
    solvers = getattr(args, "solvers", {})
    layout_runtime = getattr(args, "layout_runtime", {})

    blocks = [{"name": name, "state": spec.get("state"),
               "components": spec.get("components"), "spatial": "bind-time"}
              for name, spec in sorted(instances.items())]

    fields = [{"name": name, "solver": spec.get("solver")}
              for name, spec in sorted(solvers.items())]

    states = [name for name, spec in sorted(instances.items()) if spec.get("required")]
    req_params = [name for name, spec in sorted(getattr(args, "params", {}).items())
                  if spec.get("required")]
    req_aux = [name for name, spec in sorted(getattr(args, "aux", {}).items())
               if spec.get("required")]

    program = getattr(compiled, "program", None)
    prog_summary = {
        "name": getattr(compiled, "program_name", None) or "problem",
        "ops": len(getattr(program, "_values", [])) if program is not None else 0,
        "commits": sorted(program.commits()) if (program is not None
                                                  and hasattr(program, "commits")) else [],
        "hash": _short(getattr(compiled, "program_hash", None)),
    }

    platform = "mpi" if layout_runtime.get("supports_mpi") else "serial"
    layout = layout_runtime.get("layout", "system")
    from pops.runtime_environment import compiled_runtime_facts
    runtime = compiled_runtime_facts(supports_mpi=layout_runtime.get("supports_mpi"))

    abi_key = getattr(compiled, "abi_key", None)
    artifacts = {"so_path": getattr(compiled, "so_path", None),
                 "abi_key": _short(abi_key),
                 "abi_key_full": abi_key,
                 "header_signature": _abi_token(abi_key, "headers") or "unknown",
                 "cache_key": _short(getattr(compiled, "cache_key", None))}
    from pops._capabilities import native_capability_report
    try:
        capability_report = native_capability_report(
            flags=compiled.manifest().supports(), source="manifest").to_dict()
    except Exception:
        capability_report = {}

    # The active codegen POPS_* environment snapshot (sec.12.4, #47-48): the resolved CodegenEnv as a
    # plain dict, or {} for a handle that carries none. Surfacing it keeps the env state -- including
    # the UNSAFE jit_backdoor gate -- inspectable rather than hidden.
    codegen_env = getattr(compiled, "codegen_env", None)
    env = codegen_env.to_dict() if codegen_env is not None else {}

    # The operator-first Module manifest (ADC-585), when the artifact carries a backing Module.
    manifest = getattr(compiled, "module_manifest", None)
    module_manifest = manifest.to_dict() if manifest is not None else None

    return CompiledReport(
        name=prog_summary["name"], backend="production", platform=platform, layout=layout,
        blocks=blocks, fields=fields, program=prog_summary,
        inputs={"states": states, "params": req_params, "aux": req_aux},
        artifacts=artifacts, status="compiled, waiting for pops.bind(...)", env=env,
        runtime=runtime, capabilities=capability_report, options=_compiled_options(compiled),
        module_manifest=module_manifest)


def _compiled_options(compiled):
    """Effective defaults/options visible before bind; inert metadata-only."""
    from pops.runtime.defaults import PHYSICAL_DEFAULT_GAMMA, numerical_defaults_report

    defaults = numerical_defaults_report()
    model = getattr(compiled, "model", None)
    params = dict(getattr(model, "params", {}) or {})
    const_params = sorted(
        name for name, param in params.items() if getattr(param, "kind", "const") != "runtime")
    runtime_params = sorted(
        name for name, param in params.items() if getattr(param, "kind", "const") == "runtime")

    default_gamma = defaults.get("physical", {}).get("gamma", PHYSICAL_DEFAULT_GAMMA)
    model_gamma = getattr(model, "gamma", None)
    gamma_source = "compiled_model_metadata" if model_gamma is not None else "legacy_fallback"
    gamma_value = model_gamma if model_gamma is not None else default_gamma

    param_rows = []
    for name in sorted(params):
        param = params[name]
        kind = getattr(param, "kind", "const")
        param_rows.append({
            "name": name,
            "kind": kind,
            "value": getattr(param, "value", None),
            "affects_cache_key": kind != "runtime",
        })

    return {
        "schema_version": 1,
        "defaults": defaults,
        "physical": {
            "gamma": {
                "value": gamma_value,
                "source": gamma_source,
                "affects_cache_key": model_gamma is not None,
            },
            "params": param_rows,
        },
        "cache_key": {
            "cache_key": getattr(compiled, "cache_key", None),
            "problem_hash": getattr(compiled, "problem_hash", None),
            "program_hash": getattr(compiled, "program_hash", None),
            "model_hash": getattr(model, "model_hash", None),
            "abi_key": getattr(compiled, "abi_key", None),
            "participates": [
                "program_source",
                "model_hash",
                "abi_key",
                "compiler",
                "cxx_standard",
                "const_params",
                "route_registry",
                "capability_vocab",
                "platform",
            ],
            "const_params": const_params,
            "runtime_params": runtime_params,
            "runtime_params_affect_cache_key": False,
            # Route registry / report vocabulary components (ADC-599): the native catalog the
            # artifact was keyed against. A registry change (route added/removed/re-tokenized)
            # is a cache MISS; these fields make the participating identity inspectable.
            "route_registry": _route_registry_components(),
        },
    }


def _route_registry_components():
    """The route-registry / vocabulary cache-key components (ADC-599), inspectable."""
    from pops.runtime.routes import (CAPABILITY_VOCAB_VERSION, ROUTE_REGISTRY_VERSION,
                                     route_registry_hash, route_registry_signature)
    return {
        "version": ROUTE_REGISTRY_VERSION,
        "hash": route_registry_hash(),
        "signature": route_registry_signature(),
        "capability_vocab_version": CAPABILITY_VOCAB_VERSION,
    }
