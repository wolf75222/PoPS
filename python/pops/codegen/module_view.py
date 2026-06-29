"""Codegen view for :class:`pops.model.Module`.

The Program emitters historically read a small "model implementation" protocol:
conservative names, primitive definitions, aux names, flux/source/operator tables
and runtime-parameter routing helpers.  This module implements that protocol
directly from the operator-first ``Module`` registry.  It does not construct a
legacy physics facade or a legacy hyperbolic-model object.
"""

from pops.ir import _wrap
from pops.ir.values import RuntimeParamRef
from pops.ir.visitors import _children
from pops.model import Module
from pops.physics.aux import AUX_CANONICAL, _K_MAX_RUNTIME_PARAMS


def codegen_model(model):
    """Return the object the Program codegen should read for ``model``."""
    if isinstance(model, Module):
        return ModuleCodegenView(model)
    raise TypeError(
        "compile_problem: model must be a pops.model.Module. If you authored with "
        "pops.physics.Model, pass model=physics_model.to_module() / lower(); legacy "
        "facades carrying a private _m are not a valid lowering path.")


def compile_module_for_runtime(module, *, so_path=None, include=None, backend=None,
                               target="system", name=None, cxx=None, std=None,
                               require_metadata=False, hoist_reciprocals=False):
    """Compile a ``Module`` as a native runtime block and package a ``CompiledModel``.

    This is the Module-native counterpart of the old facade ``_compile_for_runtime`` seam:
    it feeds ``ModuleCodegenView`` into the existing C++ brick emitter and returns the
    runtime metadata object ``System.install`` needs.  It never builds a legacy physics facade.
    """
    from pops.codegen.abi import _abi_key_python
    from pops.codegen.backends import BACKEND_DESCRIPTORS, Production, lower_internal_backend
    from pops.codegen.cache import _record_so_backend
    from pops.codegen.compile import compile_model
    from pops.codegen.compile_emit import _BACKENDS, _BACKEND_CAPS
    from pops.codegen.loader import CompiledModel
    from pops.codegen.toolchain import (
        _default_cxx,
        _native_kokkos_compiler,
        loader_cxx_std,
        pops_include,
    )
    view = codegen_model(module)
    backend = lower_internal_backend(backend or Production())
    if backend == "auto":
        raise TypeError(
            "compile_module_for_runtime: backend='auto' was removed; pass Production()")
    if backend not in _BACKENDS:
        raise ValueError("compile_module_for_runtime: unknown backend %r" % (backend,))
    mode, adder = _BACKENDS[backend]
    if target == "amr_system" and mode != "native":
        raise ValueError(
            "compile_module_for_runtime: target='amr_system' requires Production()")
    include = include or pops_include()
    eff_std = std or (loader_cxx_std() if mode == "native" else "c++20")
    kokkos_like = mode in ("native", "compile")
    eff_cxx = _native_kokkos_compiler(cxx) if kokkos_like else _default_cxx(cxx)
    abi_key = _abi_key_python(include, eff_cxx, eff_std)
    model_hash = view._model_hash()
    out_path = compile_model(
        view, so_path=so_path, include=include, backend=BACKEND_DESCRIPTORS[backend](),
        name=name, cxx=cxx, std=std, require_metadata=require_metadata, target=target,
        hoist_reciprocals=hoist_reciprocals)
    _record_so_backend(out_path, backend)
    return CompiledModel(
        so_path=out_path, backend=backend, adder=adder, target=target,
        cons_names=view.cons_names, cons_roles=view.cons_roles, prim_names=view.prim_state,
        n_vars=view.n_vars, gamma=view.gamma, n_aux=view._total_n_aux(),
        params=view.params,
        caps=_BACKEND_CAPS[backend], abi_key=abi_key, model_hash=model_hash, cxx=eff_cxx,
        std=eff_std,
        hllc=view._hllc, roe=(view._roe or view._roe_rows is not None
                              or view._roe_jacobian is not None),
        aux_extra_names=view.aux_extra_names,
        wave_speeds=(view._wave_speeds is not None or view._ws_jacobian is not None
                     or bool(view._eig)),
        elliptic_field_names=list(view._elliptic_fields))


class ModuleCodegenView:
    """A direct, non-public codegen protocol adapter over ``pops.model.Module``."""

    def __init__(self, module):
        self._module = module
        self.name = module.name
        self._state = _single_state(module)
        self.cons_names = list(self._state.components)
        self.cons_roles = _roles(self._state)
        self.prim_defs = {name: _wrap(expr) for name, expr in module.primitive_defs().items()}
        self.prim_roles = None
        self.aux_names = []
        self.aux_extra_names = []
        self._flux = {}
        self._flux_terms = {}
        self._eig = module._eigenvalues
        riemann = module.riemann_metadata()
        self._wave_speeds = _wrap_wave_speeds(riemann.get("wave_speeds"))
        self._ws_jacobian = None
        self._source = None
        self._source_terms = {}
        self._linear_sources = {}
        self._elliptic = None
        self._elliptic_fields = {}
        self._proj = None
        self._stab_speed = None
        self._stab_dt = None
        self._src_freq = None
        self._src_jac = None
        self._hllc = bool(riemann.get("hllc", False))
        self._riemann_hook_forms = {name: _wrap(expr)
                                    for name, expr in riemann.get("hooks", {}).items()}
        self._roe = bool(riemann.get("roe", False))
        self._roe_rows = None
        self._roe_jacobian = None
        self._rate_operators = {}
        self.params = module.params()
        self.gamma = None
        self.prim_state = list(self.cons_names)
        self.cons_from = [_wrap_var(nm) for nm in self.cons_names]
        self._collect_aux()
        self._collect_operators()

    @property
    def n_vars(self):
        return len(self.cons_names)

    def _collect_aux(self):
        names = []
        for fields in self._module.field_spaces().values():
            names.extend(fields.components)
        names.extend(a.name for a in self._module.aux().values())
        for name in names:
            if name in AUX_CANONICAL:
                if name not in self.aux_names:
                    self.aux_names.append(name)
            elif name not in self.aux_extra_names:
                self.aux_extra_names.append(name)

    def _collect_operators(self):
        field_default_taken = False
        for op in self._module.operator_registry():
            body = op.body
            if _needs_ir_body(op.kind) and (body is None or callable(body)):
                raise ValueError(
                    "compile_problem: operator %r (%s) has no IR body; a compilable Module "
                    "operator needs an expression body" % (op.name, op.kind))
            if op.kind == "grid_operator":
                flux = _flux_body(body, op.name)
                if _is_default_op(op, "flux"):
                    self._flux = flux
                else:
                    self._flux_terms[op.name] = flux
            elif op.kind == "local_source":
                exprs = _expr_list(body)
                if _is_default_op(op, "source"):
                    self._source = exprs
                else:
                    self._source_terms[op.name] = exprs
            elif op.kind == "local_linear_operator":
                self._linear_sources[op.name] = [_expr_list(row) for row in body]
            elif op.kind == "field_operator":
                rhs = _wrap(body)
                if _is_default_op(op, "fields") and not field_default_taken:
                    self._elliptic = rhs
                    field_default_taken = True
                else:
                    self._elliptic_fields[op.name] = {
                        "rhs": rhs,
                        "operator": op.requirements.get("elliptic_operator", "poisson"),
                        "aux": list(getattr(op.signature.output, "components", ()) or (op.name,)),
                    }
            elif op.kind == "projection":
                self._proj = _expr_list(body)
            elif op.kind == "local_rate":
                self._rate_operators[op.name] = dict(op.lowering)

    def _all_exprs(self):
        out = list(self.prim_defs.values())
        for direction in ("x", "y"):
            out += list(self._flux.get(direction, []))
            if self._eig is not None:
                out += list(self._eig.get(direction, []))
            if self._wave_speeds is not None:
                out += list(self._wave_speeds.get(direction, []))
        for term in self._flux_terms.values():
            out += list(term.get("x", [])) + list(term.get("y", []))
        if self._source is not None:
            out += list(self._source)
        for exprs in self._source_terms.values():
            out += list(exprs)
        for rows in self._linear_sources.values():
            for row in rows:
                out += list(row)
        if self._elliptic is not None:
            out.append(self._elliptic)
        for spec in self._elliptic_fields.values():
            out.append(spec["rhs"])
        if self._proj is not None:
            out += list(self._proj)
        out += list(self._riemann_hook_forms.values())
        return out

    def _aux_locals_lines(self):
        lines = ["    const pops::Real %s = a.%s;" % (name, name)
                 for name in self.aux_names]
        lines += ["    const pops::Real %s = a.extra_field(%d);" % (name, idx)
                  for idx, name in enumerate(self.aux_extra_names)]
        return lines

    def _reads_aux(self):
        return bool(self.aux_names or self.aux_extra_names)

    def _total_n_aux(self):
        width = 3
        for name in self.aux_names:
            width = max(width, AUX_CANONICAL[name] + 1)
        if self.aux_extra_names:
            width = max(width, 5 + len(self.aux_extra_names))
        return width

    def runtime_param_nodes(self):
        seen = {}

        def walk(expr):
            if isinstance(expr, RuntimeParamRef):
                seen.setdefault(expr.name, expr)
                return
            for child in _children(expr):
                walk(child)

        for expr in self._all_exprs():
            walk(expr)
        return [seen[name] for name in sorted(seen)]

    def assign_runtime_indices(self):
        nodes = self.runtime_param_nodes()
        if len(nodes) > _K_MAX_RUNTIME_PARAMS:
            raise ValueError(
                "module %r: %d runtime parameters > kMaxRuntimeParams bound=%d"
                % (self.name, len(nodes), _K_MAX_RUNTIME_PARAMS))
        for idx, node in enumerate(nodes):
            node.index = idx
        return nodes

    def has_runtime_params(self):
        return bool(self.runtime_param_nodes())

    def _live_prims(self, exprs, seed=()):
        return set()

    def _runtime_params_member(self):
        nodes = self.assign_runtime_indices()
        if not nodes:
            return ""
        values = ", ".join(repr(node.value) for node in nodes)
        return "  pops::RuntimeParams params{%d, {%s}};\n" % (len(nodes), values)

    def _check_require_metadata(self, require_metadata, backend):
        if not require_metadata:
            return
        missing = []
        if all(role in (None, "Custom") for role in self.cons_roles):
            missing.append("physical roles")
        if self.gamma is None:
            missing.append("gamma")
        if missing:
            raise ValueError(
                "compile(require_metadata=True): module %r does not provide %s"
                % (self.name, " nor ".join(missing)))

    def _model_hash(self, params=None):
        from pops.codegen.compile_emit import model_hash
        return model_hash(self, params=self.params if params is None else params)

    def _validate_hook_form(self, hook, form, allow_aux=True):
        known = set(self.cons_names) | set(self.prim_defs)
        if allow_aux:
            known |= set(self.aux_names) | set(self.aux_extra_names)
        missing = sorted(form.deps() - known)
        if missing:
            raise ValueError(
                "riemann hook %r references undeclared quantity %s" % (hook, missing))


def _single_state(module):
    states = module.state_spaces()
    if len(states) != 1:
        raise ValueError(
            "compile_problem: a Module must declare exactly one StateSpace for the current "
            "single-block Program ABI (got %s)" % sorted(states))
    return next(iter(states.values()))


def _roles(state):
    mapping = {"density": "Density", "momentum_x": "MomentumX", "momentum_y": "MomentumY",
               "momentum_z": "MomentumZ", "energy": "Energy", "pressure": "Pressure",
               "velocity_x": "VelocityX", "velocity_y": "VelocityY", "velocity_z": "VelocityZ",
               "temperature": "Temperature"}
    roles = []
    for component in state.components:
        role = state.roles.get(component)
        roles.append(mapping.get(role, role) if role is not None else "Custom")
    return roles


def _needs_ir_body(kind):
    return kind in ("grid_operator", "local_source", "local_linear_operator",
                    "field_operator", "projection")


def _is_default_op(op, family):
    if op.capabilities.get("default"):
        return True
    if family == "flux":
        return op.name in ("flux", "flux_default")
    if family == "source":
        return op.name in ("source", "source_default", "default")
    if family == "fields":
        return op.name in ("fields", "fields_from_state")
    return False


def _expr_list(values):
    return [_wrap(v) for v in values]


def _flux_body(body, name):
    if not isinstance(body, dict) or "x" not in body or "y" not in body:
        raise ValueError("grid_operator %r needs body {'x': [...], 'y': [...]}" % name)
    return {"x": _expr_list(body["x"]), "y": _expr_list(body["y"])}


def _wrap_var(name):
    from pops.ir.expr import Var
    return Var(name, "cons")


def _wrap_wave_speeds(waves):
    if waves is None:
        return None
    return {key: [_wrap(value) for value in waves.get(key, [])] for key in ("x", "y")}
