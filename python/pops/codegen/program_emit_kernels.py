"""pops.codegen.program_emit_kernels : shared codegen primitives for the program emitter.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  This is the LEAF module of the program emitter: the
emission-only op tables, the small text helpers (coeff rendering, field combine, aux
component resolution), the model-free per-cell kernels (cell_compare / where) and the
``_PROGRAM_CPP_TEMPLATE``.  ``dsl`` is imported LAZILY inside the helpers that need it.

The model-coefficient per-cell kernels (source / flux / apply / local solves) live in
``program_emit_model_kernels``; the dispatch core in ``program_emit_ops`` /
``program_emit_control``; the schedule wrap in ``program_emit_schedule``; the matrix-free
Krylov emitters in ``program_emit_solve``.  ``program_codegen`` re-imports every name so
its public surface is unchanged.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from pops.identity.scalar import scalar_cpp
from pops.fields._prepared_nullspace_registry import prepared_nullspace_provider_from_attrs
from pops.solvers._prepared_preconditioner_registry import (
    prepared_preconditioner_provider_from_attrs,
)
from pops.solvers.krylov._prepared_method_registry import prepared_krylov_method_provider_from_attrs
from pops.solvers.providers import prepared_hierarchy_solver_provider_from_attrs
from pops.time.values import ProgramValue, _to_affine  # noqa: F401

# Emission-only op tables (formerly Program class constants; the lowering owns them).
# Ops the Phase-4b codegen lowers ONLY when a physical model is supplied (they read the model's
# symbolic source_term / linear_source coefficients). Without a model they raise NotImplementedError.
_MODEL_OPS = (
    "source",
    "apply",
    "local_transform",
    "solve_local_linear",
    "solve_local_nonlinear",
)

_ALLOWED_OPS = frozenset(
    {
        "state",
        "solve_fields",
        "solve_fields_from_blocks",
        "rhs",
        "linear_combine",
        "linear_source",
        "reduce",
        "scalar_op",
        "compare",
        "hmin",
        "max_wave_speed",
        "cfl",
        "while",
        "range",
        "subcycle",
        "branch",
        "synchronize",
        "acceptance_guard",
        "matrix_free_operator",
        "scalar_field",
        "vector_field",
        "laplacian",
        "gradient",
        "divergence",
        "solve_linear",
        "solve_outcome",
        "solve_outcome_component",
        "apply_in",
        "apply_out",
        "history",
        "store_history",
        "fill_boundary",
        "project",
        "record_scalar",
        "record_balance_term",
        "cell_compare",
        "where",
        "rhs_jacvec",
        "apply_laplacian_coeff",
        "condensed_coeffs",
        "condensed_rhs",
        "condensed_reconstruct",
        "condensed_energy",
        "coupled_rate",
        "coupled_rate_out",
        "solve_coupled_implicit",
    }
)

_PROFILE_SKIP_OPS = frozenset({"state", "history", "hmin", "cfl"})

_AUX_OUTPUT_OPS = frozenset({"solve_fields", "solve_fields_from_blocks"})


class ProgramProviderPlans:
    """One immutable local provider ABI per generated Program node.

    A Program consumer does not inherit the model flux plan: it may read a different
    union of auxiliary/field components and its compact local order is the source
    expression's first-use order.  The native registry resolves each key to a late
    storage address after all package providers have been registered.
    """

    def __init__(self) -> None:
        self._plans: dict[str, tuple[tuple[Any, Any], ...]] = {}

    def bind(self, impl: Any, exprs: Any, qid: str) -> dict[str, Any]:
        if not isinstance(qid, str) or not qid:
            raise ValueError("Program provider consumer qid must be a non-empty string")
        pack = getattr(impl, "_auxiliary_provider_pack", None)
        if pack is None:
            raise ValueError(
                "Program provider consumer requires the exact auxiliary ProviderPack"
            )
        from pops._ir.expr import Var
        from pops._ir.visitors import _children

        ordered_names: list[str] = []
        seen_nodes: set[int] = set()

        def visit(value: Any) -> None:
            if id(value) in seen_nodes:
                return
            seen_nodes.add(id(value))
            if isinstance(value, Var) and value.kind == "aux":
                if value.name not in ordered_names:
                    ordered_names.append(value.name)
                return
            for child in _children(value):
                visit(child)

        for expression in exprs:
            visit(expression)
        rows: list[tuple[Any, Any]] = []
        slots: dict[str, int] = {}
        for name in ordered_names:
            matches = [
                key for key in pack
                if key.space_kind in {"aux", "field"} and key.component == name
            ]
            if len(matches) != 1:
                detail = "absent" if not matches else "ambiguous"
                raise ValueError(
                    "Program provider component %r is %s in the exact ProviderPack; "
                    "qualify the expression/provider declaration" % (name, detail)
                )
            key = matches[0]
            contract = pack.contract(key)
            if contract.centering != "cell" or contract.layout != "cell":
                raise ValueError(
                    "Program pointwise provider %s has centering=%r layout=%r; "
                    "declare an explicit projection before use" % (
                        key.space, contract.centering, contract.layout,
                    )
                )
            slots[name] = len(rows)
            rows.append((key, contract))
        frozen = tuple(rows)
        prior = self._plans.get(qid)
        if prior is not None and prior != frozen:
            raise ValueError(
                "Program provider consumer qid %r was emitted with conflicting requirements" % qid
            )
        self._plans[qid] = frozen
        return {"qid": qid, "count": len(rows), "slots": slots}

    def cpp_install(self, target: str) -> str:
        """Emit the registry calls before the Program execution context is installed."""
        if target not in {"system", "amr_system"}:
            raise ValueError("Program provider plan target must be system or amr_system")
        import json

        lines: list[str] = []
        if self._plans:
            lines.extend((
                "  using namespace pops::runtime::system;",
                "  using Key = AuxiliaryComponentKey;",
                "  using Contract = AuxiliaryComponentContract;",
                "  using Shape = AuxiliaryStorageShape<pops::kNativeDimension>;",
                "  using Dependency = AuxiliaryDependency<pops::kNativeDimension>;",
                "  using ConsumerValue = AuxiliaryConsumerValue<pops::kNativeDimension>;",
                "  using ConsumerPlan = AuxiliaryConsumerProviderPlan<pops::kNativeDimension>;",
            ))
        for qid, rows in self._plans.items():
            values = []
            for slot, (key, contract) in enumerate(rows):
                optional_unit = (
                    "std::nullopt" if contract.unit is None
                    else "std::optional<std::string>{%s}" % json.dumps(contract.unit)
                )
                optional_kind = (
                    "std::nullopt" if contract.value_kind is None
                    else "std::optional<std::string>{%s}" % json.dumps(contract.value_kind)
                )
                rendered_key = "Key{%s, %s, %s, %s}" % tuple(
                    json.dumps(value) for value in (
                        key.owner_qid, key.space_kind, key.space_name, key.component,
                    )
                )
                rendered_contract = "Contract{%s, %s, %s, %s, %s}" % (
                    json.dumps(contract.representation), json.dumps(contract.centering), optional_unit,
                    json.dumps(contract.layout), optional_kind,
                )
                shape = (
                    "Shape{pops::kNativeDimension, 1, [] { "
                    "pops::Index<pops::kNativeDimension> halo{}; return halo; }()}"
                )
                values.append(
                    "ConsumerValue{Dependency{%s, %s, %s}, %d}" % (
                        rendered_key, rendered_contract, shape, slot,
                    )
                )
            lines.extend((
                "  sys->install_auxiliary_consumer_plan(ConsumerPlan{%s, " % json.dumps(qid),
                "      std::vector<ConsumerValue>{%s}});" % ", ".join(values),
            ))
        return "\n".join(lines)


def program_provider_consumer_qid(model: Any, value_id: Any) -> str:
    """Return the stable Program-node qid; block names never enter this identity."""
    if isinstance(value_id, bool) or not isinstance(value_id, int) or value_id < 0:
        raise ValueError("Program provider consumer value id must be a non-negative integer")
    impl = _model_impl(model)
    owner = getattr(impl, "owner_path", None)
    canonical = getattr(owner, "canonical", None)
    if not callable(canonical):
        raise ValueError("Program provider consumer model has no canonical owner path")
    return str(canonical()) + "/program/" + str(value_id)


def _prepared_native_components(program: Any) -> tuple[Any, ...]:
    """Return used native components in first-use order after authenticating every provider."""

    def walk(values: Any) -> Any:
        for value in values:
            yield value
            for key in (
                "cond_block",
                "body_block",
                "apply_block",
                "residual_block",
                "true_block",
                "false_block",
            ):
                nested = value.attrs.get(key)
                if isinstance(nested, (list, tuple)):
                    yield from walk(nested)

    components: list[Any] = []
    seen: set[str] = set()
    for value in walk(program._values):
        if value.op != "solve_linear":
            continue
        providers = [
            prepared_krylov_method_provider_from_attrs(value.attrs),
            prepared_preconditioner_provider_from_attrs(value.attrs),
            prepared_nullspace_provider_from_attrs(value.attrs),
        ]
        if "hierarchy_solver_provider" in value.attrs:
            providers.append(prepared_hierarchy_solver_provider_from_attrs(value.attrs))
        for provider in providers:
            component = provider.native_component
            identity = component.manifest_sha256
            if identity not in seen:
                seen.add(identity)
                components.append(component)
    return tuple(components)


def _prepared_native_component_includes(program: Any) -> str:
    """Return entry headers from the typed native components used by prepared providers.

    Provider selection is fully data-driven: no backend name, include root or arbitrary compiler
    flag is known here.
    """
    headers: list[str] = []
    seen: set[str] = set()
    for component in _prepared_native_components(program):
        for header in component.entry_headers:
            if header not in seen:
                seen.add(header)
                headers.append(header)
    return "".join("#include <%s>  // prepared native provider\n" % header for header in headers)


# Ops whose emitted kernels call pops::detail::block_inverse<N> (ADC-637): the GENERIC condensed-implicit
# emitters. A generated .so pulls block_inverse.hpp in ONLY when the IR carries one of these.
# condensed_energy is a pure-kinematic in-place kernel (no block inverse), so it is NOT listed here.
_CONDENSED_OPS = frozenset({"condensed_coeffs", "condensed_rhs", "condensed_reconstruct"})

_BLOCK_INVERSE_INCLUDE = (
    "#include <pops/numerics/linalg/block_inverse.hpp>"
    "  // pops::detail::block_inverse (condensed-implicit solve, ADC-637)\n"
)


def _block_inverse_include(program: Any) -> str:
    """The closed-form block-inverse #include for @p program's generated .so, or "" when it carries no
    condensed-implicit op (ADC-637): only a Program using condensed_* emits pops::detail::block_inverse.
    (block_inverse.hpp itself includes dense_eig.hpp, already pulled in by the template.)"""
    return _BLOCK_INVERSE_INCLUDE if any(v.op in _CONDENSED_OPS for v in program._values) else ""


# --- module-level emission helpers (per-cell kernels, coeff rendering, the .so template) ---
def _deref(tok: Any) -> Any:
    """C++ MultiFab-lvalue argument for a top-level (step-body) field token. Every top-level token is
    already a MultiFab lvalue expression: a state / RHS scratch (``u5``, ``r5``), a history (``h5``) or
    a dereferenced scratch scalar field (``(*sf5)``). The step-body laplacian / gradient / divergence /
    schur ops take exact-ranked ``pops::MultiFab<kNativeDimension>&`` arguments, so the token passes
    through unchanged (the apply-block
    counterpart is `_apply_in_arg`, which additionally const_casts the lambda's ``in`` param)."""
    return tok


def _apply_in_arg(sub: Any, value: Any) -> str:
    """C++ argument for the INPUT field of a laplacian / gradient inside an apply lambda. When the input
    is the lambda's ``in`` (a const&), const_cast it (ctx.laplacian / gradient take a non-const
    exact-ranked MultiFab&
    and only write the ghosts, never the valid cells -- the prepared operator contract);
    a persistent scratch shared_ptr is dereferenced."""
    tok = sub[value.id]
    if tok == "in":
        return "const_cast<pops::MultiFab<pops::kNativeDimension>&>(in)"
    return "*%s" % tok


def _emit_field_combine(result: Any, target: Any, sub: Any, acc: Any, *, dt_symbol: str) -> list:
    """Emit C++ writing the affine combination @p result into the field @p target (a C++ MultiFab token,
    e.g. ``out``). Mirrors the linear_combine commit: zero the PERSISTENT accumulator @p acc (a scratch
    shared_ptr allocated once at install time -- no per-call/per-iteration allocation), accumulate the
    non-`target` terms onto it, then ``ctx.lincomb(target, c_target, target, 1, *acc)``. A single unit
    term that already is the target is a no-op. @p sub maps IR value ids to C++ tokens (``in``/``out``/
    scratch shared_ptrs); @p acc is the install-time accumulator shared_ptr name. Only valid cells
    participate in the algebra, so the reset is a Kokkos valid-region fill; no host cell sweep or
    ghost initialization occurs in the Krylov hot path."""
    aff = _to_affine(result)._merge()
    terms = [(v, c.as_dict()) for v, c in aff]
    lines = ["pops::PureFieldAlgebra::zero_valid(*%s);" % acc]
    c_target = {0: 0}
    for value, coeff in terms:
        tok = sub[value.id]
        ref = (
            "const_cast<pops::MultiFab<pops::kNativeDimension>&>(in)"
            if tok == "in"
            else ("*%s" % tok if tok.startswith("sf") else tok)
        )
        if tok == target:
            c_target = coeff
        else:
            lines.append(
                "pops::PureFieldAlgebra::axpy(*%s, %s, %s);"
                % (acc, _coeff_cpp(coeff, dt_symbol=dt_symbol), ref)
            )
    lines.append(
        "pops::PureFieldAlgebra::lincomb(%s, %s, %s, static_cast<pops::Real>(1), *%s);"
        % (target, _coeff_cpp(c_target, dt_symbol=dt_symbol), target, acc)
    )
    return lines


def _coeff_cpp(powers: Any, *, dt_symbol: str = "dt") -> str:
    """Render an exact dt-polynomial coefficient as a C++ ``pops::Real`` expression.

    Literal kind is preserved until this target-lowering boundary: integer, rational, decimal and
    binary64 inputs get distinct canonical spellings instead of a shared intermediate ``float``.
    In the closure's ``dt`` parameter, ``{1: 1.0}`` -> ``static_cast<pops::Real>(dt)``,
    ``{1: 0.5}`` -> ``static_cast<pops::Real>(0.5 * dt)``, ``{0: 2.0}`` ->
    ``static_cast<pops::Real>(2.0)``. Drops a unit factor and a zero polynomial collapses to 0."""
    if not powers:
        return "static_cast<pops::Real>(0)"
    terms = []
    for power, coeff in sorted(powers.items()):
        factors = [dt_symbol] * int(power)
        if coeff != 1 or not factors:
            factors = [scalar_cpp(coeff)] + factors
        terms.append(" * ".join(factors))
    return "static_cast<pops::Real>(%s)" % " + ".join(terms)


def _coeff_metadata_terms(powers: Any) -> tuple[tuple[int, int, int], ...]:
    """Return one checked exact native-ledger term per nonzero dt monomial."""
    terms = []
    for power, coefficient in sorted(powers.items()):
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise TypeError("AMR conservative coefficient has an invalid dt power")
        value = coefficient.to_python() if hasattr(coefficient, "to_python") else coefficient
        try:
            ratio = Fraction.from_float(value) if isinstance(value, float) else Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise TypeError(
                "AMR conservative coefficient is not an exact rational literal"
            ) from error
        if ratio:
            signed_limit = (1 << 63) - 1
            if (
                power > (1 << 31) - 1
                or ratio.numerator < -signed_limit
                or ratio.numerator > signed_limit
                or ratio.denominator > signed_limit
            ):
                raise OverflowError(
                    "AMR conservative coefficient cannot be represented by the native exact ledger"
                )
            terms.append((power, ratio.numerator, ratio.denominator))
    return tuple(terms)


def _coeff_metadata_cpp(powers: Any) -> str:
    """Render the same dt polynomial as checked exact native-ledger metadata."""
    return "{%s}" % ", ".join("{%d, %d, %d}" % term for term in _coeff_metadata_terms(powers))


# --- Phase-4b: lower a model's split-source / local-linear ops to per-cell C++ kernels ----------
# These helpers emit the body of a for_each_cell kernel over the VALID cells of each local fab. They
# reuse the dsl Expr -> C++ machinery (Var.to_cpp returns the bare name; we bind those names to locals)
# and the existing numerics (pops::detail::mat_inverse). A device kernel must stay heap-free /
# allocation-free: only stack scalars + fixed-size arrays, no std::vector / std::function / Eigen.


def _model_impl(model: Any) -> Any:
    """Return the HyperbolicModel that owns the symbolic coefficients.

    The public physics board wraps the DSL facade as ``_dsl`` and the facade wraps its implementation
    as ``_m``.  Compiler-internal call sites may already carry either of the latter two objects.
    """

    facade = getattr(model, "_dsl", model)
    return getattr(facade, "_m", facade)


def _named_fluxes(v: Any) -> Any:
    """Resolve a ``rhs`` op's ``fluxes`` attr to the list of NAMED fluxes to assemble (ADC-419), or
    ``None`` for the historical default flux path (``ctx.rhs_into`` -- byte-identical -div F). ``None``
    or ``["default"]`` -> default path; a list of named fluxes -> that list. Mixing ``"default"`` with
    named fluxes is rejected (the centered-FV named-flux stencil differs from the Riemann rhs_into
    stencil, so they cannot be summed)."""
    fluxes = v.attrs.get("fluxes")
    if not fluxes or tuple(fluxes) == ("default",):
        return None
    named = [f for f in fluxes if f != "default"]
    if len(named) != len(fluxes):
        raise ValueError(
            "rhs '%s': fluxes mixes 'default' with named fluxes %r; request either the default flux "
            "(-div F via rhs_into) or a set of named fluxes (their -div sum), not both"
            % (v.name, named)
        )
    return named


def _has_runtime_param(exprs: Any) -> bool:
    """True if any of @p exprs reads a RUNTIME parameter (a RuntimeParamRef anywhere in the tree).
    A runtime-param read lowers to ``params.get(<index>)``; the kernel binds a ``params`` local from
    ``ctx.program_params(<block>)`` (ADC-510) so a compiled time Program reads the CURRENT value
    without recompiling (mirror of the AOT-native RuntimeParams member, P7-b)."""
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children

    stack = list(exprs)
    seen = set()
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, RuntimeParamRef):
            return True
        stack.extend(_children(e))
    return False


def _cell_locals(impl: Any, exprs: Any, state_var: Any, *, with_cons: Any, with_prim: Any,
                 provider_binding: Any = None) -> list:
    """C++ local declarations binding the names the @p exprs reference to per-cell values:
      - provider fields -> ``const pops::Real <name> = providers(index, <local-slot>);``;
      - conservative vars -> ``const pops::Real <name> = <state>A(index, <idx>);`` (when @p with_cons);
      - primitives -> their dsl formula, in declaration order, only the LIVE ones (when @p with_prim).
    @p impl is the HyperbolicModel; @p state_var the C++ MultiFab variable (its read FieldView is
    ``<state_var>A`` and the provider view is bound separately below. A runtime-param read lowers to
    ``params.get(idx)``;
    the ``params`` struct is bound by _kernel_open at the fab-loop level (ADC-510), so no per-cell
    binding is emitted here (a runtime param is NOT a per-cell aux/cons local)."""
    from pops._ir.visitors import _children, _dependencies

    deps = _dependencies(exprs)
    lines = []
    live = impl._live_prims(exprs) if with_prim else set()
    # A live primitive's formula (e.g. u = mx / rho) references conservative variables that the top
    # expressions may not name directly: bind those TRANSITIVE cons too, else the emitted prim line
    # references an undeclared local. (Existing source/apply kernels read cons directly, so this only
    # ADDS the cons a live prim pulls in -- it never drops one that was already bound.)
    cons_needed = set(deps)
    for p in live:
        cons_needed |= {d for d in _dependencies(impl.prim_defs[p]) if d in impl.cons_names}
    if with_cons:
        for idx, c in enumerate(impl.cons_names):
            if c in cons_needed:
                lines.append("const pops::Real %s = %sA(index, %d);" % (c, state_var, idx))
    if with_prim:
        for p, expr in impl.prim_defs.items():  # declaration order (a prim may use an earlier prim)
            if p in live:
                lines.append("const pops::Real %s = %s;" % (p, expr.to_cpp()))
    # The ProviderPack plan, not a model-side named component cache, is the sole
    # authority for auxiliary/field values.  Walk typed leaves to distinguish a
    # provider named ``rho`` from the conservative variable ``rho``.
    from pops._ir.expr import Var

    provider_names: set[str] = set()
    seen_nodes: set[int] = set()

    def collect(node: Any) -> None:
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))
        if isinstance(node, Var) and node.kind == "aux":
            provider_names.add(node.name)
            return
        for child in _children(node):
            collect(child)

    for expression in exprs:
        collect(expression)
    used_provider_names = provider_names
    if used_provider_names:
        if provider_binding is None:
            raise ValueError("Program kernel reads providers without one exact consumer plan")
        slots = provider_binding["slots"]
        if not used_provider_names <= set(slots):
            raise ValueError(
                "Program provider plan does not cover its emitted expressions"
            )
        for name in sorted(used_provider_names, key=lambda item: slots[item]):
            lines.append("const pops::Real %s = providers(index, %d);" % (name, slots[name]))
    return lines


def _kernel_open(
    out_var: Any,
    state_var: Any,
    params_block: Any = None,
    *,
    ghost_depth: int = 0,
    provider_binding: Any = None,
    program_block: Any = 0,
) -> list:
    """Open the per-fab loop + per-cell for_each_cell over the VALID cells of @p out_var, binding the
    write handle ``outA``, the read state handle ``<state_var>A`` and the exact local provider view.

    The runtime resolves the consumer's ComponentKeys to immutable group/component addresses once;
    the generated kernel sees only a compact local view for this `li`.  A zero-provider kernel does
    not query the registry or storage at all.

    @p params_block (ADC-510): the PROGRAM block index whose RuntimeParams the kernel reads, or None
    when no formula reads a runtime parameter. When set, bind ``const pops::RuntimeParams params =
    ctx.program_params(<block>);`` at the FAB-LOOP level (a host map lookup, NOT inside the device
    for_each_cell) so the per-cell lambda captures the trivially-copyable struct by value; the lowered
    ``params.get(idx)`` then reads the CURRENT value (no recompile, mirror of the AOT-native member)."""
    if isinstance(ghost_depth, bool) or not isinstance(ghost_depth, int) or ghost_depth < 0:
        raise ValueError("generated kernel ghost depth must be a non-negative integer")
    iteration_box = (
        "%s.box(li)" % out_var
        if ghost_depth == 0
        else "%s.fab(li).box().grow(%d)" % (out_var, ghost_depth)
    )
    lines = [
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> outA = %s.fab(li).view();"
        % out_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> %sA = "
        "std::as_const(%s).fab(li).view();" % (state_var, state_var),
    ]
    if provider_binding is not None:
        count = provider_binding["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Program provider plan count must be a non-negative integer")
        lines.append(
            "  const auto providers = ctx.template provider_values_view<%d>(%s, %d, li);"
            % (count, json.dumps(provider_binding["qid"]), program_block)
        )
    if params_block is not None:
        # Read the per-block RuntimeParams ONCE per fab (host scope), captured by value into the device
        # lambda below (trivially copyable, get() is POPS_HD): the no-recompile runtime-param read.
        lines.append("  const pops::RuntimeParams params = ctx.program_params(%d);" % params_block)
    lines.append(
        "  pops::for_each_cell(%s, [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % iteration_box
    )
    return lines


def _kernel_close() -> list:
    return ["  });", "}"]


# --- per-cell conditional select (spec op 17, ADC-418): model-free for_each_cell kernels --------------
# `cell_compare` and `where` are pure layout ops over co-distributed MultiFabs (no aux / no model
# coefficients): they reuse the same for_each_cell + FieldView per-Fab pattern as the source kernels, but
# bind several read handles and loop over the runtime component count `<out>.ncomp()`. Pairing
# by local fab index li is sound: a cell_compare mask is alloc_scalar_field (the System (ba, dm)), a
# where scratch is scratch_state_like(a) (a's (ba, dm)) and the inputs are the same co-distributed
# states / scalar_fields, so fab(li) is the same box on every rank.


def _emit_cell_compare_kernel(field_var: Any, mask_var: Any, cmp: Any, value: Any) -> list:
    """Lower ``cell_compare``: maskA(index,0) = fieldA(index,0) <cmp> value ? 1 : 0 over the valid cells of
    the 1-component mask. Reads component 0 of @p field_var; writes the 0/1 mask into @p mask_var."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % mask_var,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> maskA = "
        "%s.fab(li).view();" % mask_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> fieldA = "
        "std::as_const(%s).fab(li).view();" % field_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % mask_var,
        "    maskA(index, 0) = (fieldA(index, 0) %s %s) "
        "? static_cast<pops::Real>(1) : static_cast<pops::Real>(0);" % (cmp, scalar_cpp(value)),
        "  });",
        "}",
    ]


def _emit_where_kernel(mask_var: Any, a_var: Any, b_var: Any, out_var: Any) -> list:
    """Lower ``where``: outA(index,c) = maskA(index,mc) != 0 ? aA(index,c) : bA(index,c)
    component-wise over the
    valid cells of @p out_var (out's runtime ncomp). The mask component mc is 0 when the mask is
    1-component (a shared mask) and c when the mask has the SAME ncomp as a/b (a per-component mask) --
    decided per cell from the mask's own ncomp, so both layouts lower with ONE kernel."""
    return [
        "for (int li = 0; li < %s.local_size(); ++li) {" % out_var,
        "  const pops::FieldView<pops::Real, pops::kNativeDimension> outA = "
        "%s.fab(li).view();" % out_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> maskA = "
        "std::as_const(%s).fab(li).view();" % mask_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> aA = "
        "std::as_const(%s).fab(li).view();" % a_var,
        "  const pops::FieldView<const pops::Real, pops::kNativeDimension> bA = "
        "std::as_const(%s).fab(li).view();" % b_var,
        "  const int ncomp_ = %s.ncomp();" % out_var,
        "  const int mask_ncomp_ = %s.ncomp();" % mask_var,
        "  pops::for_each_cell(%s.box(li), [=] POPS_HD("
        "const pops::CellIndex<pops::kNativeDimension>& index) {" % out_var,
        "    for (int c = 0; c < ncomp_; ++c) {",
        "      const int mc = (mask_ncomp_ == 1) ? 0 : c;",
        "      outA(index, c) = (maskA(index, mc) != static_cast<pops::Real>(0)) ? "
        "aA(index, c) : bA(index, c);",
        "    }",
        "  });",
        "}",
    ]


# Source of a generated problem.so. The includes + pops_install_program closure match the shape
# tests/test_program_loader compiles+runs in CI; pops_program_hash is added per the spec .so ABI (a
# cache/restart key) and is not yet consumed by System::install_program. {name} is a JSON-escaped C
# string literal, {hash} the IR hash, {prelude} the INSTALL-TIME C++ (persistent scratch + matrix-free
# apply lambdas, captured into the step closure by [=]), {body} the step-closure body (both already
_PROGRAM_CPP_TEMPLATE = """\
// GENERATED by pops.codegen.program_codegen.emit_cpp_program (epic ADC-399 / ADC-401). Do not edit.
// A compiled time Program installed across the stable .so ABI: it drives sim.step(dt) entirely in
// C++ via the shared Program execution service and its topology provider (no runtime reimplementation).
#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#error "generated Program loaders require the shared runtime exception ABI consumer contract"
#endif
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>
{prepared_native_component_includes}{block_inverse_include}#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/field_view.hpp>     // exact-ranked per-cell handles
#include <pops/mesh/execution/for_each.hpp>     // for_each_cell (Phase-4b per-cell kernels)
#include <pops/numerics/linalg/dense_eig.hpp>   // pops::detail::mat_inverse (local dense solve)
#include <pops/numerics/nonlinear/local_nonlinear_collective.hpp>  // exact failure location
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>  // one prepared local solver
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>  // prepared affine Krylov route
#include <pops/core/foundation/types.hpp>
#include <array>                               // exact-ranked axis packs
#include <chrono>                              // std::chrono::steady_clock (per-node profiling pair, ADC-459)
#include <cmath>                               // std::sqrt / std::fabs / std::pow in lowered formulas
#include <limits>                              // std::numeric_limits (dt_bound +inf sentinel)
#include <functional>                          // per-level AMR persistent Program closures
#include <memory>                              // std::make_shared (persistent matrix-free scratch)
#include <optional>                            // exact ProviderPack contract optional unit/kind
#include <stdexcept>                           // std::runtime_error (AMR install fail-loud, ADC-508)
#include <utility>                             // std::as_const (read-only field views)
#include <vector>                              // pointer list for the coupled multi-block field-solve (ADC-457)

{model_helpers}
extern "C" const char* pops_program_abi_key() {{ return POPS_ABI_KEY_LITERAL; }}
{route_manifest}extern "C" const char* pops_program_name() {{ return {name}; }}
extern "C" const char* pops_program_hash() {{ return "{hash}"; }}
{history_replay_authorities}
{operator_authorities}

{block_names}
{module_metadata}
{program_params}
{field_boundaries}
{system_install}
{amr_install}

// OPTIONAL dt bound (spec s18 / ADC-417). pops_program_has_dt_bound() is true iff the Program set one.
// The target-qualified entry receives the runtime facade, obtains the shared execution provider, and
// returns the lowered scalar bound (min'd into native CFL). With no bound it returns +inf (unreached).
extern "C" bool pops_program_has_dt_bound() {{ return {has_dt_bound}; }}
{dt_bound}
"""
