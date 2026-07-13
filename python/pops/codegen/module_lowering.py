"""pops.codegen.module_lowering : lower a pops.model.Module to a dsl.Model for kernel emission.

The operator-first :class:`pops.model.Module` is the canonical compile IR (ADC-557); the kernel
emitters still consume a :class:`pops.dsl.Model`, so ``compile_problem`` lowers the Module to a
dsl.Model INTERNALLY. That translation -- a re-expression of the SAME physics, not a second backend
-- lives here, out of ``compile_drivers`` so both stay under the 500-line budget.

``_module_to_model`` embeds the single structural validation of a compilable Module (exactly one
StateSpace, an expression body for every codegen operator, at most one field_operator). That IS the
compile pipeline's one validation (ADC-557): there is no second ``model.check()`` path. Imported
lazily by ``compile_problem`` to avoid a top-level physics import.

``lower_and_validate`` is the SINGLE entry the compile pipeline calls (ADC-557): it validates the
model ONCE and returns the ``(emit_model, source_module)`` pair -- the dsl model the kernel emitters
consume plus the operator-first Module that is the canonical compile-IR authority (the trace shown in
``compiled.inspect()`` and the hash bind drifts against). A lowering error is remapped onto the
facade handles the user actually wrote via ``remap_lowering_error``.
"""


from __future__ import annotations

from types import MappingProxyType
from typing import Any

from .lowering_coverage import (
    LoweringCoverageReport,
    LoweringCoverageRow,
    LoweringRejection,
)


def _module_to_model(module: Any, state_space: Any = None) -> Any:
    """Lower a :class:`pops.model.Module` to a :class:`pops.dsl.Model`
    (Spec 2, S2-11), reusing the dsl codegen engine -- a translation, NOT a
    second backend.  The Module's typed operators carry dsl ``Expr`` bodies;
    each is mapped to the dsl method of its kind.

    Imported lazily by compile_problem to avoid a top-level physics import.
    """
    # Import the model facade + aux constants lazily here (called only at
    # compile_problem time, not at import time).
    from pops.physics._facade import Model  # noqa: PLC0415
    from pops.physics.aux import AUX_CANONICAL  # noqa: PLC0415
    from pops.model.operators import OPERATOR_KINDS  # noqa: PLC0415
    coverage_rows = [LoweringCoverageRow(
        "module:%s:metadata" % module.name, "documentary")]

    def _reject(source: str, gate: str, message: str) -> None:
        report = LoweringCoverageReport([
            *coverage_rows,
            LoweringCoverageRow(source, "rejected", gate=gate),
        ])
        raise LoweringRejection(
            message, coverage_report=report, source=source, gate=gate)

    states = module.state_spaces()
    if len(states) == 1:
        state = next(iter(states.values()))
    elif isinstance(state_space, str) and state_space in states:
        state = states[state_space]
    else:
        _reject(
            "module:%s:state_spaces" % module.name,
            "state_space_route_required",
            "compile_problem: a multi-StateSpace Module must be lowered for an exact block/state "
            "route; requested %r, declared %s" % (state_space, sorted(states)))
    from pops.model.state_symbols import rebind_state_symbols  # noqa: PLC0415

    def _body_for_state(body: Any) -> Any:
        return rebind_state_symbols(body, state, states.values())

    m = Model(module.name)
    # Preserve the canonical source-Module identity across the internal facade lowering. The
    # resulting CompiledModel authenticates this scalar hash; it never retains ``module`` itself.
    object.__setattr__(m, "_compile_source_module_hash", module.module_hash())
    from pops.model.provider_pack import (  # noqa: PLC0415
        build_operator_provider_pack,
        build_provider_pack,
    )

    provider_pack = build_provider_pack(module)
    object.__setattr__(m, "_component_provider_pack", provider_pack)
    object.__setattr__(m, "_component_provider_metadata", provider_pack.to_data())
    operator_provider_packs = {
        operator.name: build_operator_provider_pack(module, operator)
        for operator in module.operator_registry()
    }
    object.__setattr__(m, "_component_operator_provider_packs",
                       MappingProxyType(operator_provider_packs))
    object.__setattr__(m, "_component_operator_provider_metadata", MappingProxyType({
        name: pack.to_data() for name, pack in operator_provider_packs.items()
    }))
    flux_keys = []
    for operator in module.operator_registry():
        if operator.kind == "grid_operator":
            flux_keys.extend(operator_provider_packs[operator.name])
    flux_provider_pack = provider_pack.select(flux_keys)
    object.__setattr__(m, "_component_flux_provider_pack", flux_provider_pack)
    object.__setattr__(m, "_component_flux_provider_metadata", flux_provider_pack.to_data())
    # The facade is a lowering view of THIS Module, not a newly declared model. Re-anchor its empty
    # backing model before the first declaration so every derived operator registry retains the
    # Module's exact authoring authority. Without this, owner-qualified Program nodes would be
    # rejected (correctly) as belonging to a different model during codegen.
    object.__setattr__(m._m, "_owner_path", module.owner_path)
    m._m._invalidate_authoring_views()
    # This is a lowering view, not a second declaration authority.  Reuse the
    # Module's registry itself so every RuntimeParamRef and every report keeps
    # the exact ParamHandle identity authored by the user.
    registry = module.param_registry()
    if registry.owner_path != module.owner_path:
        raise ValueError("compile_problem: Module ParamRegistry owner drift")
    object.__setattr__(m, "_param_registry", registry)
    _spec_role = {"density": "Density", "momentum_x": "MomentumX", "momentum_y": "MomentumY",
                  "momentum_z": "MomentumZ", "energy": "Energy", "pressure": "Pressure",
                  "velocity_x": "VelocityX", "velocity_y": "VelocityY", "velocity_z": "VelocityZ",
                  "temperature": "Temperature"}
    roles = None
    if state.roles:
        roles = [_spec_role.get(state.roles.get(c)) for c in state.components]
        if all(r is None for r in roles):
            roles = None
    cvars = m.conservative_vars(*state.components, roles=roles)
    coverage_rows.append(LoweringCoverageRow(
        "state_space:%s" % state.name, "lowered",
        ("dsl:state_space:%s" % state.name,
         *("dsl:conservative:%s" % component for component in state.components))))
    m.primitive_vars(*cvars)
    for component in state.components:
        coverage_rows.append(LoweringCoverageRow(
            "derived:primitive:%s" % component, "derived",
            ("dsl:primitive:%s" % component,),
            rule="default primitive variables are derived from conservative state"))
    m.conservative_from(list(cvars))
    coverage_rows.append(LoweringCoverageRow(
        "derived:conservative_from", "derived", ("dsl:conservative_from",),
        rule="default conservative reconstruction is the declared conservative state"))
    for declaration in module.params().values():
        if registry.handle(declaration) != module.param_handle(declaration):
            raise ValueError("compile_problem: Module parameter authority is inconsistent")
        if declaration.name == "gamma":
            from pops.params import ConstParam

            if not isinstance(declaration, ConstParam):
                raise ValueError(
                    "compile_problem: EOS metadata parameter 'gamma' must be a ConstParam"
                )
            m._m.set_gamma(declaration.value)
        handle = module.param_handle(declaration)
        targets = ["dsl:param_registry:%s" % handle.qualified_id]
        if declaration.name == "gamma":
            targets.append("dsl:eos:gamma")
        coverage_rows.append(LoweringCoverageRow(
            "parameter:%s" % handle.qualified_id, "lowered", tuple(targets)))
    declared = {}

    def _declare_aux(nm: Any, key: Any) -> None:
        previous = declared.get(nm)
        if previous is not None and previous != key:
            raise ValueError(
                "compile_problem: typed components %s and %s both lower to legacy aux name %r; "
                "the spaces are distinct and cannot be merged silently" % (previous, key, nm))
        if previous is not None:
            return
        declared[nm] = key
        if nm in AUX_CANONICAL:
            m.aux(nm)
        else:
            m.aux_field(nm)

    for fs in module.field_spaces().values():
        targets = ["dsl:field_space:%s" % fs.name]
        for comp in fs.components:
            _declare_aux(comp, "field/%s/%s" % (fs.name, comp))
            targets.append("dsl:aux:%s" % comp)
        coverage_rows.append(LoweringCoverageRow(
            "field_space:%s" % fs.name, "lowered", tuple(targets)))
    for a in module.aux().values():
        _declare_aux(a.name, "aux/%s/%s" % (a.name, a.name))
        coverage_rows.append(LoweringCoverageRow(
            "aux:%s" % a.name, "lowered", ("dsl:aux:%s" % a.name,)))
    if module._eigenvalues is not None:
        m.eigenvalues(
            x=_body_for_state(module._eigenvalues["x"]),
            y=_body_for_state(module._eigenvalues["y"]),
        )
        coverage_rows.append(LoweringCoverageRow(
            "module:%s:eigenvalues" % module.name, "lowered", ("dsl:eigenvalues",)))
    else:
        coverage_rows.append(LoweringCoverageRow(
            "module:%s:eigenvalues" % module.name, "documentary"))

    for key in provider_pack:
        key_data = key.to_data()
        stable_key = "%s/%s/%s" % (
            key_data["space_kind"], key_data["space_name"], key_data["component"])
        coverage_rows.append(LoweringCoverageRow(
            "provider:%s" % stable_key, "lowered",
            ("component_provider_pack:%s" % stable_key,)))
    _CODEGEN_KINDS = ("grid_operator", "local_source", "local_linear_operator", "field_operator",
                      "projection")
    # The native HyperbolicModel concept always needs one base flux, even when a typed Program reads
    # only a named grid operator.  For one exact StateSpace route, a sole named flux selected by its
    # local_rate is unambiguous: install it both as the concept's base flux and under its authored
    # name.  This is a lowering fact, not an authoring alias; multiple candidates remain fail-closed.
    operators = tuple(module.operator_registry())
    applicable_rates = tuple(
        op for op in operators if op.kind == "local_rate" and state in op.signature.inputs)
    routed_fluxes = {
        flux_name
        for op in applicable_rates
        for flux_name in (op.lowering.get("fluxes") or ())
    }
    applicable_grid_names = {
        op.name for op in operators
        if op.kind == "grid_operator" and (
            not tuple(item for item in op.signature.inputs
                      if getattr(item, "kind", None) == "state")
            or state in op.signature.inputs)
    }
    explicit_default = applicable_grid_names & {"flux", "flux_default"}
    fallback_default = None
    if not explicit_default and len(routed_fluxes) == 1:
        candidate = next(iter(routed_fluxes))
        if candidate in applicable_grid_names:
            fallback_default = candidate
    # ADC-642: one decode -- a {kind: builder} dispatch over the shared OPERATOR_KINDS vocabulary.
    # Each builder holds its arm body verbatim. _CODEGEN_KINDS is the body-requirement set (local_rate
    # lowers from op.lowering, not a body, so it stays out); the assert makes an unwired kind loud.
    def _b_grid_operator(op: Any) -> None:
        body = _body_for_state(op.body)
        if op.name in ("flux", "flux_default"):
            m.flux(x=body["x"], y=body["y"])
        elif op.name == fallback_default:
            m.flux(x=body["x"], y=body["y"])
            m.flux_term(op.name, x=body["x"], y=body["y"])
        else:
            m.flux_term(op.name, x=body["x"], y=body["y"])

    def _b_local_source(op: Any) -> None:
        m.source_term(op.name, _body_for_state(op.body))

    def _b_local_linear_operator(op: Any) -> None:
        m.linear_source(op.name, _body_for_state(op.body))

    def _b_field_operator(op: Any) -> None:
        outputs = tuple(getattr(op.signature.output, "components", ()))
        if len(outputs) == 2 or len(outputs) > 3:
            raise ValueError(
                "compile_problem: field_operator %r outputs must have length 1 or 3; the runtime "
                "cannot register %d outputs yet" % (op.name, len(outputs)))
        if not outputs:
            raise ValueError(
                "compile_problem: field_operator %r must declare at least one output" % op.name)
        for output in outputs:
            # FieldSpace lowering above has already installed every output.  Canonical auxiliary
            # names use their dedicated slots and must never be redeclared as named extras.
            if output not in AUX_CANONICAL and output not in m._m.aux_extra_names:
                m.aux_field(output)
        m.elliptic_field(
            op.name, _body_for_state(op.body), operator="poisson", aux=outputs)

    def _b_local_rate(op: Any) -> None:
        low = op.lowering
        m.rate_operator(op.name, flux=low.get("flux", True),
                        sources=low.get("sources"), fluxes=low.get("fluxes"))

    def _b_projection(op: Any) -> None:
        m.projection(_body_for_state(op.body))

    builders = {"grid_operator": _b_grid_operator, "local_source": _b_local_source,
                "local_linear_operator": _b_local_linear_operator,
                "field_operator": _b_field_operator, "local_rate": _b_local_rate,
                "projection": _b_projection}
    builder_targets = {
        "grid_operator": "dsl:flux", "local_source": "dsl:source_term",
        "local_linear_operator": "dsl:linear_source",
        "field_operator": "dsl:elliptic_field", "local_rate": "dsl:rate_operator",
        "projection": "dsl:projection",
    }
    assert set(_CODEGEN_KINDS) <= set(OPERATOR_KINDS) and set(builders) <= set(OPERATOR_KINDS)
    for op in module.operator_registry():
        source = "operator:%s" % op.name
        coverage_rows.append(LoweringCoverageRow(
            "operator_metadata:%s" % op.name, "documentary"))
        body = op.body
        state_inputs = tuple(
            item for item in op.signature.inputs
            if getattr(item, "kind", None) == "state")
        if state_inputs and state not in state_inputs:
            coverage_rows.append(LoweringCoverageRow(
                source, "documentary"))
            continue
        if op.kind == "coupled_rate" or (
                op.kind == "field_operator" and len(state_inputs) > 1):
            coverage_rows.append(LoweringCoverageRow(
                source, "lowered", ("program:multi_block_operator",)))
            continue
        if op.kind in _CODEGEN_KINDS and (body is None or callable(body)):
            _reject(
                source,
                "expression_body_required",
                "compile_problem: operator %r (%s) has no IR body; a compilable Module operator "
                "needs an expression body (Module.operator(..., expr=...))" % (op.name, op.kind))
        if op.kind not in builders:
            _reject(
                source,
                "operator_kind_not_lowerable",
                "compile_problem: operator %r (%s) has no codegen lowering; every operator must "
                "map to executable behavior or be rejected" % (op.name, op.kind))
        builder = builders[op.kind]
        try:
            builder(op)
        except LoweringRejection:
            raise
        except Exception as exc:  # noqa: BLE001 -- every builder failure is a structured gate
            _reject(source, "operator_lowering_failed", str(exc))
        coverage_rows.append(LoweringCoverageRow(
            source, "lowered", (builder_targets[op.kind],)))
    coverage_report = LoweringCoverageReport(coverage_rows)
    object.__setattr__(m, "lowering_coverage_report", coverage_report)
    object.__setattr__(m, "_lowering_coverage_report", coverage_report)
    return m


def remap_lowering_error(exc: Any, facade: Any) -> None:
    """Re-raise a lowering ``ValueError`` citing the user's facade handles, not internal dsl symbols.

    When the user authored a physics :class:`pops.physics.Model` and the internal Module -> dsl
    lowering (or the model dependency check) fails, the raw message may name a dsl symbol the user
    never typed. This wraps it with the facade context -- the model name and its declared operator /
    state handle names -- so the diagnostic points at what the user WROTE (ADC-557 I3). @p facade is
    the physics Model (or ``None`` for a raw Module, where the message already speaks the user's IR).
    """
    if facade is None:
        raise exc
    name = getattr(facade, "name", None) or "model"
    ops = states = ()
    module = getattr(facade, "module", None)
    if module is not None:
        try:
            ops = tuple(op.name for op in module.operator_registry())
            states = tuple(module.state_spaces())
        except Exception:  # noqa: BLE001 -- a best-effort context, never mask the real error
            ops = states = ()
    message = (
        "pops.compile: lowering the physics model %r failed while validating it for compile.\n"
        "  %s\n"
        "Your model declares states %s and operators %s -- check that every quantity the flux / "
        "sources / field solve reference is declared on the model."
        % (name, exc, sorted(states) or "(none)", sorted(ops) or "(none)"))
    if isinstance(exc, LoweringRejection):
        raise LoweringRejection(
            message, coverage_report=exc.coverage_report,
            source=exc.source, gate=exc.gate) from exc
    raise ValueError(message) from exc


def lower_and_validate(model: Any, facade: Any = None, state_space: Any = None) -> Any:
    """The SINGLE validate + lower entry of the compile pipeline (ADC-557).

    Validates @p model ONCE and returns ``(emit_model, source_module)``:

      - every model provider supplies :class:`CompilerLowering`; its ``emit_model`` is the model the
        kernel emitters consume and its ``source_module`` is the exact operator-first canonical IR.
        ``pops.model.Module`` is itself such a provider and adapts to a dsl emitter; the physics
        facades return their existing emitter directly.
      - ``source_module`` is the operator-first :class:`pops.model.Module` authority carried by
        ``compiled.inspect()`` and used for ``module_hash`` drift detection.

    @p facade is the physics Model the user wrote (for the error remap); pass it when @p model was
    resolved FROM a facade so a lowering error cites the user's handles (:func:`remap_lowering_error`).
    A lowering / validation ``ValueError`` is remapped through @p facade and re-raised.
    """
    diagnostic_facade = facade
    try:
        from pops.codegen._compiler_lowering import require_compiler_lowering

        lowering = require_compiler_lowering(model)
        if diagnostic_facade is None:
            diagnostic_facade = lowering.facade
        states = lowering.source_module.state_spaces()
        if len(states) > 1:
            emit_model = _module_to_model(
                lowering.source_module, state_space=state_space)
            emit_model.check()
            return emit_model, lowering.source_module
        lowering.emit_model.check()
        return lowering.emit_model, lowering.source_module
    except ValueError as exc:
        remap_lowering_error(exc, diagnostic_facade)
