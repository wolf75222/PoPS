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


def _module_to_model(module):
    """Lower a :class:`pops.model.Module` to a :class:`pops.dsl.Model`
    (Spec 2, S2-11), reusing the dsl codegen engine -- a translation, NOT a
    second backend.  The Module's typed operators carry dsl ``Expr`` bodies;
    each is mapped to the dsl method of its kind.

    Imported lazily by compile_problem to avoid a top-level physics import.
    """
    # Import the model facade + aux constants lazily here (called only at
    # compile_problem time, not at import time).
    from pops.physics.facade import Model  # noqa: PLC0415
    from pops.physics.aux import AUX_CANONICAL  # noqa: PLC0415
    states = module.state_spaces()
    if len(states) != 1:
        raise ValueError("compile_problem: a Module must declare exactly one StateSpace to compile "
                         "(got %s)" % sorted(states))
    state = next(iter(states.values()))
    m = Model(module.name)
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
    m.primitive_vars(*cvars)
    m.conservative_from(list(cvars))
    for p in module.params().values():
        m.param(p.name, p.default)  # (name, value) shorthand -> a const param (no kind= string)
    declared = set()

    def _declare_aux(nm):
        if nm in declared:
            return
        declared.add(nm)
        if nm in AUX_CANONICAL:
            m.aux(nm)
        else:
            m.aux_field(nm)

    for fs in module.field_spaces().values():
        for comp in fs.components:
            _declare_aux(comp)
    for a in module.aux().values():
        _declare_aux(a.name)
    if module._eigenvalues is not None:
        m.eigenvalues(x=module._eigenvalues["x"], y=module._eigenvalues["y"])
    _CODEGEN_KINDS = ("grid_operator", "local_source", "local_linear_operator", "field_operator",
                      "projection")
    n_field_ops = 0
    for op in module.operator_registry():
        body = op.body
        if op.kind in _CODEGEN_KINDS and (body is None or callable(body)):
            raise ValueError(
                "compile_problem: operator %r (%s) has no IR body; a compilable Module operator "
                "needs an expression body (Module.operator(..., expr=...))" % (op.name, op.kind))
        if op.kind == "grid_operator":
            if op.name in ("flux", "flux_default"):
                m.flux(x=body["x"], y=body["y"])
            else:
                m.flux_term(op.name, x=body["x"], y=body["y"])
        elif op.kind == "local_source":
            m.source_term(op.name, body)
        elif op.kind == "local_linear_operator":
            m.linear_source(op.name, body)
        elif op.kind == "field_operator":
            n_field_ops += 1
            if n_field_ops > 1:
                raise ValueError(
                    "compile_problem: a Module currently supports one field_operator (the default "
                    "elliptic solve); multiple solved fields are deferred (operator %r)" % op.name)
            m.elliptic_rhs(body)
        elif op.kind == "local_rate":
            low = op.lowering
            m.rate_operator(op.name, flux=low.get("flux", True),
                            sources=low.get("sources"), fluxes=low.get("fluxes"))
        elif op.kind == "projection":
            m.projection(body)
    return m


def remap_lowering_error(exc, facade):
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
    raise ValueError(
        "pops.compile: lowering the physics model %r failed while validating it for compile.\n"
        "  %s\n"
        "Your model declares states %s and operators %s -- check that every quantity the flux / "
        "sources / field solve reference is declared on the model."
        % (name, exc, sorted(states) or "(none)", sorted(ops) or "(none)")) from exc


def lower_and_validate(model, facade=None):
    """The SINGLE validate + lower entry of the compile pipeline (ADC-557).

    Validates @p model ONCE and returns ``(emit_model, source_module)``:

      - ``emit_model`` is the model the kernel emitters consume: a raw :class:`pops.model.Module` is
        lowered to a dsl model via :func:`_module_to_model` (whose embedded checks ARE the validation);
        a dsl / physics ``Model`` is consumed as-is (byte-identical emit) after its ``check()``
        dependency validation runs -- the ONE validation, replacing the removed divergent
        ``model.check()`` compile step.
      - ``source_module`` is the operator-first :class:`pops.model.Module` -- the canonical compile-IR
        authority: the raw Module itself, or the dsl / physics model's ``.module`` view. It is what
        ``compiled.inspect()`` carries as the lowered-module trace and what ``module_hash`` drifts
        against. ``None`` only for a bare dsl model with no backing Module.

    @p facade is the physics Model the user wrote (for the error remap); pass it when @p model was
    resolved FROM a facade so a lowering error cites the user's handles (:func:`remap_lowering_error`).
    A lowering / validation ``ValueError`` is remapped through @p facade and re-raised.
    """
    try:
        from pops import model as _model_pkg
    except ImportError:
        _model_pkg = None
    try:
        if _model_pkg is not None and isinstance(model, _model_pkg.Module):
            source_module = model
            emit_model = _module_to_model(model)
            return emit_model, source_module
        # A dsl / physics Model: the ONE dependency validation is its own check() (fail-loud); the
        # operator-first Module view is the canonical trace authority.
        if model is not None and hasattr(model, "check"):
            model.check()
        source_module = getattr(model, "module", None)
        return model, source_module
    except ValueError as exc:
        remap_lowering_error(exc, facade)
