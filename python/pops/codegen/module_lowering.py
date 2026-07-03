"""pops.codegen.module_lowering : lower a pops.model.Module to a dsl.Model for kernel emission.

The operator-first :class:`pops.model.Module` is the canonical compile IR (ADC-557); the kernel
emitters still consume a :class:`pops.dsl.Model`, so ``compile_problem`` lowers the Module to a
dsl.Model INTERNALLY. That translation -- a re-expression of the SAME physics, not a second backend
-- lives here, out of ``compile_drivers`` so both stay under the 500-line budget.

``_module_to_model`` embeds the single structural validation of a compilable Module (exactly one
StateSpace, an expression body for every codegen operator, at most one field_operator). That IS the
compile pipeline's one validation (ADC-557): there is no second ``model.check()`` path. Imported
lazily by ``compile_problem`` to avoid a top-level physics import.
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
