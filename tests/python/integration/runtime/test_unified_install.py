"""Spec 3 section 22 + 24 (ADC-466): unified ``sim._install_compiled(...)`` + install-time validation.

``sim._install_compiled(compiled, instances=, params=, aux=, solvers=)`` is the single Spec-3 entry that
installs the compiled handle, binds each named instance's block by name, sets its initial state and
spatial brick, sets the field solvers / aux fields / runtime params, and finally installs the
compiled time Program -- LOWERING to the existing lower-layer calls (add_equation / set_poisson /
set_magnetic_field / set_aux_field / set_block_params / install_program), no parallel runtime.

The full compiled-.so install RUN needs a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) and is
validated on ROMEO / CI-Kokkos (mirrors test_install_requirement_validation.py). The API SHAPE, the
lowering, and the section-24 capability/aux/solver validation messages are host-testable WITHOUT a
full run -- exercised here. cf. docs/sphinx/reference/board-like-dsl.md.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import HLL
from pops.numerics.riemann import HLLC
from pops.numerics.variables import Primitive
from pops.numerics.riemann import Roe
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import WENO5
import sys

try:
    import numpy as np

    import pops
    from pops.codegen._plans import BindInputs, InstallPlan, ResolvedBlock, ResolvedSimulationPlan
    from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
    from pops.codegen.loader import CompiledModel
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops.params import ConstParam, RuntimeParam
    from pops.problem._snapshot import AuthoringSnapshot
    from pops import time as adctime
    from pops.runtime.system import System  # ADC-545 advanced runtime seam
except Exception as exc:  # noqa: BLE001
    print("skip test_unified_install (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

from tests.python.support.layout_plan import resolved_layout_contract

N = 16


def _fake_compiled(*, hllc=False, roe=False, prim_names=("rho", "u", "v"), wave_speeds=False,
                   params=None):
    """A real pops.dsl.CompiledModel object (the engine class) carrying only metadata -- NOT a built
    .so. Used to exercise the host-testable section-24 capability check and the params routing
    WITHOUT compiling (which needs Kokkos). It is never install_program'd."""
    from pops.codegen._compiled_model_identity import compiled_model_identity
    model = CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["density", "momentum_x", "momentum_y"],
        prim_names=list(prim_names), n_vars=3, gamma=None, n_aux=3, params=params or {},
        caps={}, abi_key="", model_hash="fixture-model", cxx="c++", std="23",
        hllc=hllc, roe=roe, wave_speeds=wave_speeds)
    model.definition_identity = compiled_model_identity(model_hash="fixture-model")
    return model


def _compiled_artifact(compiled, block_model, *, spatial=None, bind_schema=None):
    """Build the exact immutable output of compile for the low-level install-seam tests."""
    from pops.model.bind_schema import BindSchema

    if compiled is None:
        raise ValueError("the final system artifact requires a whole-system compiled Program")
    discard_authoring = getattr(compiled, "_discard_authoring", None)
    if callable(discard_authoring) and getattr(compiled, "model", None) is not None:
        discard_authoring()
    bind_schema = bind_schema or BindSchema()
    snapshot = AuthoringSnapshot({
        "kind": "unified-install-integration",
        "block": "plasma",
        "model_hash": block_model.model_hash,
        "has_program": True,
    })
    layout_plan, layout_coverage = resolved_layout_contract(
        None, target="system", block_names=("plasma",))
    resolved = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="system",
        backend=block_model.backend,
        layout=None,
        layout_plan=layout_plan,
        time={"kind": "compiled-program"},
        blocks=(ResolvedBlock("plasma", {"model_hash": block_model.model_hash}, spatial,
                              block_model.backend),),
        bind_schema=bind_schema,
        compile_values=bind_schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={},
        capabilities={},
        lowering_coverage=layout_coverage,
    )
    return CompiledSimulationArtifact(
        plan=resolved,
        program=compiled,
        blocks=(CompiledBlockArtifact("plasma", block_model, spatial),),
    )


def _bind_plan(artifact, initial, *, params=None, aux=None):
    """Materialize the exact bind output without consulting an authoring model."""
    inputs = BindInputs(initial_state={"plasma": initial}, params=params or {}, aux=aux or {})
    resolved_params = artifact.bind_schema.resolve_bind(
        inputs.params, compile_values=artifact.plan.compile_values)
    block = artifact.blocks[0]
    return InstallPlan(
        artifact=artifact,
        bind_inputs=inputs,
        instances={"plasma": {
            "model": block.model,
            "spatial": block.spatial,
            "initial": initial,
        }},
        params=resolved_params,
        aux=inputs.aux,
    )


def test_lower_spatial_accepts_runtime_and_catalog():
    """install lowers BOTH an pops.FiniteVolume (runtime) and an pops.numerics.spatial.FiniteVolume
    (catalog descriptor) to the same add_equation spatial args."""
    sim = System(n=N, L=1.0, periodic=True)
    # Runtime descriptor passes through unchanged.
    rt = pops.FiniteVolume(limiter=WENO5(), riemann=HLL(), variables=Primitive())
    low = sim._lower_spatial(rt)
    assert low is rt, "runtime Spatial must pass through unchanged"
    # catalog descriptor: riemann/reconstruction/positivity_floor -> limiter/flux/recon.
    # NB pops.numerics.spatial.FiniteVolume is the brick-CATALOG descriptor: it stores its scheme
    # choice as STRING options (lowered to typed tokens by _lower_spatial), distinct from the
    # runtime pops.FiniteVolume which now requires typed pops.numerics descriptors (Spec 5 sec.7).
    libdesc = pops.numerics.spatial.FiniteVolume(riemann="hllc", reconstruction="weno5",
                                                 positivity_floor=1e-12)
    low = sim._lower_spatial(libdesc)
    assert low.flux == "hllc", "riemann -> Spatial.flux (got %r)" % low.flux
    assert low.limiter == "weno5", "reconstruction -> Spatial.limiter (got %r)" % low.limiter
    assert low.positivity_floor == 1e-12, "positivity_floor lowered (got %r)" % low.positivity_floor
    # None -> default Spatial.
    assert isinstance(sim._lower_spatial(None), pops.Spatial)
    print("OK  _lower_spatial accepts runtime + lib descriptors")


def test_solver_token_lowering():
    """A field-solver selection lowers to its set_poisson token: string as-is, or the lib
    descriptor's scheme (pops.fields.catalog.GeometricMG -> 'geometric_mg')."""
    sim = System(n=N, L=1.0, periodic=True)
    assert sim._solver_token("geometric_mg") == "geometric_mg"
    assert sim._solver_token(pops.fields.catalog.GeometricMG()) == "geometric_mg"
    print("OK  _solver_token lowers string + lib descriptor")


def test_install_solver_sets_poisson():
    """install lowers solvers={'phi': GeometricMG(...)} to set_poisson, reflected by poisson_solver()
    (the section-24 accessor) when the binding is present."""
    sim = System(n=N, L=1.0, periodic=True)
    sim._install_solver("phi", pops.fields.catalog.GeometricMG())
    if hasattr(sim._s, "poisson_solver"):
        assert sim.poisson_solver() == "geometric_mg", \
            "set_poisson lowered (got %r)" % sim.poisson_solver()
        print("OK  _install_solver lowers to set_poisson (poisson_solver() == geometric_mg)")
    else:
        print("OK  _install_solver lowers to set_poisson (poisson_solver accessor absent; rebuild _pops)")
    # C1-System: a DECLARED named elliptic field routes through the shared elliptic solver (set_poisson);
    # an UNDECLARED field name is a typo, rejected LOUD against the declared set (not silently dropped).
    sim2 = System(n=N, L=1.0, periodic=True)
    sim2._install_solver("temperature", pops.fields.catalog.GeometricMG(),
                         declared_fields=frozenset({"temperature"}))
    if hasattr(sim2._s, "poisson_solver"):
        assert sim2.poisson_solver() == "geometric_mg", \
            "a declared named field routes to set_poisson (got %r)" % sim2.poisson_solver()
    print("OK  _install_solver routes a DECLARED named elliptic field (C1-System)")
    try:
        sim2._install_solver("temprature", pops.fields.catalog.GeometricMG(),
                             declared_fields=frozenset({"temperature"}))
        raise AssertionError("MISMATCH: an undeclared field name should raise ValueError")
    except ValueError as exc:
        assert "temprature" in str(exc) and "temperature" in str(exc)
        print("OK  _install_solver rejects an UNDECLARED field name, naming the declared set")


def test_riemann_capability_verbatim():
    """Section 24: the selected Riemann flux must be backed by the model capability. The install
    check now delegates to the shared gate (ADC-642), so it raises ValueError naming the missing
    capability. A compiled model WITHOUT the HLLC capability and WITHOUT a pressure rejects
    riemann='hllc'."""
    sim = System(n=N, L=1.0, periodic=True)
    model = _fake_compiled(hllc=False, prim_names=("rho", "u", "v"))
    try:
        sim._validate_riemann_capability(model, pops.FiniteVolume(riemann=HLLC()))
        raise AssertionError("MISMATCH: hllc without capability should raise")
    except ValueError as exc:
        assert "hllc_star_state" in str(exc), \
            "names the missing capability (got %r)" % str(exc)
        print("OK  riemann HLLC requires capability 'hllc_star_state'")
    # Roe without capability / pressure rejects too.
    try:
        sim._validate_riemann_capability(model, pops.FiniteVolume(riemann=Roe()))
        raise AssertionError("MISMATCH: roe without capability should raise")
    except ValueError as exc:
        assert "roe_dissipation" in str(exc).lower() or "Roe requires capability" in str(exc), \
            "roe capability message (got %r)" % str(exc)
        print("OK  riemann Roe requires its capability")
    # With the capability emitted, the same flux passes.
    ok_model = _fake_compiled(hllc=True, prim_names=("rho", "u", "v", "p"))
    sim._validate_riemann_capability(ok_model, pops.FiniteVolume(riemann=HLLC()))
    print("OK  riemann capability accepted once the model emits it")


def test_install_aux_derived_rejected():
    """install rejects aux={'T_e': ...} (T_e is DERIVED, not a static aux field) and a named aux not
    declared by any installed instance -- both host-testable, no .so."""
    sim = System(n=N, L=1.0, periodic=True)
    try:
        sim._install_aux("T_e", np.ones(N * N))
        raise AssertionError("MISMATCH: T_e should be rejected (derived)")
    except ValueError as exc:
        assert "T_e" in str(exc) and "set_electron_temperature_from" in str(exc)
        print("OK  install rejects aux 'T_e' (derived)")
    try:
        sim._install_aux("grad_phi_custom", np.ones(N * N))
        raise AssertionError("MISMATCH: an undeclared named aux should be rejected")
    except ValueError as exc:
        assert "not declared by any installed instance" in str(exc)
        print("OK  install rejects an undeclared named aux field")


def test_install_params_routing():
    """BindSchema rejects ownerless parameter names before install routing."""
    from pops.model import Module
    from pops.model.bind_schema import BindSchema
    from pops.problem import Case

    module = Module("qualified-routing")
    module.param(RuntimeParam("nu", default=1.0))
    problem = Case(name="qualified-routing")
    problem.block("plasma", module)
    schema = BindSchema.from_problem(problem)
    try:
        schema.resolve_bind({"nu": 1.0}, compile_values=schema.resolve_compile())
        raise AssertionError("MISMATCH: an ownerless param name should raise")
    except TypeError as exc:
        assert "ParamHandle" in str(exc)
        print("OK  BindSchema rejects an ownerless parameter name")


def test_install_params_routes_declared_runtime_param():
    """A complete qualified mapping projects to the compiled model's native slot order."""
    from pops.model import Module
    from pops.model.bind_schema import BindSchema
    from pops.problem import Case
    from pops.runtime._install_param_routing import route_block_params

    declarations = {
        "nu": RuntimeParam("nu", default=0.0),
        "cs2": RuntimeParam("cs2", default=1.0),
        "g": ConstParam("g", 9.8),
    }
    module = Module("qualified-routing")
    handles = {name: module.param(declaration) for name, declaration in declarations.items()}
    problem = Case(name="qualified-routing")
    block = problem.block("plasma", module)
    schema = BindSchema.from_problem(problem)

    # A RESOLVED model declaring two runtime params + one const (const excluded; names SORTED).
    resolved = _fake_compiled(params=declarations)
    assert resolved.runtime_param_names == ["cs2", "nu"], \
        "runtime params SORTED, const excluded (got %r)" % resolved.runtime_param_names
    values = schema.resolve_bind(
        {block[handles["nu"]]: 2.5}, compile_values=schema.resolve_compile())
    per_block = route_block_params({"plasma": resolved}, schema, values)
    assert per_block == {"plasma": [1.0, 2.5]}, \
        "set_block_params vector sorted by name: cs2 keeps default 1.0, nu set to 2.5 (got %r)" \
        % per_block
    print("OK  qualified values project to native slot order with defaults materialized")


def _lorentz_model(name="adc466_model"):
    """An isothermal fluid whose Lorentz linear source reads the aux field B_z (a hard requirement),
    same shape as test_install_requirement_validation -- used for the Kokkos-gated end-to-end."""
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs = sqrt(0.5)
    m.flux(x=[mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + 0.5 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)
    m.conservative_from([rho, mx, my])
    bz = m.aux("B_z")
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def _lie_program(name="adc466_prog"):
    from pops.model import Module
    from pops.problem import Case

    module = Module(name + "-state")
    state = module.state_space("U", ("rho", "mx", "my"))
    problem = Case(name=name + "-case")
    block = problem.block("plasma", module)
    P = adctime.Program(name)
    endpoint = P.state(block, module.state_handle(state))
    u = endpoint.n
    fields = P.solve_fields(u)
    r = P._rhs_legacy(state=u, fields=fields)
    P.commit(endpoint.next, P.value("u1", u + P.dt * r, at=endpoint.next.point))
    return P


def test_install_end_to_end_kokkos():
    """End-to-end unified install (needs a compiler + Kokkos -> ROMEO / CI-Kokkos). A single
    sim._install_compiled(compiled, instances=, aux=, solvers=) wires + installs; the NEGATIVE case (no B_z)
    raises the section-24 aux requirement at install."""
    if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip test_install_end_to_end_kokkos (_pops lacks install_program; rebuild _pops)")
        return
    m = _lorentz_model()
    try:
        from pops.codegen.compile_drivers import compile_problem
        compiled = compile_problem(model=m, time=_lie_program())
        block_model = m.compile(backend="production", target="system")
    except RuntimeError as exc:
        print("skip test_install_end_to_end_kokkos (no Kokkos to build the .so: %s)"
              % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho, 0.4 * rho, -0.2 * rho])
    spatial = pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov())
    artifact = _compiled_artifact(compiled, block_model, spatial=spatial)

    # Negative: install WITHOUT aux B_z -> section-24 aux requirement raised at install_program.
    sim_missing = System(n=N, L=1.0, periodic=True)
    try:
        missing_plan = _bind_plan(artifact, u0)
        sim_missing._install_compiled(
            missing_plan.artifact,
            instances=missing_plan.instances,
            params=missing_plan.params,
            solvers={"phi": pops.fields.catalog.GeometricMG()})
        raise AssertionError("MISMATCH: unified install accepted a simulation missing B_z")
    except RuntimeError as exc:
        assert "lorentz" in str(exc) and "B_z" in str(exc) and "did not provide" in str(exc), \
            "section-24 aux message (got %r)" % str(exc)
        print("OK  unified install rejects a missing required aux: %s" % str(exc))

    # Positive: the SAME install with aux={'B_z': ...} wires + installs cleanly.
    sim_ok = System(n=N, L=1.0, periodic=True)
    ok_plan = _bind_plan(artifact, u0, aux={"B_z": 3.0 * np.ones(N * N)})
    sim_ok._install_compiled(
        ok_plan.artifact,
        instances=ok_plan.instances,
        params=ok_plan.params,
        aux=ok_plan.aux,
        solvers={"phi": pops.fields.catalog.GeometricMG()})
    assert "plasma" in sim_ok.block_names(), "instance bound by name"
    print("OK  unified install wires instance + aux + solver and installs the program")


def _iso_runtime_model(name="adc466_rt_model", *, with_handle=False):
    """An isothermal fluid with a declared runtime ``cs2`` parameter and no required aux.

    Compilation freezes its ABI slot into ``CompiledModel.runtime_param_names`` so install can route
    values without retaining this authoring object.
    """
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2_param = m.param(RuntimeParam("cs2", default=0.5))
    cs2 = m.value(cs2_param)
    cs = sqrt(cs2)
    m.flux(x=[mx, mx * mx / rho + cs2 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + cs2 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)
    m.conservative_from([rho, mx, my])
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return (m, cs2_param) if with_handle else m


def test_install_routes_runtime_param_kokkos():
    """End-to-end (Kokkos-gated): the install path carries a separately compiled block loader in
    ``InstallPlan`` and routes ``params={...}`` to ``set_block_params`` on that block. Bind consumes
    detached ``CompiledModel`` metadata directly; it neither reads ``compiled.model`` nor compiles
    the authoring model. Self-skips without a compiler / Kokkos."""
    if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip test_install_routes_runtime_param_kokkos (_pops lacks install_program; rebuild _pops)")
        return
    m, cs2_param = _iso_runtime_model(with_handle=True)
    try:
        from pops.codegen.compile_drivers import compile_problem
        compiled = compile_problem(model=m, time=_lie_program())
        block_model = m.compile(backend="aot", target="system")
    except RuntimeError as exc:
        print("skip test_install_routes_runtime_param_kokkos (no Kokkos to build the .so: %s)"
              % str(exc)[:120])
        return

    from pops.model.bind_schema import BindSchema
    from pops.problem import Case

    problem = Case(name="adc466-runtime")
    block = problem.block("plasma", m)
    bind_schema = BindSchema.from_problem(problem)
    bind_values = {block[cs2_param]: 1.0}

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)])  # u=0 -> momentum residual ~ cs2*rho
    spatial = pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov())
    artifact = _compiled_artifact(
        compiled, block_model, spatial=spatial, bind_schema=bind_schema)
    plan = _bind_plan(artifact, u0, params=bind_values)

    sim = System(n=N, L=1.0, periodic=True)
    sim._install_compiled(
        plan.artifact,
        instances=plan.instances,
        params=plan.params,
        solvers={"phi": pops.fields.catalog.GeometricMG()})
    assert "plasma" in sim.block_names(), "InstallPlan instance bound by name"
    print("OK  headline install(params=) routes a runtime param from detached metadata")

    # The routed param is LIVE on the block: with u=0 the momentum residual is -div(cs2*rho), so cs2
    # 1 -> 4 scales it x4 -- proof set_block_params reached the real block (P7-b).
    sim._s.set_block_params("plasma", [1.0])
    R1 = np.array(sim._s.eval_rhs("plasma")).reshape(3, N, N)
    sim._s.set_block_params("plasma", [4.0])
    R4 = np.array(sim._s.eval_rhs("plasma")).reshape(3, N, N)
    assert np.max(np.abs(R1[1])) > 1e-3, "momentum residual non-trivial at cs2=1"
    assert np.allclose(R4[1], 4.0 * R1[1], rtol=1e-9, atol=1e-12), \
        "momentum residual -div(cs2*rho) must scale x4 when cs2 1 -> 4 (param routed to the block)"
    print("OK  the routed runtime param is live on the block (eval_rhs scales with cs2)")

    # Time is compile-owned in the final artifact. Bind cannot inject a competing per-instance
    # integrator into the final InstallPlan.
    try:
        InstallPlan(
            artifact=plan.artifact,
            bind_inputs=plan.bind_inputs,
            instances={"plasma": {**dict(plan.instances["plasma"]),
                                  "time": pops.Explicit(method="euler")}},
            params=plan.params,
            aux=plan.aux,
        )
        raise AssertionError("MISMATCH: bind injected a per-instance time scheme")
    except TypeError as exc:
        assert "model/spatial/initial" in str(exc), exc
        print("OK  bind cannot override the compile-owned time Program")


def test_install_cadence_routing():
    """install(cadence=CompiledTime(...)) absorbs the compiled-program macro-step cadence: it routes
    to set_program_cadence(substeps, stride). A bad type is rejected up front; a NUMERIC cfl is pinned
    on the System (C7) so a bare sim.run(t_end) defaults to it (not silently ignored).
    Host-testable -- set_program_cadence is a pure System-level setter (no installed .so needed)."""
    sim = System(n=N, L=1.0, periodic=True)
    if not hasattr(sim._s, "set_program_cadence"):
        print("skip test_install_cadence_routing (_pops lacks set_program_cadence; rebuild _pops)")
        return
    # A CompiledTime is routed to set_program_cadence(substeps, stride) (no error).
    sim._install_cadence(adctime.CompiledTime(substeps=2, stride=3))
    # A non-CompiledTime is rejected BEFORE any engine call.
    try:
        sim._install_cadence("not a cadence")
        raise AssertionError("install(cadence=) accepted a non-CompiledTime")
    except TypeError as exc:
        assert "CompiledTime" in str(exc), exc
    # C7: a NUMERIC cfl is accepted and PINNED on the System, so run() with no explicit cfl uses it.
    assert sim._program_cadence_cfl is None, "no cadence cfl pinned yet"
    sim._install_cadence(adctime.CompiledTime(substeps=1, stride=1, cfl=0.5))
    assert sim._program_cadence_cfl == 0.5, \
        "a numeric cadence cfl is pinned on the System (got %r)" % sim._program_cadence_cfl
    print("OK  install(cadence=) routes CompiledTime -> set_program_cadence; pins a numeric cfl (C7)")


def test_install_native_cadence_rejected():
    """A native sim (compiled=None) has no compiled Program, so install(cadence=) is rejected -- the
    cadence is a compiled-program concept. Host-testable (the guard fires before any engine run)."""
    sim = System(n=N, L=1.0, periodic=True)
    try:
        sim._install_compiled(None, cadence=adctime.CompiledTime(substeps=2, stride=1))
        raise AssertionError("install(compiled=None, cadence=) was accepted")
    except ValueError as exc:
        assert "cadence" in str(exc) and "native" in str(exc), exc
    print("OK  native install rejects cadence= (no compiled Program)")


def test_system_artifact_rejects_missing_whole_program():
    """The final System contract has no legacy per-block/no-Program compiler path."""
    block_model = _fake_compiled()
    try:
        _compiled_artifact(None, block_model, spatial=pops.FiniteVolume(
            limiter=FirstOrder(), riemann=Rusanov()))
        raise AssertionError("a System artifact without a whole-system Program was accepted")
    except ValueError as exc:
        assert "whole-system compiled Program" in str(exc)
    print("OK  exact System artifact rejects the superseded no-Program path")


def main():
    test_lower_spatial_accepts_runtime_and_catalog()
    test_solver_token_lowering()
    test_install_solver_sets_poisson()
    test_riemann_capability_verbatim()
    test_install_aux_derived_rejected()
    test_install_params_routing()
    test_install_params_routes_declared_runtime_param()
    test_install_cadence_routing()
    test_install_native_cadence_rejected()
    test_install_end_to_end_kokkos()
    test_system_artifact_rejects_missing_whole_program()
    test_install_routes_runtime_param_kokkos()
    return 0


if __name__ == "__main__":
    sys.exit(main())
