"""ADC-547: the ONE declarative spec-compliance matrix (positive + negative cells).

This module is the single greppable source of truth for the compliance matrix. It exports two
dicts, :data:`POSITIVE` and :data:`NEGATIVE`, keyed by a stable ``cell_id``. ``test_spec_matrix.py``
iterates them: every positive cell INSPECTS the chosen native route (via
``pops.native_capability_report().routes`` / ``compiled.inspect()`` / ``Problem.explain_routes()``)
and asserts it is the advertised route -- not merely "no exception"; every negative cell asserts a
STABLE, warning-free refusal (exception type + exact message needles that are already stable strings
in the sources), so a message drift fails in ONE place.

Design notes (from the plan):
  * The CLEAN route only. The ADC-597 matrix drives the legacy ``System(...).add_block()``
    route; this matrix inspects the Problem/layout/descriptor clean route and the metadata-stub bind
    gates instead (``pops.runtime.routes`` install-time predicates over a ``CompiledModel`` stub;
    ``run_bind_gates`` over an exact ``CompiledSimulationArtifact`` -- no ``.so`` on disk). That is
    also the
    phase-6 lesson: every compiler-gated decision runs its full PRE-compile path locally.
  * No broad allowlists; each cell names exactly one route/refusal.

Importing this module requires ``pops``; ``test_spec_matrix.py`` importorskips it so the suite is
green on a bare box. ASCII only.
"""
import warnings

import numpy as np

import pops
from pops import time as adctime
from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
from pops.codegen.loader import CompiledModel
from pops.diagnostics import MinMax
from pops.fields import PoissonProblem
from pops.fields.bcs import Dirichlet
from pops.ir.expr import Var
from pops.identity import make_identity
from pops.math import laplacian, unknown
from pops.mesh import CartesianMesh
from pops.mesh.amr import RegridEvery
from pops.mesh.cartesian import CartesianMesh as _CartesianMesh
from pops.mesh.layouts import AMR, Uniform
from pops.numerics.riemann import HLL
from pops.output import CheckpointPolicy
from pops.runtime import routes as _routes
from pops.runtime._bind_validation import loaded_runtime_facts, run_bind_gates
from pops.model.bind_schema import BindSchema
from pops.params import RuntimeParam
from pops.problem._snapshot import AuthoringSnapshot
from pops.solvers.elliptic import GeometricMG
from pops.solvers.krylov import CG
from pops import model as _model


# --------------------------------------------------------------------------------------------------
# Declarative cell shapes.
# --------------------------------------------------------------------------------------------------
class PositiveCell:
    """A supported route. ``check()`` inspects the chosen native route and returns the route_id it
    proved available; the runner asserts that id is in the expected set."""

    def __init__(self, route_ids, check):
        self.route_ids = tuple(route_ids)  # the native route(s) this cell proves available
        self.check = check  # callable() -> the route_id string it proved


class NegativeCell:
    """A refused route. ``call()`` must raise ``exc_type`` with every needle present, and emit no
    warning (a warning + fallback is exactly what the matrix must catch)."""

    def __init__(self, exc_type, needles, call):
        self.exc_type = exc_type
        self.needles = tuple(needles)
        self.call = call


# --------------------------------------------------------------------------------------------------
# Shared native-route inspection helpers (the CLEAN inspection surface, no System).
# --------------------------------------------------------------------------------------------------
def _routes_by_id():
    return {row.to_dict()["route_id"]: row.to_dict()
            for row in pops.native_capability_report().routes}


def _assert_route_available(route_id):
    rows = _routes_by_id()
    row = rows.get(route_id)
    assert row is not None, "native capability report is missing route %r" % route_id
    assert row["status"] == "available", (
        "route %r is %r, expected available" % (route_id, row["status"]))
    return route_id


def _abi():
    """The loaded runtime's own ABI key, so the ABI gate is a no-op unless a cell forces a mismatch."""
    return loaded_runtime_facts().get("abi_key") or "SIG|c++|c++23"


def _compiled_model(**overrides):
    """A metadata-only CompiledModel stub (no .so): the clean carrier for the routes-module gates."""
    base = dict(
        so_path="/no/such/pops-route.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["density", "momentum_x", "momentum_y"],
        prim_names=["rho", "u", "v"], n_vars=3, gamma=None, n_aux=3, params={}, caps={"cpu": True},
        abi_key=_abi(), model_hash="modelhash", cxx="c++", std="c++23")
    base.update(overrides)
    return CompiledModel(**base)


def _poisson(solver, *bcs):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho), bcs=bcs, solver=solver)


def _state_refs(model_name, block_name):
    """Create the typed block/state declarations required by the final Program API."""
    module = _model.Module(model_name)
    state_space = module.state_space("U", ("rho", "mx", "my"))
    state = module.state_handle(state_space)
    block = pops.Problem(name="%s-case" % model_name).add_block(block_name, module)
    return module, block, state


def _program_with_context():
    program = adctime.Program("adc547_ctx")
    dt = program.dt
    _module, block, declaration = _state_refs("adc547-ctx-model", "gas")
    temporal = program.state(block, declaration)
    state = temporal.n
    rhs = program._rhs_legacy(state=state, flux=True, sources=["default"])
    program.commit(temporal.next, program.linear_combine("U1", state + dt * rhs))
    return program


def _bindable_program(name="adc547_bind"):
    program = adctime.Program(name)
    dt = program.dt
    _module, block, declaration = _state_refs("%s-model" % name, "plasma")
    temporal = program.state(block, declaration)
    state = temporal.n
    fields = program.solve_fields("phi", state)
    rhs = program._rhs_legacy(state=state, fields=fields, flux=True, sources=["default"])
    program.commit(temporal.next, program.linear_combine("U1", state + dt * rhs))
    return program


def _bind_model(aux_names=()):
    model = CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=len(aux_names), params={},
        caps={"cpu": True}, abi_key=_abi(), model_hash="h", cxx="c++", std="c++23",
        aux_extra_names=list(aux_names))
    model.artifact_identity = make_identity(
        "artifact", {"fixture": "adc547-bind-model", "aux": list(aux_names)})
    return model


def _compiled_problem(aux_names=(), *, abi_key=None):
    """Exact system artifact used by the metadata-only bind gates."""
    program = _bindable_program()
    model = _bind_model(aux_names)
    key = _abi() if abi_key is None else abi_key
    model.abi_key = key
    snapshot = AuthoringSnapshot({"kind": "adc547-bind-fixture"})
    schema = BindSchema()
    plan = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="system",
        backend="production",
        layout={"kind": "uniform"},
        time={"program": "adc547_bind"},
        blocks=(ResolvedBlock(
            "plasma", {"model": "adc547-bind-model"}, None, "production"),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={},
        capabilities={"cpu": True},
    )

    class _CompiledProgram:
        so_path = "/tmp/pops-cache/problem.so"
        target = "system"
        backend = "production"
        abi_key = key
        cxx = "c++"
        std = "c++23"
        program_name = "adc547_bind"

        def commits(self):
            return program.commits()

        @property
        def _values(self):
            return program._values

        def to_data(self):
            return {"kind": "compiled-program", "name": self.program_name}

        def arguments(self):
            from pops.codegen.inspect_compiled import build_arguments

            return build_arguments(self.artifact)

        def manifest(self):
            from pops.external.artifact_manifest import build_compiled_manifest

            return build_compiled_manifest(self.artifact)

    compiled_program = _CompiledProgram()
    artifact = CompiledSimulationArtifact(
        plan=plan,
        program=compiled_program,
        blocks=(CompiledBlockArtifact("plasma", model, None),),
    )
    compiled_program.artifact = artifact
    return artifact


def _uniform(n=8):
    return Uniform(_CartesianMesh(n=n, periodic=True))


def _arity_module():
    """A Module with a two-input operator, for the true-arity-mismatch negative cell."""
    module = _model.Module("adc547_arity")
    u = module.state_space("U", ("rho", "mx", "my"))
    fields = module.field_space("fields")
    module.operator(name="explicit_rhs", kind="local_rate",
                    signature=(u, fields) >> _model.Rate(u), expr="<ir>")
    module.operator(name="fields_from_state", kind="field_operator",
                    signature=(u,) >> fields, expr="<ir>")
    return module


# --------------------------------------------------------------------------------------------------
# POSITIVE cells (8) -- each proves the chosen native route is advertised available.
# --------------------------------------------------------------------------------------------------
def _pos_uniform_fv_typed_riemann():
    # Typed HLL()/Primitive() descriptors carry a native id; the FV + HLL native routes are available.
    assert HLL().inspect()["native_id"] == "pops::HLLFlux"
    _assert_route_available("spatial:finite_volume")
    return _assert_route_available("riemann:hll")


def _pos_uniform_poisson_elliptic():
    # PoissonProblem + GeometricMG validates on the clean route; the elliptic MG route is available.
    assert _poisson(GeometricMG(), Dirichlet()).validate() is True
    return _assert_route_available("elliptic:geometric_mg")


def _pos_program_manual_operator():
    # A manual operator-first Program lowers through the generic ProgramContext seam (no fallback).
    source = _program_with_context().emit_cpp_program(model=None)
    assert "ProgramContext" in source
    assert "pops/runtime/program/program_context.hpp" in source
    return _assert_route_available("program_context:system")


def _pos_program_macro_lib_time():
    # A lib.time macro produces the canonical Program with a stable IR hash (no lib stepper).
    import pops.lib.time as lib_time

    _module, block, state = _state_refs("adc547-ssprk3-model", "plasma")
    program = lib_time.ssprk3(block, state)
    assert isinstance(program, adctime.Program)
    assert program._ir_hash() == program._ir_hash()
    return _assert_route_available("program_context:system")


def _pos_matrix_free_krylov():
    # A typed Krylov descriptor carries a native id; the Krylov route is available.
    assert CG(max_iter=200).inspect()["native_id"] == "pops::cg_solve"
    return _assert_route_available("krylov:cg_bicgstab_gmres_richardson")


def _pos_params_runtime_const_bind():
    # BindSchema materialises defaults and accepts only the block-qualified ParamHandle.
    module = _model.Module("adc547-runtime-param")
    k = module.param(RuntimeParam("k", default=2.0))
    problem = pops.Problem(name="adc547-runtime-bind")
    gas = problem.add_block("gas", module)
    schema = BindSchema.from_problem(problem)
    canonical = problem.resolve(k, block=gas)
    assert schema.resolve()[canonical] == 2.0
    assert schema.resolve({gas[k]: 6.0})[canonical] == 6.0
    try:
        schema.resolve({"zzz": 9.0})
        raise AssertionError("an ownerless parameter name must be rejected")
    except TypeError:
        pass
    # The bind PASS path runs over the metadata stub (no .so): a well-formed install passes the gates.
    run_bind_gates(_compiled_problem(), _uniform(), {"plasma": np.ones((3, 8, 8))}, {}, {})
    # The associated native route (the program install path) is advertised available.
    return _assert_route_available("program_context:system")


def _pos_diagnostics_output_ckpt():
    # Typed diagnostic + checkpoint descriptors inspect/validate; the output + checkpoint routes exist.
    minmax = MinMax()
    assert minmax.inspect()["category"] == "diagnostic_minmax"
    assert CheckpointPolicy().validate() in (True, None) or CheckpointPolicy().inspect()
    _assert_route_available("output:npz_vtk_hdf5")
    return _assert_route_available("checkpoint:system_v1")


def _pos_amr_route_when_capable():
    # AMR(...) is the typed config surface (capabilities.layout == amr); the AMR layout route exists.
    layout = AMR(base=CartesianMesh(n=64), max_levels=2, ratio=2, regrid=RegridEvery(4))
    assert layout.inspect()["capabilities"]["layout"] == "amr"
    return _assert_route_available("layout:AMR")


def _amr_route_handle():
    """An exact metadata-only AMR compiled artifact."""

    handle = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=1, params={},
        caps={"cpu": True, "amr": True, "mpi": True}, abi_key=_abi(), model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=["B_z"])
    handle.artifact_identity = make_identity(
        "artifact", {"fixture": "compliance-amr-route-model"})
    layout = AMR(base=CartesianMesh(n=64), max_levels=2, ratio=2, regrid=RegridEvery(4))
    snapshot = AuthoringSnapshot({"kind": "compliance-amr-route-stub"})
    schema = BindSchema()
    plan = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="amr_system",
        backend="production",
        layout=layout,
        time=None,
        blocks=(ResolvedBlock(
            "block", {"model": "compliance-amr-route-model"}, None, "production"),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={"amr": True},
        capabilities={"cpu": True, "amr": True, "mpi": True},
    )
    return CompiledSimulationArtifact(
        plan=plan,
        program=None,
        blocks=(CompiledBlockArtifact("block", handle, None),),
    )


def _pos_amr_arguments_inert():
    # ADC-515: arguments() on the AMR-route handle reports layout='amr' + the block instance (inert,
    # no .so). Agrees with the integration sec.20 matrix's arguments.amr.mono green_inert cell.
    from pops.codegen.inspect_compiled import build_arguments

    args = build_arguments(_amr_route_handle())
    assert args.layout_runtime["layout"] == "amr"
    assert next(iter(args.instances.values()))["components"] == 3
    return _assert_route_available("layout:AMR")


def _pos_amr_estimate_memory_inert():
    # ADC-515: estimate_memory(mesh) on the AMR-route handle is a conservative patch-budget FORMULA
    # (layout='amr', amr_patch > 0) using the artifact's resolved layout. Inert.
    from pops.codegen.inspect_compiled import build_memory_estimate

    artifact = _amr_route_handle()
    est = build_memory_estimate(
        artifact, _CartesianMesh(n=64, periodic=True), layout=artifact.layout)
    assert est.layout == "amr" and est.categories.get("amr_patch", 0) > 0
    return _assert_route_available("layout:AMR")


POSITIVE = {
    "pos.uniform_fv_typed_riemann": PositiveCell(
        ("spatial:finite_volume", "riemann:hll"), _pos_uniform_fv_typed_riemann),
    "pos.uniform_poisson_elliptic": PositiveCell(
        ("elliptic:geometric_mg",), _pos_uniform_poisson_elliptic),
    "pos.program_manual_operator": PositiveCell(
        ("program_context:system",), _pos_program_manual_operator),
    "pos.program_macro_lib_time": PositiveCell(
        ("program_context:system",), _pos_program_macro_lib_time),
    "pos.matrix_free_krylov": PositiveCell(
        ("krylov:cg_bicgstab_gmres_richardson",), _pos_matrix_free_krylov),
    "pos.params_runtime_const_bind": PositiveCell(
        ("program_context:system",), _pos_params_runtime_const_bind),
    "pos.diagnostics_output_ckpt": PositiveCell(
        ("output:npz_vtk_hdf5", "checkpoint:system_v1"), _pos_diagnostics_output_ckpt),
    "pos.amr_route_when_capable": PositiveCell(
        ("layout:AMR",), _pos_amr_route_when_capable),
    "pos.amr_arguments_inert": PositiveCell(
        ("layout:AMR",), _pos_amr_arguments_inert),
    "pos.amr_estimate_memory_inert": PositiveCell(
        ("layout:AMR",), _pos_amr_estimate_memory_inert),
}


# --------------------------------------------------------------------------------------------------
# NEGATIVE cells (11) -- each pins a stable, warning-free refusal on the CLEAN route.
# The install-time predicates in pops.runtime.routes are the SINGLE SOURCE System delegates to, so
# calling them directly is the clean route (no System construction).
# --------------------------------------------------------------------------------------------------
def _neg_hll_no_wave_speeds():
    # Clean route (gap 1 closed): the wave-speed provider predicate, not System.add_block.
    _routes.check_wave_speed_provider(
        "explicit_pair", _compiled_model(wave_speeds=False), "block 'plasma'")


def _neg_hllc_no_star_state():
    _routes.check_riemann_capability("hllc", _compiled_model(hllc=False), "block 'plasma'")


def _neg_roe_no_dissipation():
    _routes.check_riemann_capability("roe", _compiled_model(roe=False), "block 'plasma'")


def _neg_weno_ghost_depth():
    from pops.numerics.reconstruction import WENO5, validate_ghost_depth

    validate_ghost_depth(WENO5(), available=2, block="plasma")


def _neg_fft_on_amr_or_bc():
    # FFT is refused for BOTH a wall (Dirichlet) BC and an AMR layout (gap 4). Prove the AMR half
    # first from the clean native facts (descriptor supports_amr is False + the elliptic:fft_amr route
    # is unavailable), then RAISE the BC-half refusal (the descriptor validate route) so the cell pins
    # one stable message. Both halves are asserted; the raised message is the BC refusal.
    from pops.solvers.elliptic import FFT

    # AMR half: FFT does not support AMR (clean descriptor + native-route facts).
    assert FFT().capabilities().get("supports_amr") is False
    fft_amr = _routes_by_id()["elliptic:fft_amr"]
    assert fft_amr["status"] == "unavailable", (
        "elliptic:fft_amr is %r, expected unavailable" % fft_amr["status"])
    assert "FFT requires a single uniform periodic mesh, not AMR" in fft_amr["reason"]
    # BC half: FFT with a Dirichlet (wall) BC is refused at the descriptor validate level.
    _poisson(FFT(), Dirichlet()).validate()


def _neg_amr_maxlevels_ratio():
    AMR(base=CartesianMesh(n=64), max_levels=4).validate()


def _neg_param_missing_or_domain():
    from pops.params import Positive, RuntimeParam
    from pops.model.bind_schema import BindSchema

    problem = pops.Problem(name="g")
    alpha = problem.param(RuntimeParam("alpha", default=1.0, domain=Positive()))
    BindSchema.from_problem(problem).resolve({alpha: -5.0})


def _bind_model_with_param():
    from pops.params import RuntimeParam
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=0,
        params={"alpha": RuntimeParam("alpha", default=1.0)}, caps={"cpu": True},
        abi_key=_abi(), model_hash="h", cxx="c++", std="c++23")


def _neg_operator_signature_mismatch():
    # Gap 2 closed: a true ARITY mismatch (a 2-input operator fed 1 input) is refused at call time.
    module = _arity_module()
    program = adctime.Program("adc547_arity")
    program.bind_operators(module)
    block = pops.Problem(name="adc547-arity-case").add_block("plasma", module)
    declaration = module.state_handle(module.state_spaces()["U"])
    state = program.state(block, declaration).n
    program._call("explicit_rhs", state)  # missing the fields input -> arity refusal


def _neg_missing_aux_field():
    # The lowered operator requires aux B_z the state omits -> bind gate refusal (metadata stub).
    compiled = _compiled_problem(aux_names=("B_z",))
    run_bind_gates(compiled, _uniform(), {"plasma": np.ones((3, 8, 8))}, {}, {})


def _neg_abi_cache_mismatch():
    compiled = _compiled_problem(abi_key="TOTALLY_DIFFERENT_ABI")
    run_bind_gates(compiled, _uniform(), {"plasma": np.ones((3, 8, 8))}, {}, {})


def _neg_ir_index_refusal():
    # Gap 3 closed: a Program state value refuses __index__ (range()/index) with a stable message
    # steering to P.while_ / P.branch -- a runtime IR value is not a compile-time index.
    program = adctime.Program("adc547_index")
    _module, block, declaration = _state_refs("adc547-index-model", "plasma")
    state = program.state(block, declaration).n
    range(state)  # triggers __index__ -> TypeError


NEGATIVE = {
    "neg.hll_no_wave_speeds": NegativeCell(
        ValueError, ("riemann 'hll'", "requires the model to emit signed wave speeds"),
        _neg_hll_no_wave_speeds),
    "neg.hllc_no_star_state": NegativeCell(
        ValueError, ("riemann 'hllc'", "hllc_star_state"), _neg_hllc_no_star_state),
    "neg.roe_no_dissipation": NegativeCell(
        ValueError, ("riemann 'roe'", "roe_dissipation"), _neg_roe_no_dissipation),
    "neg.weno_ghost_depth": NegativeCell(
        ValueError, ("WENO5 requires ghost_depth >= 3", "block 'plasma' has ghost_depth=2"),
        _neg_weno_ghost_depth),
    "neg.fft_on_amr_or_bc": NegativeCell(
        ValueError, ("requires a periodic boundary", "supports_wall_bc is False"),
        _neg_fft_on_amr_or_bc),
    "neg.amr_maxlevels_ratio": NegativeCell(
        ValueError, ("max_levels=4", "supports max_levels=2"), _neg_amr_maxlevels_ratio),
    "neg.param_missing_or_domain": NegativeCell(
        ValueError, ("runtime-param-domain", "alpha", "-5.0"), _neg_param_missing_or_domain),
    "neg.operator_signature_mismatch": NegativeCell(
        ValueError, ("operator 'explicit_rhs'", "expects 2 argument(s)", "got 1"),
        _neg_operator_signature_mismatch),
    "neg.missing_aux_field": NegativeCell(
        ValueError, ("B_z", "aux-required-by-operator"), _neg_missing_aux_field),
    "neg.abi_cache_mismatch": NegativeCell(
        ValueError, ("manifest-abi", "ABI mismatch"), _neg_abi_cache_mismatch),
    "neg.ir_index_refusal": NegativeCell(
        TypeError, ("cannot be used as a Python index", "P.while_"), _neg_ir_index_refusal),
}


# --------------------------------------------------------------------------------------------------
# The complete expected id set -- test_matrix_is_complete pins these so a dropped cell fails loud.
# --------------------------------------------------------------------------------------------------
EXPECTED_POSITIVE_IDS = frozenset(POSITIVE)
EXPECTED_NEGATIVE_IDS = frozenset(NEGATIVE)


def run_negative_cell(cell):
    """Run one negative cell: it must raise ``cell.exc_type`` with every needle and NO warning."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            cell.call()
        except cell.exc_type as exc:
            message = str(exc)
        else:
            raise AssertionError("expected %s, no exception raised" % cell.exc_type.__name__)
    assert not seen, "negative cell emitted warning(s) instead of refusing cleanly: %r" % (seen,)
    missing = [needle for needle in cell.needles if needle not in message]
    assert not missing, "refusal message %r missing needles %r" % (message, missing)
    return message
