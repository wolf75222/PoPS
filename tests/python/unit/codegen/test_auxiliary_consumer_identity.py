"""Exact auxiliary consumer identities stay unique across shared-model blocks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pops.codegen._layout_resolution import layout_lowering_coverage
from pops.identity import artifact_spec_identity, make_identity
from pops.codegen._plans import (
    ResolvedBlock,
    ResolvedSimulationPlan,
    authenticate_block_instance_owner_qid,
    attest_precompiled_consumer_owner,
)
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.program_emit_kernels import program_provider_consumer_qid
from pops.layouts import Uniform
from pops.mesh import normalize_layout_plan
from pops.model import Handle, OwnerPath
from pops.model.bind_schema import BindSchema
from pops.physics._facade import Model
from pops.problem import Case
from pops.problem._snapshot import AuthoringSnapshot
from pops.time import Program
from tests.python.support.block_instance_owner import make_testing_block_instance_owner
from tests.python.support.layout_plan import cartesian_grid


def _transport():
    model = Model("shared_aux_transport")
    (density,) = model.conservative_vars("density")
    model.flux(x=[density])
    model.eigenvalues(x=[density])
    model.primitive_vars(density=density)
    model.conservative_from([density])
    return model


def test_shared_model_block_instance_owners_are_officially_distinct() -> None:
    model = _transport()
    case = Case("aux-consumer-identity")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)

    ion_owner = str(ions.instance_owner_path.canonical())
    electron_owner = str(electrons.instance_owner_path.canonical())

    assert ion_owner != electron_owner
    assert "block:ions" in ion_owner
    assert "block:electrons" in electron_owner
    assert str(model.owner_path.canonical()) in ion_owner
    assert str(model.owner_path.canonical()) in electron_owner


def test_native_registrar_uses_block_instance_consumer_qid() -> None:
    model = _transport()
    case = Case("aux-consumer-identity-emit")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_owner = str(ions.instance_owner_path.canonical())
    electron_owner = str(electrons.instance_owner_path.canonical())
    model_owner = str(model.owner_path.canonical())

    ion_src = model.__pops_native_loader_source__(
        name="ions", target="system", consumer_owner_qid=ion_owner
    )
    electron_src = model.__pops_native_loader_source__(
        name="electrons", target="system", consumer_owner_qid=electron_owner
    )

    ion_plan = ion_owner + "/physical_flux"
    electron_plan = electron_owner + "/physical_flux"
    model_plan = model_owner + "/physical_flux"

    assert 'ConsumerPlan{"%s"' % ion_plan in ion_src
    assert 'ConsumerPlan{"%s"' % electron_plan in electron_src
    assert ion_plan != electron_plan
    assert 'ConsumerPlan{"%s"' % model_plan not in ion_src
    assert 'ConsumerPlan{"%s"' % model_plan not in electron_src
    assert 'ConsumerPlan{"%s"' % ion_plan not in electron_src
    assert 'ConsumerPlan{"%s"' % electron_plan not in ion_src


def test_program_consumer_qid_uses_block_instance_owner() -> None:
    model = _transport()
    case = Case("aux-consumer-identity-program")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)

    ion_qid = program_provider_consumer_qid(model, 4, ions)
    electron_qid = program_provider_consumer_qid(model, 4, electrons)
    model_qid = program_provider_consumer_qid(model, 4)

    assert ion_qid != electron_qid
    assert ion_qid == str(ions.instance_owner_path.canonical()) + "/program/4"
    assert electron_qid == str(electrons.instance_owner_path.canonical()) + "/program/4"
    assert model_qid == str(model.owner_path.canonical()) + "/program/4"
    assert model_qid != ion_qid


def test_program_consumer_qid_refuses_a_block_without_canonical_instance_owner() -> None:
    model = _transport()
    with pytest.raises(ValueError, match="canonical instance_owner_path"):
        program_provider_consumer_qid(model, 4, SimpleNamespace(name="ions"))


def test_resolved_plan_requires_and_identifies_instance_owner() -> None:
    owner = make_testing_block_instance_owner("aux-plan", "fluid", "transport")
    other = make_testing_block_instance_owner("aux-plan", "plasma", "transport")
    plan = _resolved_plan("fluid", owner)
    other_plan = _resolved_plan("plasma", other)

    assert plan.blocks[0].instance_owner_qid == str(owner)
    assert plan._payload()["blocks"][0]["instance_owner_qid"] == str(owner)
    assert plan.plan_identity != other_plan.plan_identity
    plan.verify()
    object.__setattr__(plan.blocks[0], "instance_owner_qid", str(other))
    with pytest.raises(ValueError, match="identity verification failed"):
        plan.verify()


def test_resolved_block_refuses_a_raw_string_owner() -> None:
    with pytest.raises(ValueError, match="canonical OwnerPath string"):
        ResolvedBlock(
            "fluid",
            _Canonical("model"),
            None,
            "production",
            ("U",),
            ("test::fluid::state::U",),
            "not-an-authenticated-owner",
        )


def test_resolved_block_round_trips_its_canonical_owner_qid() -> None:
    owner = make_testing_block_instance_owner("aux-plan", "fluid", "transport")
    original = _resolved_plan("fluid", owner).blocks[0]

    restored = replace(original, instance_owner_qid=original.instance_owner_qid)

    assert restored.instance_owner_qid == str(owner)
    assert restored.instance_owner == owner.to_data()
    assert restored.model_owner == original.model_owner


@pytest.mark.parametrize(
    "qid",
    (
        "not-an-authenticated-owner",
        "case:aux-plan#authoring=7/block:fluid/model_definition:transport@"
        "pops.model:sha256:" + "0" * 64,
        "case:%61ux-plan/block:fluid/model_definition:transport@"
        "pops.model:sha256:" + "0" * 64,
    ),
)
def test_resolved_block_refuses_malformed_authoring_or_noncanonical_owner_qid(qid: str) -> None:
    with pytest.raises(ValueError, match="canonical OwnerPath string"):
        ResolvedBlock(
            "fluid",
            _Canonical("model"),
            None,
            "production",
            ("U",),
            ("test::fluid::state::U",),
            qid,
        )


def test_resolved_plan_refuses_an_empty_instance_owner() -> None:
    with pytest.raises(ValueError, match="canonical Case-block instance owner"):
        _resolved_plan("fluid", "")


def test_compiled_plan_identity_includes_instance_owner() -> None:
    owner = make_testing_block_instance_owner("aux-compiled", "fluid", "transport")
    other = make_testing_block_instance_owner("aux-compiled", "plasma", "transport")
    record = CompiledPlanRecord.from_resolved(_resolved_plan("fluid", owner))
    other_record = CompiledPlanRecord.from_resolved(_resolved_plan("plasma", other))

    assert record.blocks[0].instance_owner_qid == str(owner)
    assert record._payload()["blocks"][0]["instance_owner_qid"] == str(owner)
    assert record.contract_identity != other_record.contract_identity
    record.verify()
    object.__setattr__(record.blocks[0], "instance_owner_qid", str(other))
    with pytest.raises(ValueError, match="identity verification failed"):
        record.verify()


def test_artifact_spec_separates_shared_model_block_owners() -> None:
    model = _transport()
    case = Case("aux-consumer-cache")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_owner = authenticate_block_instance_owner_qid(ions)
    electron_owner = authenticate_block_instance_owner_qid(electrons)
    semantic = make_identity("semantic", {"kind": "model", "name": model.name})

    def spec(owner: str) -> object:
        return artifact_spec_identity(
            semantic,
            target="system",
            backend="production",
            precision="binary64",
            abi="test-abi",
            toolchain="c++|c++23",
            routes={"registry": "test", "features": "test"},
            components={"model_hash": "x", "emitted_name": "ions", "consumer_owner_qid": owner},
            flags=[],
            libraries=(),
        )

    assert spec(ion_owner) != spec(electron_owner)
    assert spec(ion_owner) != spec("")


def test_precompiled_consumer_owner_match_and_mismatch() -> None:
    model = _transport()
    case = Case("aux-precompiled")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_owner = authenticate_block_instance_owner_qid(ions)
    electron_owner = authenticate_block_instance_owner_qid(electrons)
    baked = SimpleNamespace(consumer_owner_qid=ion_owner, declares_auxiliary_providers=True)
    attest_precompiled_consumer_owner(baked, ion_owner)
    attest_precompiled_consumer_owner(
        baked, ion_owner, declare_auxiliary_providers=True)
    with pytest.raises(ValueError, match="recompile"):
        attest_precompiled_consumer_owner(baked, electron_owner)
    with pytest.raises(ValueError, match="provider-declaration role"):
        attest_precompiled_consumer_owner(
            baked, ion_owner, declare_auxiliary_providers=False)
    with pytest.raises(ValueError, match="canonical Case-block instance owner"):
        attest_precompiled_consumer_owner(baked, "")


def _aux_model(name: str) -> Model:
    model = Model(name)
    (density,) = model.conservative_vars("density")
    rate = model.aux("collision_rate")
    model.flux(x=[rate * density])
    model.eigenvalues(x=[density])
    model.primitive_vars(density=density)
    model.conservative_from([density])
    model.source([-(rate * density)])
    return model


def test_shared_model_sibling_emits_consumers_only() -> None:
    model = _aux_model("shared_aux_providers")
    case = Case("aux-provider-single-declaration")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_owner = authenticate_block_instance_owner_qid(ions)
    electron_owner = authenticate_block_instance_owner_qid(electrons)
    declared = model.__pops_native_loader_source__(
        name="ions", target="system", consumer_owner_qid=ion_owner,
        declare_auxiliary_providers=True)
    sibling = model.__pops_native_loader_source__(
        name="electrons", target="system", consumer_owner_qid=electron_owner,
        declare_auxiliary_providers=False)
    provider = "install_prepared_auxiliary_provider(Provider{"
    assert declared.count(provider) == 1
    assert provider not in sibling
    assert 'ConsumerPlan{"%s/physical_flux"' % ion_owner in declared
    assert 'ConsumerPlan{"%s/physical_flux"' % electron_owner in sibling
    assert str(model.owner_path.canonical()) in declared


def test_duplicate_provider_emission_is_identical_and_must_not_both_register() -> None:
    model = _aux_model("shared_aux_duplicate")
    case = Case("aux-provider-duplicate")
    ions = case.block("ions", model)
    electrons = case.block("electrons", model)
    ion_src = model.__pops_native_loader_source__(
        name="ions", target="system",
        consumer_owner_qid=authenticate_block_instance_owner_qid(ions),
        declare_auxiliary_providers=True)
    electron_src = model.__pops_native_loader_source__(
        name="electrons", target="system",
        consumer_owner_qid=authenticate_block_instance_owner_qid(electrons),
        declare_auxiliary_providers=True)
    marker = "install_prepared_auxiliary_provider(Provider{"
    ion_decl = ion_src[ion_src.index(marker):ion_src.index("install_auxiliary_consumer_plan")]
    electron_decl = electron_src[
        electron_src.index(marker):electron_src.index("install_auxiliary_consumer_plan")]
    assert ion_decl == electron_decl
    assert ion_decl.count(marker) == 1


def test_plan_selects_one_declaration_owner_and_reorder_changes_identity() -> None:
    model = _Canonical("shared")
    first = _two_block_plan(("ions", "electrons"), model)
    second = _two_block_plan(("electrons", "ions"), model)
    assert first.blocks[0].declares_auxiliary_providers is True
    assert first.blocks[1].declares_auxiliary_providers is False
    assert second.blocks[0].name == "electrons"
    assert second.blocks[0].declares_auxiliary_providers is True
    assert second.blocks[1].declares_auxiliary_providers is False
    assert first.plan_identity != second.plan_identity
    first_record = CompiledPlanRecord.from_resolved(first)
    second_record = CompiledPlanRecord.from_resolved(second)
    assert first_record.contract_identity != second_record.contract_identity
    assert first_record.blocks[0].declares_auxiliary_providers is True
    assert first_record.blocks[1].declares_auxiliary_providers is False


def test_declaration_role_separates_cache_identity() -> None:
    semantic = make_identity("semantic", {"kind": "model", "name": "shared"})

    def spec(*, declare: bool) -> object:
        return artifact_spec_identity(
            semantic,
            target="system",
            backend="production",
            precision="binary64",
            abi="test-abi",
            toolchain="c++|c++23",
            routes={"registry": "test", "features": "test"},
            components={
                "model_hash": "x",
                "emitted_name": "ions",
                "consumer_owner_qid": "owner",
                "declares_auxiliary_providers": declare,
            },
            flags=[],
            libraries=(),
        )

    assert spec(declare=True) != spec(declare=False)


def test_instance_owner_block_name_must_match() -> None:
    owner = make_testing_block_instance_owner("aux-plan", "ions", "transport")
    with pytest.raises(ValueError, match="does not match resolved block name"):
        ResolvedBlock(
            "electrons",
            _Canonical("model"),
            None,
            "production",
            ("U",),
            ("test::electrons::state::U",),
            owner,
        )


def test_instance_owner_model_must_match_resolved_model() -> None:
    from pops.codegen._plans import canonical_block_instance_owner
    from tests.python.support.block_instance_owner import testing_model_owner

    model = _transport()
    _ = model.module
    wrong = canonical_block_instance_owner(
        case="aux-model-mismatch", block="ions", model_owner=testing_model_owner("other"))
    with pytest.raises(ValueError, match="does not match resolved model owner"):
        ResolvedBlock(
            "ions",
            model,
            None,
            "production",
            ("U",),
            ("test::ions::state::U",),
            wrong,
        )


def test_public_amr_shared_model_bind_if_native_available() -> None:
    pytest.importorskip("pops._pops")
    from pops.amr import (
        AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
        Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
    )
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Constant
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, every
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    missing = missing_compiler_requirement()
    if missing:
        pytest.skip(missing)

    import pops

    frame = Rectangle("aux-amr-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("aux_amr_shared", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    x_axis, y_axis = frame.axes
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (rho * 0.0 + 1.0,), y_axis: (rho * 0.0 + 1.0,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("aux-amr-shared-model")
    ions = case.block("ions", model, states=(state,))
    electrons = case.block("electrons", model, states=(state,))
    for block, instance in ((ions, ions[state]), (electrons, electrons[state])):
        numerics = DiscretizationPlan()
        numerics.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(numerics, block=block)
        case.initials.add(
            InitialCondition(
                state=instance, value=Constant((1.0,)), projection=ConservativeCellAverage()
            )
        )
    import pops.lib.time as libtime

    program = libtime.RungeKutta(
        routes=(
            libtime.RungeKuttaRoute(ions[state], rate),
            libtime.RungeKuttaRoute(electrons[state], rate),
        ),
        tableau=libtime.SSPRK2_TABLEAU,
    )
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)
    transfer = AMRTransfer()
    transfer.state(ions[state], StateTransfer())
    transfer.state(electrons[state], StateTransfer())
    refine_threshold = case.param(RuntimeParam("aux_shared_refine_threshold", default=2.0))
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(ions[state]) > case.value(refine_threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": repo_include()},
    )
    owners = [block.instance_owner_qid for block in resolved.blocks]
    assert len(owners) == 2
    assert owners[0] != owners[1]
    compiled = pops.compile(resolved)
    bound = pops.bind(compiled, {})
    assert bound is not None


class _Canonical:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_data(self) -> dict[str, str]:
        return {"name": self.name}


def _two_block_plan(order: tuple[str, str], model: object) -> ResolvedSimulationPlan:
    case_owner = OwnerPath.case("aux-decl")
    layout_plan = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)),
        owner=case_owner,
        blocks=tuple(Handle(name, kind="block", owner=case_owner) for name in order),
    )
    return ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "aux-decl", "order": list(order)}),
        target="system",
        backend="production",
        layout={"mesh": {"shape": [8, 8]}},
        layout_plan=layout_plan,
        layout_targets={
            row.handle.qualified_id: "system" for row in layout_plan.layouts
        },
        time=Program("rk2"),
        blocks=tuple(
            ResolvedBlock(
                name,
                model,
                {"flux": ["hll"]},
                "production",
                ("U",),
                ("test::%s::state::U" % name,),
                make_testing_block_instance_owner("aux-decl", name, "shared"),
            )
            for name in order
        ),
        bind_schema=BindSchema(),
        compile_values={},
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={"cpu": True},
        lowering_coverage=layout_lowering_coverage(layout_plan),
    )


def _resolved_plan(block_name: str, instance_owner: object) -> ResolvedSimulationPlan:
    case_owner = OwnerPath.case("aux-plan")
    layout_plan = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)),
        owner=case_owner,
        blocks=(Handle(block_name, kind="block", owner=case_owner),),
    )
    return ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "aux-plan", "block": block_name}),
        target="system",
        backend="production",
        layout={"mesh": {"shape": [8, 8]}},
        layout_plan=layout_plan,
        layout_targets={
            row.handle.qualified_id: "system" for row in layout_plan.layouts
        },
        time=Program("rk2"),
        blocks=(
            ResolvedBlock(
                block_name,
                _Canonical("model"),
                {"flux": ["hll"]},
                "production",
                ("U",),
                ("test::%s::state::U" % block_name,),
                instance_owner,
            ),
        ),
        bind_schema=BindSchema(),
        compile_values={},
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={"cpu": True},
        lowering_coverage=layout_lowering_coverage(layout_plan),
    )
