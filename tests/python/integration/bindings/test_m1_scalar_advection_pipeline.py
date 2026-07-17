"""One real scalar-advection assembly crosses every typed lifecycle phase."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pops
from pops.mesh.boundaries import GhostProducerPlan
from pops.mesh.boundaries.composition import compose_boundary_plans


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


class _ThirdPartyBoundaryComposer:
    """Small extension fixture proving the central dispatcher does not name implementations."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def canonical_identity(self):
        return {
            "authority_type": "third_party_boundary_composer",
            "plan": self.plan.canonical_identity(),
        }

    def ghost_plan_composer_capability(self):
        return {"schema_version": 1, "scope": "self"}

    def compose_ghost_plan(self, context):
        assert context.authorities == (self,)
        self.calls += 1
        return self.plan


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_m1_scalar_advection", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scalar_advection_completes_typed_phase_pipeline():
    example = _load_example()
    target = example.build_final_case(output_mode=example._native_output_mode())

    validated = pops.validate(target.authoring.case)
    resolved = pops.resolve(validated, layout=target.layout)
    boundary_plan = resolved.blocks[0].numerics.boundaries[0]
    assert isinstance(boundary_plan, GhostProducerPlan)
    assert len(boundary_plan.productions) == len(boundary_plan.regions) == 6
    assert resolved.blocks[0].state_identities == (
        boundary_plan.compile_boundary_data()["state"]["qualified_id"],
    )
    extension = _ThirdPartyBoundaryComposer(boundary_plan)
    extension_numerics = replace(
        resolved.blocks[0].numerics, boundaries=(extension,))
    recomposed = compose_boundary_plans(
        extension_numerics,
        layout_plan=resolved.layout_plan,
        amr_transfer=resolved.amr_transfer,
    )
    assert recomposed.boundaries == (boundary_plan,)
    assert extension.calls == 1
    assert [row.producer.qualified_id for row in boundary_plan.execution_order()] \
        == boundary_plan.runtime_boundary_data(
            resolved.bind_schema.resolve_bind(
                example.build_bind_params(target.authoring),
                compile_values=resolved.compile_values,
            )
        )["producer_order"]
    artifact = pops.compile(resolved)
    simulation = example._bind_artifact(
        artifact,
        params=example.build_bind_params(target.authoring),
    )

    assert validated is target.authoring.case and validated.frozen
    assert artifact.plan.plan_identity == resolved.plan_identity
    assert artifact.plan is not resolved
    artifact.verify()
    assert simulation._executor.lifecycle_state() == "bound"
    installed = simulation._executor._boundary_authorities["tracer"]
    assert installed["ghost_plan_identity"] == boundary_plan.canonical_id
    assert installed["required_depth"] == \
        resolved.blocks[0].numerics.primary_spatial().ghost_depth
    assert simulation.bind_identity.domain == "bind"


def test_scalar_advection_final_example_runs_outputs_and_bit_identical_restart(tmp_path):
    example = _load_example()

    evidence = example.run_manual_and_restart(tmp_path / "final-scalar-acceptance")
    preset = example.run_preset_parity(
        tmp_path / "final-scalar-preset", evidence.continuous)

    assert evidence.hdf5_path.is_file()
    assert evidence.paraview_path.is_file()
    assert evidence.checkpoint_path.is_file()
    assert evidence.accepted.macro_step > 0
    assert evidence.restored.macro_step == evidence.accepted.macro_step
    assert evidence.continuous.macro_step == evidence.restarted.macro_step
    assert preset.macro_step == evidence.continuous.macro_step
    assert preset.program_hash == evidence.continuous.program_hash
    from pops.output import read_paraview

    reopened = read_paraview(evidence.paraview_path)
    diagnostics = reopened.manifest["snapshot"]["diagnostics"]
    assert {row["key"]["reduction"] for row in diagnostics} == {
        "integral",
        "l1",
        "l2",
        "linf",
        "min",
        "max",
    }
