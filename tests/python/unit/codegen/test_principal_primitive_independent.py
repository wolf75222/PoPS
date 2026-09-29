"""Independent source checks for joint primitive identity and manifest migration."""

import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.frames import Cartesian2D
from pops.model._module_manifest import ModuleManifest
from tests.python.support.principal_primitive_case import primitive_case


def _model(*, inverse_scale=1.0, domain_floor=0.0):
    model = pops.Model("joint-identity", frame=Cartesian2D())
    density = model.species("density", state=("rho",))
    momentum = model.species("momentum", state=("m",))
    rho, = density
    m, = momentum
    velocity = model.primitive("velocity", m / rho)
    model.primitive_state(
        rho, velocity, states=(density, momentum),
        conservative=(rho, inverse_scale * rho * velocity),
    )
    model.recovery_admissibility(
        states=(density, momentum), rho=rho > domain_floor,
    )
    return model


def test_inverse_and_domain_are_both_part_of_artifact_identity():
    base = _model()
    altered_inverse = _model(inverse_scale=2.0)
    altered_domain = _model(domain_floor=0.1)
    identities = {model.module.module_hash() for model in
                  (base, altered_inverse, altered_domain)}
    assert len(identities) == 3
    manifests = {model.module.manifest().hash for model in
                 (base, altered_inverse, altered_domain)}
    assert len(manifests) == 3


def test_schema_nine_cannot_be_rehydrated_as_joint_coordinate_schema_ten():
    payload = _model().module.manifest().to_dict()
    assert payload["schema_version"] == 10
    assert ModuleManifest.from_dict(payload).to_dict() == payload
    old = dict(payload, schema_version=9)
    with pytest.raises(ValueError, match="unsupported ModuleManifest schema_version"):
        ModuleManifest.from_dict(old)


def test_seven_components_across_reversed_rows_and_map_emit_checked_conversions():
    case, layout, *_ = primitive_case((3, 4), reverse=True, map_reverse=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "n_vars = 7" in source
    assert "component_counts{3,4}" in source
    assert "StateConversionStatus::InvalidEquationOfState" in source
    assert "result.status=pops::nd::StateConversionStatus::Success" in source
    assert "if(!Kokkos::isfinite" in source
