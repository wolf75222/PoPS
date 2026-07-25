"""ADC-537: the pure bind-time refusal core (host-testable, no engine).

``pops.bind`` refuses a bad install with precise context BEFORE the native artifact is loaded. The
four gates are pure functions over plain metadata (a manifest / arguments stand-in, the mesh layout,
the declared runtime params, the supplied initial state), so they are exercised here with plain
Python objects -- no compiled ``.so``, no ``_pops``. The compiler-gated end-to-end refusal lives in
the integration tier; this tier proves the refusal LOGIC.
"""
from types import SimpleNamespace

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.runtime import _bind_validation as bv  # noqa: E402
from pops.params.runtime import RuntimeParam  # noqa: E402
from pops.params.constraints import Positive, Interval  # noqa: E402


class _Manifest:
    """A minimal stand-in for CompiledArtifactManifest (only the fields the gates read)."""

    def __init__(self, *, abi_key="H|clang|c++17", ghost_depth=2, precision="double",
                 supports_mpi=None, supports_gpu=None, communicator="unknown",
                 aux_required=(), operators=()):
        self.abi_key = abi_key
        self.ghost_depth = ghost_depth
        self.precision = precision
        self.supports_mpi = supports_mpi
        self.supports_gpu = supports_gpu
        self.communicator = communicator
        self.aux_required = list(aux_required)
        self.operators = list(operators)


class _Arguments:
    def __init__(self, instances):
        self.instances = dict(instances)


class _Mesh:
    def __init__(self, n):
        self.n = n


class _Uniform:
    def __init__(self, n):
        self.mesh = _Mesh(n)


class _FramedUniform:
    """Stand-in for the public CartesianGrid, whose extent is carried by ``cells``."""

    def __init__(self, n):
        self.mesh = SimpleNamespace(cells=(n, n))


class _MultiLayout:
    """Minimal exact multi-layout authority used by the pure bind gate."""

    def __init__(self, assignments):
        handles = {name: object() for name in assignments}
        self.rows = tuple(SimpleNamespace(handle=handles[name], descriptor=layout)
                          for name, layout in assignments.items())
        self.plan = SimpleNamespace(assignments=tuple(
            SimpleNamespace(
                subject_kind="block",
                subject=SimpleNamespace(local_id=name),
                layout=handles[name],
            )
            for name in assignments
        ))

    def descriptor(self, handle):
        matches = [row.descriptor for row in self.rows if row.handle is handle]
        if len(matches) != 1:
            raise KeyError("unknown layout handle")
        return matches[0]


class _AMR:
    """An AMR layout stand-in exposing capabilities() and runtime_layout_data()."""

    def __init__(self, n):
        self.cells = (n, n) if isinstance(n, int) else tuple(n)

    def capabilities(self):
        return {"layout": "amr"}

    def runtime_layout_data(self):
        return {"grid": {"cells": list(self.cells)}}


class _Array:
    """A duck-typed array carrying only .shape / .dtype (no numpy dependency)."""

    def __init__(self, shape, dtype="float64"):
        self.shape = tuple(shape)
        self.dtype = type("D", (), {"name": dtype})()


class _BoundSubject:
    def __init__(self, block):
        self.block_ref = SimpleNamespace(local_id=block)


def _one_block_args(components=1):
    return _Arguments({"ne": {"state": "U", "components": components, "required": True}})


# ---------------------------------------------------------------------------
# Gate d -- initial_state shape / dtype / components / ghost
# ---------------------------------------------------------------------------

def test_valid_initial_state_passes():
    manifest, args, layout = _Manifest(), _one_block_args(1), _Uniform(64)
    lines = bv.validate_initial_state(manifest, args, layout, {"ne": _Array((64, 64))})
    assert lines == []


def test_haloed_initial_state_shape_passes():
    manifest, args, layout = _Manifest(ghost_depth=2), _one_block_args(4), _Uniform(64)
    # A ghost-ringed (4, 68, 68) array is accepted alongside the valid (4, 64, 64).
    assert bv.validate_initial_state(manifest, args, layout, {"ne": _Array((4, 68, 68))}) == []
    assert bv.validate_initial_state(manifest, args, layout, {"ne": _Array((4, 64, 64))}) == []


def test_amr_initial_state_table_is_refused_for_every_shape():
    manifest, args, layout = _Manifest(), _one_block_args(4), _AMR(64)
    for shape in ((64, 64), (64 * 64,), (4, 64, 64), (3, 64, 64)):
        lines = bv.validate_initial_state(manifest, args, layout, {"ne": _Array(shape)})
        assert len(lines) == 1
        assert "initial_state for AMR block 'ne' is not a supported authority" in lines[0]
        assert "InitialConditionPlan" in lines[0]
        assert "initial_values" in lines[0]


def test_typed_amr_initial_value_requires_complete_state():
    manifest, args, layout = _Manifest(), _one_block_args(4), _AMR(64)
    subject = _BoundSubject("ne")
    assert bv.validate_bound_initial_values(
        manifest, args, layout, {subject: _Array((4, 64, 64))}) == []
    lines = bv.validate_bound_initial_values(
        manifest, args, layout, {subject: _Array((64, 64))})
    assert len(lines) == 1
    assert "BindArray requires the complete state" in lines[0]


def test_typed_amr_initial_value_preserves_rectangular_axis_order():
    manifest, args, layout = _Manifest(), _one_block_args(4), _AMR((24, 10))
    subject = _BoundSubject("ne")
    assert bv.validate_bound_initial_values(
        manifest, args, layout, {subject: _Array((4, 10, 24))}) == []
    lines = bv.validate_bound_initial_values(
        manifest, args, layout, {subject: _Array((4, 24, 10))})
    assert len(lines) == 1
    assert "BindArray requires the complete state" in lines[0]


def test_typed_uniform_bind_array_uses_cartesian_grid_cells():
    manifest, args, layout = _Manifest(), _one_block_args(2), _FramedUniform(32)
    subject = _BoundSubject("ne")
    assert bv.validate_bound_initial_values(
        manifest, args, layout, {subject: _Array((2, 32, 32))},
    ) == []


def test_wrong_shape_is_refused():
    manifest, args, layout = _Manifest(), _one_block_args(4), _Uniform(64)
    lines = bv.validate_initial_state(manifest, args, layout, {"ne": _Array((4, 32, 32))})
    assert len(lines) == 1
    assert "has shape (4, 32, 32)" in lines[0]
    assert "ghost depth 2" in lines[0]


def test_wrong_dtype_is_refused():
    manifest, args, layout = _Manifest(precision="double"), _one_block_args(1), _Uniform(64)
    lines = bv.validate_initial_state(manifest, args, layout, {"ne": _Array((64, 64), "float32")})
    assert any("dtype 'float32'" in l and "float64" in l for l in lines)


def test_unknown_block_is_refused():
    manifest, args, layout = _Manifest(), _one_block_args(1), _Uniform(64)
    lines = bv.validate_initial_state(manifest, args, layout, {"bogus": _Array((64, 64))})
    assert any("unknown block 'bogus'" in l for l in lines)


def test_missing_ghost_depth_manifest_is_refused_as_abi_incomplete():
    manifest = _Manifest(ghost_depth=None)
    lines = bv.validate_initial_state(manifest, _one_block_args(1), _Uniform(64),
                                      {"ne": _Array((64, 64))})
    assert any("no ghost_depth" in l and "ABI-incomplete" in l for l in lines)


def test_multi_layout_uses_exact_per_block_ghost_depth_and_mesh():
    manifest = _Manifest(ghost_depth=None)
    manifest.ghost_depth_by_block = {"fine": 3, "coarse": 1}
    args = _Arguments({
        "fine": {"state": "U", "components": 1, "required": True},
        "coarse": {"state": "U", "components": 1, "required": True},
    })
    layout = _MultiLayout({"fine": _Uniform(16), "coarse": _Uniform(8)})

    assert bv.validate_initial_state(
        manifest,
        args,
        layout,
        {"fine": _Array((1, 22, 22)), "coarse": _Array((1, 10, 10))},
    ) == []


def test_multi_layout_refuses_partial_per_block_ghost_authority():
    manifest = _Manifest(ghost_depth=3)
    manifest.ghost_depth_by_block = {"fine": 3}
    args = _Arguments({
        "fine": {"state": "U", "components": 1, "required": True},
        "coarse": {"state": "U", "components": 1, "required": True},
    })
    layout = _MultiLayout({"fine": _Uniform(16), "coarse": _Uniform(8)})

    lines = bv.validate_initial_state(
        manifest,
        args,
        layout,
        {"fine": _Array((16, 16)), "coarse": _Array((8, 8))},
    )
    assert len(lines) == 1
    assert "block 'coarse'" in lines[0]
    assert "ghost_depth_by_block" in lines[0]


def test_non_array_initial_state_is_refused():
    lines = bv.validate_initial_state(_Manifest(), _one_block_args(1), _Uniform(64), {"ne": [1, 2, 3]})
    assert any("not an array" in l for l in lines)


# ---------------------------------------------------------------------------
# Gate c -- runtime param domain enforcement (ADC-541: the 4-part message)
# ---------------------------------------------------------------------------

def test_param_out_of_domain_is_refused_with_four_part_message():
    decl = {"alpha": RuntimeParam("alpha", default=1.0, domain=Positive())}
    lines = bv.validate_runtime_param_domains(decl, {"alpha": -3.0})
    assert len(lines) == 1
    msg = lines[0]
    assert "alpha" in msg and "-3.0" in msg and "bind" in msg  # param, value, phase


def test_param_in_domain_passes():
    decl = {"beta": RuntimeParam("beta", default=0.5, domain=Interval(0.0, 1.0))}
    assert bv.validate_runtime_param_domains(decl, {"beta": 0.3}) == []


def test_unknown_param_name_is_left_to_the_artifact_check():
    # A supplied name declared by nothing is NOT this gate's job (no duplicate refusal).
    assert bv.validate_runtime_param_domains({}, {"ghost": 1.0}) == []


def test_const_declaration_carries_no_bind_domain_check():
    # A non-runtime declaration (no check_bind) is skipped by the domain gate.
    assert bv.validate_runtime_param_domains({"gamma": object()}, {"gamma": 5.0}) == []


# ---------------------------------------------------------------------------
# Gate b -- ABI / Kokkos / MPI / layout manifest checks
# ---------------------------------------------------------------------------

def test_matching_abi_and_features_pass():
    manifest = _Manifest(abi_key="ABI1", supports_mpi=True, precision="double")
    facts = {"abi_key": "ABI1", "supports_mpi": True, "precision": "double",
             "communicator": "unknown", "supports_gpu": None}
    assert bv.validate_bind_manifest(manifest, facts) == []


def test_abi_mismatch_is_refused():
    manifest = _Manifest(abi_key="ABI_OLD")
    lines = bv.validate_bind_manifest(manifest, {"abi_key": "ABI_NEW"})
    assert any("ABI mismatch" in l and "ABI_OLD" in l and "ABI_NEW" in l for l in lines)


def test_missing_abi_key_is_refused_as_unverifiable():
    lines = bv.validate_bind_manifest(_Manifest(abi_key=None), {})
    assert any("no abi_key" in l and "ABI-unverifiable" in l for l in lines)


def test_mpi_required_but_runtime_lacks_it_is_refused():
    arguments = SimpleNamespace(layout_runtime={"requires_mpi": True})
    lines = bv.validate_layout_runtime_requirements(arguments, {"supports_mpi": False})
    assert any("MPI support mismatch" in l for l in lines)


def test_gpu_required_but_runtime_lacks_it_is_refused():
    arguments = SimpleNamespace(layout_runtime={"requires_gpu": True})
    lines = bv.validate_layout_runtime_requirements(arguments, {"supports_gpu": False})
    assert any("GPU / Kokkos" in l for l in lines)


def test_more_capable_runtime_is_not_a_mismatch():
    # A CPU-only artifact (supports_gpu/mpi False) binds fine on a Kokkos/MPI-capable runtime:
    # the runtime being MORE capable than the artifact needs is NOT a mismatch (directional gate).
    manifest = _Manifest(abi_key="A", supports_mpi=False, supports_gpu=False)
    facts = {"abi_key": "A", "supports_mpi": True, "supports_gpu": True}
    assert bv.validate_bind_manifest(manifest, facts) == []


def test_supported_feature_is_not_an_execution_requirement():
    manifest = _Manifest(abi_key="A", supports_mpi=True, supports_gpu=True)
    facts = {"abi_key": "A", "supports_mpi": False, "supports_gpu": False}
    assert bv.validate_bind_manifest(manifest, facts) == []


def test_honest_unknown_runtime_token_is_skipped_not_a_fallback():
    # supports_mpi known on the manifest but UNKNOWN (None) on the runtime -> not adjudicable, skipped.
    manifest = _Manifest(abi_key="A", supports_mpi=True)
    assert bv.validate_bind_manifest(manifest, {"abi_key": "A", "supports_mpi": None}) == []


# ---------------------------------------------------------------------------
# Gate b -- the ABI comparison is LIKE-WITH-LIKE across the TWO representations
# (regression: the PR #453 gate false-positived on a legitimately built artifact
# by comparing the artifact's '<headers>|<cxx>|<std>' key against the runtime's
# 'compiler=..;std=..;headers=..;...' env string as raw strings).
# ---------------------------------------------------------------------------

_SHA = "157c5531" + "a" * 56  # a full 64-hex headers signature (prefix from the CI log)
# The two representations of the SAME identity, verbatim shapes from the failed gate log.
_ARTIFACT_KEY = "%s|/usr/bin/c++|c++20" % _SHA
_RUNTIME_KEY = ("compiler=13.3.0;std=202002L;headers=%s;kokkos=1;stdlib=libstdc++_20240904" % _SHA)


def test_same_identity_across_representations_is_not_a_mismatch():
    # The CI regression verbatim: same headers sha, same std (c++20 == 202002L), different
    # spellings and an incomparable compiler token (path vs version) -> NO refusal.
    manifest = _Manifest(abi_key=_ARTIFACT_KEY)
    assert bv.validate_bind_manifest(manifest, {"abi_key": _RUNTIME_KEY}) == []


def test_headers_signature_mismatch_across_representations_is_refused():
    other = "deadbeef" + "b" * 56
    manifest = _Manifest(abi_key="%s|/usr/bin/c++|c++20" % other)
    lines = bv.validate_bind_manifest(manifest, {"abi_key": _RUNTIME_KEY})
    assert any("ABI mismatch" in l and "headers signature" in l for l in lines)
    # The full keys of BOTH sides are quoted for context.
    assert any(other in l and _SHA in l for l in lines)


def test_std_mismatch_is_refused_after_normalization():
    # Same headers sha but a genuinely different standard (c++17 vs 202002L) -> refused.
    manifest = _Manifest(abi_key="%s|/usr/bin/c++|c++17" % _SHA)
    lines = bv.validate_bind_manifest(manifest, {"abi_key": _RUNTIME_KEY})
    assert any("C++ standard mismatch" in l and "201703" in l and "202002" in l for l in lines)


def test_unparseable_std_token_is_skipped_not_refused():
    # A std token the parser does not understand is honest-unknown: never a refusal.
    manifest = _Manifest(abi_key="%s|/usr/bin/c++|weird-std" % _SHA)
    assert bv.validate_bind_manifest(manifest, {"abi_key": _RUNTIME_KEY}) == []


def test_abi_components_parses_both_representations():
    assert bv._abi_components(_ARTIFACT_KEY) == (_SHA, "202002")
    assert bv._abi_components(_RUNTIME_KEY) == (_SHA, "202002")
    # An opaque token (neither form) anchors on the whole string, std honest-unknown.
    assert bv._abi_components("OPAQUE_TOKEN") == ("OPAQUE_TOKEN", None)


def test_normalize_std_spellings():
    assert bv._normalize_std("c++20") == "202002"
    assert bv._normalize_std("202002L") == "202002"
    assert bv._normalize_std("202002") == "202002"
    assert bv._normalize_std("gnu++17") == "201703"
    assert bv._normalize_std("not-a-std") is None
    assert bv._normalize_std(None) is None


# ---------------------------------------------------------------------------
# Gate b -- communicator / precision: 'unknown' is honest-unknown; the check is
# directional (regression: communicator='unknown' vs runtime 'serial' was refused).
# ---------------------------------------------------------------------------

def test_unknown_communicator_on_the_artifact_is_skipped():
    # The CI regression verbatim: the artifact declares communicator='unknown' (honest-unknown)
    # and the runtime reports 'serial' -> SKIPPED, never refused.
    manifest = _Manifest(abi_key="A", communicator="unknown")
    assert bv.validate_bind_manifest(manifest, {"abi_key": "A", "communicator": "serial"}) == []


def test_unknown_communicator_on_the_runtime_is_skipped():
    manifest = _Manifest(abi_key="A", communicator="serial")
    assert bv.validate_bind_manifest(manifest, {"abi_key": "A", "communicator": "unknown"}) == []


def test_serial_artifact_binds_under_a_parallel_runtime():
    # Directional: a serial artifact needs no communicator the runtime could lack.
    manifest = _Manifest(abi_key="A", communicator="serial")
    facts = {"abi_key": "A", "communicator": "mpi_comm_world"}
    assert bv.validate_bind_manifest(manifest, facts) == []


def test_parallel_artifact_on_a_serial_runtime_is_refused():
    # The only refusable direction: the artifact DECLARES a communicator the runtime lacks.
    manifest = _Manifest(abi_key="A", communicator="mpi_comm_world")
    lines = bv.validate_bind_manifest(manifest, {"abi_key": "A", "communicator": "serial"})
    assert any("communicator mismatch" in l and "requires" in l for l in lines)


def test_unknown_precision_is_skipped():
    manifest = _Manifest(abi_key="A", precision="unknown")
    assert bv.validate_bind_manifest(manifest, {"abi_key": "A", "precision": "double"}) == []


# ---------------------------------------------------------------------------
# Gate a -- aux required by a lowered operator
# ---------------------------------------------------------------------------

def test_operator_required_aux_unions_manifest_and_operators():
    manifest = _Manifest(aux_required=["B_z"], operators=[{"aux": ["kappa"]}])
    assert bv.operator_required_aux(manifest) == ["B_z", "kappa"]


def test_missing_operator_aux_is_refused():
    manifest = _Manifest(aux_required=["B_z"])
    lines = bv.validate_operator_aux(manifest, aux={})
    assert any("aux field 'B_z'" in l and "lowered operator" in l for l in lines)


def test_supplied_operator_aux_passes():
    manifest = _Manifest(aux_required=["B_z"])
    assert bv.validate_operator_aux(manifest, aux={"B_z": _Array((64, 64))}) == []
    # Also accepted when already declared on the sim (provided_named_aux).
    assert bv.validate_operator_aux(manifest, aux={}, provided_named_aux={"B_z"}) == []


def test_resolved_field_outputs_are_producers_not_bind_inputs():
    registration = SimpleNamespace(native_options={
        "output_route": {
            # Semantic output labels may be "potential" + "gradient"; bind authority is the
            # resolved scalar component route installed by the native field provider.
            "components": ("relaxation_potential", "relaxation_gx", "relaxation_gy"),
        },
    })
    artifact = SimpleNamespace(plan=SimpleNamespace(field_plans={"fields": registration}))

    produced = bv.field_produced_aux(artifact)
    assert produced == ("relaxation_gx", "relaxation_gy", "relaxation_potential")
    manifest = _Manifest(aux_required=[
        "relaxation_potential", "relaxation_gx", "relaxation_gy"])
    assert bv.validate_operator_aux(manifest, aux={}, provided_named_aux=produced) == []


def test_resolved_field_output_without_native_component_route_is_refused():
    registration = SimpleNamespace(native_options={"output_route": {}})
    with pytest.raises(TypeError, match="no exact native output components"):
        bv.field_plan_produced_aux({"fields": registration})


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_folds_lines_and_labels():
    msg = bv.aggregate_bind_refusals([("gate-a", ["l1"]), ("gate-b", ["l2", "l3"])])
    assert msg is not None
    assert "3 refusal(s)" in msg
    assert "[gate-a] l1" in msg and "[gate-b] l2" in msg and "[gate-b] l3" in msg


def test_aggregate_returns_none_when_all_pass():
    assert bv.aggregate_bind_refusals([("gate-a", []), ("gate-b", [])]) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
