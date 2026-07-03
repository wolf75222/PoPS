"""ADC-537: the pure bind-time refusal core (host-testable, no engine).

``pops.bind`` refuses a bad install with precise context BEFORE the native artifact is loaded. The
four gates are pure functions over plain metadata (a manifest / arguments stand-in, the mesh layout,
the declared runtime params, the supplied initial state), so they are exercised here with plain
Python objects -- no compiled ``.so``, no ``_pops``. The compiler-gated end-to-end refusal lives in
the integration tier; this tier proves the refusal LOGIC.
"""
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


class _Array:
    """A duck-typed array carrying only .shape / .dtype (no numpy dependency)."""

    def __init__(self, shape, dtype="float64"):
        self.shape = tuple(shape)
        self.dtype = type("D", (), {"name": dtype})()


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
    # The artifact REQUIRES MPI but the runtime does not provide it -> hard refusal.
    manifest = _Manifest(abi_key="A", supports_mpi=True)
    lines = bv.validate_bind_manifest(manifest, {"abi_key": "A", "supports_mpi": False})
    assert any("MPI support mismatch" in l for l in lines)


def test_gpu_required_but_runtime_lacks_it_is_refused():
    manifest = _Manifest(abi_key="A", supports_gpu=True)
    lines = bv.validate_bind_manifest(manifest, {"abi_key": "A", "supports_gpu": False})
    assert any("GPU / Kokkos" in l for l in lines)


def test_more_capable_runtime_is_not_a_mismatch():
    # A CPU-only artifact (supports_gpu/mpi False) binds fine on a Kokkos/MPI-capable runtime:
    # the runtime being MORE capable than the artifact needs is NOT a mismatch (directional gate).
    manifest = _Manifest(abi_key="A", supports_mpi=False, supports_gpu=False)
    facts = {"abi_key": "A", "supports_mpi": True, "supports_gpu": True}
    assert bv.validate_bind_manifest(manifest, facts) == []


def test_honest_unknown_runtime_token_is_skipped_not_a_fallback():
    # supports_mpi known on the manifest but UNKNOWN (None) on the runtime -> not adjudicable, skipped.
    manifest = _Manifest(abi_key="A", supports_mpi=True)
    assert bv.validate_bind_manifest(manifest, {"abi_key": "A", "supports_mpi": None}) == []


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
