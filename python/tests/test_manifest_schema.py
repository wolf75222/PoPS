"""STRICT versioned external-brick manifest schema (ADC-611).

The external-brick manifest (the JSON ``pops_brick_manifest()`` exports) is now small, strict and
versioned: it carries a top-level ``schema_version`` and refuses a missing/unknown field per a strict
policy, with an error that NAMES the offending field. This suite exercises the parser
(``pops.descriptors.parse_brick_manifest`` / ``_register_manifest`` / ``pops.external.read_manifest``)
against corrupt / unversioned / wrong-version / missing-field / unknown-field payloads, and the rich
``CompiledArtifactManifest`` round-trip (``from_dict(to_dict()) == m``).

Pure Python: the parser is host-only and _pops-free, so no compiled module is required. pops is never
faked -- the real parser and the real manifest classes are used.
"""
import json

import pytest

_desc = pytest.importorskip("pops.descriptors")
_ext = pytest.importorskip("pops.external.manifests")
VERSION = _desc.BRICK_MANIFEST_SCHEMA_VERSION


@pytest.fixture(autouse=True)
def _clean_catalog():
    _desc._clear_external_catalog()
    yield
    _desc._clear_external_catalog()


def _entry(**over):
    row = {"id": "my_brick", "category": "riemann", "requirements": "pressure", "capabilities": ""}
    row.update(over)
    return row


def _manifest(entries=None, **top):
    doc = {"schema_version": VERSION, "bricks": entries if entries is not None else [_entry()]}
    doc.update(top)
    return json.dumps(doc)


# ---- valid path -------------------------------------------------------------------------------

def test_valid_versioned_manifest_parses_and_registers():
    n = _desc._register_manifest(_manifest())
    assert n == 1
    records, abi_key = _desc.parse_brick_manifest(_manifest(abi_key="k=1"))
    assert records[0]["id"] == "my_brick"
    assert records[0]["requirements"] == ["pressure"]
    assert abi_key == "k=1"


# ---- corrupt / shape --------------------------------------------------------------------------

def test_corrupt_json_raises():
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest("not json {")
    assert "not valid JSON" in str(exc.value)


def test_non_object_toplevel_raises():
    with pytest.raises(ValueError):
        _desc.parse_brick_manifest("[1, 2, 3]")


# ---- schema_version ---------------------------------------------------------------------------

def test_missing_schema_version_is_refused_and_names_the_field():
    doc = json.dumps({"bricks": [_entry()]})  # NO schema_version -> legacy
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(doc)
    msg = str(exc.value)
    assert "schema_version" in msg
    assert "regenerate" in msg  # actionable: rebuild the brick library


def test_wrong_schema_version_names_got_and_expected():
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest(schema_version=999))
    msg = str(exc.value)
    assert "schema_version" in msg
    assert "999" in msg
    assert str(VERSION) in msg


# ---- unknown fields (strict, no permissive silent-ignore) -------------------------------------

def test_unknown_top_level_field_is_refused_and_named():
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest(surprise=1))
    msg = str(exc.value)
    assert "surprise" in msg
    assert "unknown top-level" in msg


def test_unknown_entry_field_is_refused_and_named():
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest([_entry(extra="x")]))
    msg = str(exc.value)
    assert "extra" in msg
    assert "unknown field" in msg


# ---- missing required fields (top + entry), naming the field ----------------------------------

def test_entry_missing_required_field_names_it():
    bad = {"id": "b", "category": "riemann", "requirements": "pressure"}  # no capabilities
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest([bad]))
    msg = str(exc.value)
    assert "capabilities" in msg
    assert "missing" in msg


def test_entry_missing_id_names_it():
    bad = {"category": "riemann", "requirements": "", "capabilities": ""}
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest([bad]))
    assert "id" in str(exc.value)


# ---- read_manifest (the .json inspection path uses the same strict parser) ---------------------

def test_read_manifest_json_path_is_strict(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"bricks": [_entry()]}))  # no schema_version
    with pytest.raises(ValueError) as exc:
        _ext.read_manifest(str(p))
    assert "schema_version" in str(exc.value)


def test_read_manifest_json_path_accepts_versioned(tmp_path):
    p = tmp_path / "good.json"
    p.write_text(_manifest())
    m = _ext.read_manifest(str(p))
    assert m.ids == ["my_brick"]


# ---- rich compiled-artifact manifest: schema_version + strict round-trip -----------------------

def _artifact():
    from pops.external.artifact_manifest import CompiledArtifactManifest
    return CompiledArtifactManifest(
        model_name="demo", abi_key="headers=abc|clang|20", blocks=["e", "i"],
        variables=["rho", "mx", "my"], roles=["Density", "MomentumX", "MomentumY"],
        params_runtime=["cs2"], supports_amr=True, supports_uniform=True)


def test_artifact_manifest_to_dict_carries_schema_version():
    from pops.external.artifact_manifest import ARTIFACT_MANIFEST_SCHEMA_VERSION
    d = _artifact().to_dict()
    assert d["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION


def test_artifact_manifest_round_trip_from_dict():
    from pops.external.artifact_manifest import CompiledArtifactManifest
    m = _artifact()
    back = CompiledArtifactManifest.from_dict(m.to_dict())
    # Round-trip: the re-serialised dict is identical (capability_matrix is derived and recomputed).
    assert back.to_dict() == m.to_dict()


def test_artifact_manifest_from_dict_missing_version_refused():
    from pops.external.artifact_manifest import CompiledArtifactManifest
    d = _artifact().to_dict()
    d.pop("schema_version")
    with pytest.raises(ValueError) as exc:
        CompiledArtifactManifest.from_dict(d)
    assert "schema_version" in str(exc.value)


def test_artifact_manifest_from_dict_unknown_field_refused():
    from pops.external.artifact_manifest import CompiledArtifactManifest
    d = _artifact().to_dict()
    d["totally_new"] = 1
    with pytest.raises(ValueError) as exc:
        CompiledArtifactManifest.from_dict(d)
    assert "totally_new" in str(exc.value)


# ---- external bricks appear in the capability report (ADC-611) --------------------------------

def test_registered_external_brick_appears_in_capabilities():
    from pops import inspect_capabilities
    _desc._register_manifest(_manifest([_entry(id="ext_hllc", category="riemann")]))
    matrix = inspect_capabilities()
    externals = [row for row in matrix.entries if row.source == "external"]
    assert any(row.name == "ext_hllc" for row in externals)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
