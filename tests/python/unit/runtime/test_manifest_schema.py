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
    row = {
        "id": "my_brick", "category": "riemann", "requirements": "pressure", "capabilities": "",
        "native_id": "my_brick", "supported_layouts": "", "supported_platforms": "",
        "params": "", "options": "", "exported_symbols": "",
    }
    row.update(over)
    return row


def _manifest(entries=None, **top):
    doc = {
        "schema_version": VERSION, "abi_key": "test-abi", "annotations": {},
        "bricks": entries if entries is not None else [_entry()],
    }
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
    assert "migrate" in msg


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


@pytest.mark.parametrize("field", ["id", "category", "native_id"])
@pytest.mark.parametrize("value", [None, "", 7, True])
def test_identity_fields_are_nonempty_strings(field, value):
    with pytest.raises(ValueError, match=field):
        _desc.parse_brick_manifest(_manifest([_entry(**{field: value})]))


@pytest.mark.parametrize(
    "field",
    ["requirements", "capabilities", "supported_layouts", "supported_platforms", "params",
     "options", "exported_symbols"],
)
def test_list_fields_are_explicit_canonical_csv(field):
    with pytest.raises(ValueError, match="CSV string"):
        _desc.parse_brick_manifest(_manifest([_entry(**{field: []})]))
    with pytest.raises(ValueError, match="canonical"):
        _desc.parse_brick_manifest(_manifest([_entry(**{field: "a, a"})]))
    with pytest.raises(ValueError, match="duplicate token"):
        _desc.parse_brick_manifest(_manifest([_entry(**{field: "a,a"})]))


# ---- read_manifest (the .json inspection path uses the same strict parser) ---------------------

def test_read_manifest_json_path_is_strict(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"bricks": [_entry()]}))  # no schema_version
    with pytest.raises(ValueError) as exc:
        _ext.read_manifest(str(p))
    assert "schema_version" in str(exc.value)


def test_read_manifest_json_path_accepts_versioned(tmp_path):
    from pops.external.manifests import CompiledManifest

    p = tmp_path / "good.json"
    wire = json.loads(_manifest(annotations={"x-owner": {"team": "runtime"}}))
    p.write_text(json.dumps(wire))
    m = _ext.read_manifest(str(p))
    assert m.ids == ["my_brick"]
    assert m.to_dict() == wire
    assert CompiledManifest.from_dict(m.to_dict()) == m


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
    assert d["protocol"] == "pops.manifest"
    assert d["kind"] == "compiled-artifact"


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
    with pytest.raises(TypeError) as exc:
        CompiledArtifactManifest.from_dict(d)
    assert "schema_version" in str(exc.value)


def test_artifact_manifest_from_dict_unknown_field_refused():
    from pops.external.artifact_manifest import CompiledArtifactManifest
    d = _artifact().to_dict()
    d["totally_new"] = 1
    with pytest.raises(TypeError) as exc:
        CompiledArtifactManifest.from_dict(d)
    assert "totally_new" in str(exc.value)


def test_artifact_manifest_external_bricks_is_additive_and_round_trips():
    # ADC-544: external_bricks is an additive field -> present as [] by default and round-trips.
    from pops.external.artifact_manifest import CompiledArtifactManifest
    m = _artifact()
    assert m.to_dict()["payload"]["external_bricks"] == []
    brick = {"id": "my_ext", "native_id": "ext_native", "category": "riemann",
             "requirements": ["pressure"], "capabilities": [], "supported_layouts": ["uniform"],
             "supported_platforms": ["cpu"], "params": [], "options": [],
             "exported_symbols": ["pops_brick_residual"]}
    m2 = CompiledArtifactManifest(model_name="demo", abi_key="h|c|20", external_bricks=[brick])
    d = m2.to_dict()
    assert d["payload"]["external_bricks"][0]["native_id"] == "ext_native"
    back = CompiledArtifactManifest.from_dict(d)
    assert back.to_dict() == d


def test_artifact_manifest_is_deeply_immutable_and_returns_detached_wire_data():
    from pops.external.artifact_manifest import CompiledArtifactManifest

    source = [{
        "id": "nested", "category": "riemann",
        "requirements": {"fields": ["pressure"]},
        "options": {"reconstruction": {"order": 2}},
    }]
    manifest = CompiledArtifactManifest(
        model_name="demo",
        ghost_depth_by_block={"fluid": 2},
        external_bricks=source,
    )
    source[0]["options"]["reconstruction"]["order"] = 7
    assert manifest.external_bricks[0]["options"]["reconstruction"]["order"] == 2

    with pytest.raises(AttributeError, match="immutable"):
        manifest.model_name = "forged"
    with pytest.raises(TypeError):
        manifest.ghost_depth_by_block["fluid"] = 9
    with pytest.raises(TypeError):
        manifest.external_bricks[0]["options"]["reconstruction"]["order"] = 9

    detached = manifest.to_dict()
    detached["payload"]["external_bricks"][0]["options"]["reconstruction"]["order"] = 11
    detached["payload"]["ghost_depth_by_block"]["fluid"] = 11
    assert manifest.external_bricks[0]["options"]["reconstruction"]["order"] == 2
    assert manifest.ghost_depth_by_block["fluid"] == 2


# ---- v3 explicit fields (native_id / layouts / platforms / params / options / symbols) -----------

def test_v2_optional_fields_parse():
    entry = _entry(id="ext_full", native_id="ext_native", supported_layouts="uniform,amr",
                   supported_platforms="cpu,mpi", params="cs2", options="reconstruct",
                   exported_symbols="pops_brick_residual")
    records, _ = _desc.parse_brick_manifest(_manifest([entry]))
    rec = records[0]
    assert rec["native_id"] == "ext_native"
    assert rec["supported_layouts"] == ["uniform", "amr"]
    assert rec["supported_platforms"] == ["cpu", "mpi"]
    assert rec["params"] == ["cs2"]
    assert rec["options"] == ["reconstruct"]
    assert rec["exported_symbols"] == ["pops_brick_residual"]


@pytest.mark.parametrize(
    "field",
    ["native_id", "supported_layouts", "supported_platforms", "params", "options",
     "exported_symbols"],
)
def test_v3_current_fields_are_explicit(field):
    entry = _entry()
    entry.pop(field)
    with pytest.raises(ValueError, match=field):
        _desc.parse_brick_manifest(_manifest([entry]))


def test_v3_still_refuses_an_unknown_entry_field():
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(_manifest([_entry(surprise_v2="x")]))
    assert "surprise_v2" in str(exc.value) and "unknown field" in str(exc.value)


def test_v1_manifest_is_refused_after_the_bump():
    # The ADC-544 bump makes a version-1 manifest incompatible (refuse-never-warn on the wire format).
    doc = json.dumps({"schema_version": 1, "abi_key": "legacy", "annotations": {},
                      "bricks": [_entry()]})
    with pytest.raises(ValueError) as exc:
        _desc.parse_brick_manifest(doc)
    msg = str(exc.value)
    assert "schema_version" in msg and "1" in msg and str(VERSION) in msg


def test_annotations_are_required_validated_and_preserved_exactly():
    from pops.external.manifests import CompiledManifest, _parse_manifest_metadata

    annotations = {"x-owner": {"team": "physics"}, "urn:pops:test": [1, "two"]}
    compiled = _parse_manifest_metadata(_manifest(annotations=annotations))
    assert compiled.annotations == annotations
    assert CompiledManifest.from_dict(compiled.to_dict()).to_dict() == compiled.to_dict()
    with pytest.raises(ValueError, match="annotations"):
        _desc.parse_brick_manifest(_manifest(annotations=None))
    with pytest.raises(ValueError, match="namespace URI"):
        _desc.parse_brick_manifest(_manifest(annotations={"owner": "physics"}))


def test_duplicate_json_key_and_duplicate_ids_are_refused():
    duplicate_key = (
        '{"schema_version":3,"abi_key":"a","abi_key":"b","annotations":{},"bricks":[]}'
    )
    with pytest.raises(ValueError, match="duplicate.*key"):
        _desc.parse_brick_manifest(duplicate_key)
    with pytest.raises(ValueError, match="duplicate brick id"):
        _desc.parse_brick_manifest(_manifest([_entry(), _entry()]))


def test_registration_is_idempotent_but_refuses_different_metadata_collision():
    payload = _manifest()
    assert _desc._register_manifest(payload) == 1
    assert _desc._register_manifest(payload) == 1
    with pytest.raises(ValueError, match="collision.*different metadata"):
        _desc._register_manifest(_manifest([_entry(category="preconditioner")]))
    with pytest.raises(ValueError, match="collision.*different metadata"):
        _desc._register_manifest(_manifest(annotations={"x-owner": "other"}))


# ---- external bricks appear in the capability report (ADC-611) --------------------------------

def test_registered_external_brick_appears_in_capabilities():
    from pops._capabilities_inspect import _descriptor_catalog_report
    _desc._register_manifest(_manifest([_entry(id="ext_hllc", category="riemann")]))
    matrix = _descriptor_catalog_report()
    externals = [row for row in matrix.entries if row.source == "external"]
    assert any(row.name == "ext_hllc" for row in externals)


def test_external_brick_row_carries_v2_route_fields(tmp_path):
    # ADC-544: a resolved brick's supported layouts/platforms appear on its capability row so a reader
    # sees the declared route surface, not just that the brick exists.
    from pops._capabilities_inspect import _descriptor_catalog_report
    _desc._register_manifest(_manifest([_entry(
        id="ext_route", category="riemann", native_id="route_native",
        supported_layouts="uniform,amr", supported_platforms="cpu,mpi")]))
    matrix = _descriptor_catalog_report()
    row = next(r for r in matrix.entries if r.source == "external" and r.name == "ext_route")
    assert row.native_id == "route_native"
    assert row.layout == "amr|uniform"  # sorted join of the declared layouts
    assert row.platform == "cpu|mpi"
    assert row.mpi is True
    assert "layouts=amr,uniform" in row.limitation and "platforms=cpu,mpi" in row.limitation


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
