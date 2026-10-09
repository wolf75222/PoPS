"""Synthetic backing protocol bytes only; no native artifact or ROOT approval."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("m19_archived_test", Path(__file__).with_name("sol61_m19_saved_reception.py"))
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)


def fixture(tmp_path, originals=None, extras=()):
    originals = originals or {"/historical/installation/native.so": b"synthetic native protocol bytes", "/historical/case/state.npz": b"synthetic NOT native NPZ"}
    archive = tmp_path / "snapshot.zip"
    contents = {"sha256/" + r.digest(raw): raw for raw in originals.values()}
    with zipfile.ZipFile(archive, "w") as z:
        for member, raw in contents.items():
            z.writestr(member, raw)
        for member, raw in extras:
            z.writestr(member, raw)
    pins = {"schema": "sol61.m19-owner-pins@2", "synthetic_protocol_leaves": [{"path": path, "sha256": r.digest(raw)} for path, raw in originals.items()], "backing": {"archive": {"path": str(archive), "sha256": r.digest(archive.read_bytes())}, "members": {path: {"member": "sha256/" + r.digest(raw), "sha256": r.digest(raw)} for path, raw in originals.items()}}}
    return pins, originals


def test_archive_only_preserves_original_names_and_never_queries_live_origins(tmp_path, monkeypatch):
    pins, originals = fixture(tmp_path)
    with r.archived_context(pins):
        def forbidden(*args, **kwargs):
            raise AssertionError("queried live historical donor")
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        monkeypatch.setattr(Path, "is_symlink", forbidden)
        monkeypatch.setattr(Path, "resolve", forbidden)
        for path, raw in originals.items():
            assert r.pinned({"path": path, "sha256": r.digest(raw)}, "/historical")[1] == raw
            assert r.leaf(path)["path"] == path
    assert r._BACKING.get() is None


def test_equal_content_different_original_paths_is_valid(tmp_path):
    pins, originals = fixture(tmp_path, {"/old/a": b"shared", "/old/b": b"shared"})
    with r.archived_context(pins):
        assert [r.read_bytes(path) for path in originals] == [b"shared", b"shared"]


@pytest.mark.parametrize("attack,diagnostic", (("missing", "closed origin inventory"), ("extra", "closed origin inventory"), ("origin_alias", "canonical spelling"), ("member_alias", "identity differs"), ("digest", "identity differs"), ("member_absent", "member absent"), ("archive_digest", "archive digest differs"), ("conflicting_origin", "conflicting original path digests")))
def test_resealed_mapping_countermodels(tmp_path, attack, diagnostic):
    pins, originals = fixture(tmp_path)
    path = next(iter(originals))
    rows = pins["backing"]["members"]
    if attack == "missing":
        del rows[path]
    elif attack == "extra":
        rows["/historical/foreign"] = copy.deepcopy(rows[path])
    elif attack == "origin_alias":
        pins["synthetic_protocol_leaves"][0]["path"] = "/historical/../escaped"
    elif attack == "member_alias":
        rows[path]["member"] = "sha256/../escaped"
    elif attack == "digest":
        rows[path]["sha256"] = "0" * 64
    elif attack == "member_absent":
        sha = r.digest(b"absent")
        rows[path] = dict(member="sha256/" + sha, sha256=sha)
        pins["synthetic_protocol_leaves"][0]["sha256"] = sha
    elif attack == "conflicting_origin":
        pins["synthetic_protocol_leaves"].append(dict(path=path, sha256="0" * 64))
    else:
        pins["backing"]["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=diagnostic):
        with r.archived_context(pins):
            pytest.fail("entered corrupted backing")
    assert r._BACKING.get() is None


@pytest.mark.parametrize("name", ("../escape", "/absolute", "a/../escape", "a//alias", "a\\escape", "a/"))
def test_unsafe_members_rejected_even_if_not_mapped(tmp_path, name):
    pins, _ = fixture(tmp_path, extras=((name, b"unmapped"),))
    with pytest.raises(ValueError, match="aliases or escapes"):
        with r.archived_context(pins):
            pass


def test_duplicate_and_symlink_archive_members(tmp_path):
    pins, _ = fixture(tmp_path)
    archive = Path(pins["backing"]["archive"]["path"])
    with zipfile.ZipFile(archive, "a") as z:
        with pytest.warns(UserWarning):
            z.writestr(next(iter(pins["backing"]["members"].values()))["member"], b"duplicate")
    pins["backing"]["archive"]["sha256"] = r.digest(archive.read_bytes())
    with pytest.raises(ValueError, match="duplicate archive member"):
        r.ArchivedBacking(pins)
    pins, _ = fixture(tmp_path)
    with zipfile.ZipFile(archive, "a") as z:
        info = zipfile.ZipInfo("alias")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "target")
    pins["backing"]["archive"]["sha256"] = r.digest(archive.read_bytes())
    with pytest.raises(ValueError, match="aliases or escapes"):
        r.ArchivedBacking(pins)


def test_resealed_archive_corrupted_scientific_member_rejected(tmp_path):
    pins, _ = fixture(tmp_path)
    archive = Path(pins["backing"]["archive"]["path"])
    with zipfile.ZipFile(archive) as z:
        contents = {info.filename: z.read(info) for info in z.infolist()}
    target = pins["backing"]["members"]["/historical/case/state.npz"]["member"]
    contents[target] = b"corrupted synthetic state"
    with zipfile.ZipFile(archive, "w") as z:
        for member, raw in contents.items():
            z.writestr(member, raw)
    pins["backing"]["archive"]["sha256"] = r.digest(archive.read_bytes())
    with pytest.raises(ValueError, match="member digest differs"):
        r.ArchivedBacking(pins)


@pytest.mark.parametrize("alias", ("direct", "parent", "dangling"))
def test_physical_backing_alias_refused_before_resolution(tmp_path, alias):
    pins, _ = fixture(tmp_path)
    archive = Path(pins["backing"]["archive"]["path"])
    if alias == "parent":
        link = tmp_path / "parent-alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / archive.name
    else:
        path = tmp_path / "archive-alias.zip"
        path.symlink_to(archive if alias == "direct" else tmp_path / "missing.zip")
    pins["backing"]["archive"]["path"] = str(path)
    with pytest.raises(ValueError, match="aliases"):
        r.ArchivedBacking(pins)


def test_context_exception_resets_and_unknown_origin_never_falls_back(tmp_path):
    live = tmp_path / "live-file"
    live.write_bytes(b"live foreign bytes")
    pins, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="absent from closed backing map"):
        with r.archived_context(pins):
            r.read_bytes(live)
    assert r._BACKING.get() is None
    assert r.read_bytes(live) == b"live foreign bytes"


@pytest.mark.parametrize("extra", (False, True))
def test_mapped_phase_directory_has_exactly_thirteen_leaves(tmp_path, extra):
    # Protocol-only bytes, deliberately not valid native state/checkpoint files.
    directory = Path("/historical/closed-case")
    originals = {str(directory / "provenance.json"): b"synthetic provenance"}
    for phase in r.PHASES:
        state, checkpoint = phase.encode(), (phase + " synthetic checkpoint").encode()
        originals[str(directory / (phase + "-state.npz"))] = state
        originals[str(directory / (phase + ".checkpoint.npz"))] = checkpoint
        originals[str(directory / (phase + "-receipt.json"))] = json.dumps(dict(
            checkpoint=phase + ".checkpoint.npz", saved_state_sha256=r.digest(state),
            checkpoint_sha256=r.digest(checkpoint))).encode()
    if extra:
        originals[str(directory / "undeclared-state.npz")] = b"extra protocol bytes"
    pins, _ = fixture(tmp_path, originals)
    with r.archived_context(pins):
        if extra:
            with pytest.raises(ValueError, match="closed 13-file inventory differs"):
                r.closed_phases(directory)
        else:
            rows = r.closed_phases(directory)
            assert set(rows) == {"provenance", *r.PHASES}


def test_v2_requires_matching_external_approval_before_opening_archive(tmp_path, monkeypatch):
    pins, _ = fixture(tmp_path)
    p, approval = tmp_path / "pins.json", tmp_path / "unapproved.json"
    p.write_text(json.dumps(pins))
    approval.write_text(json.dumps(dict(schema="sol61.m19-root-approval@1", approved_by="UNAPPROVED_SYNTHETIC", pins_sha256=r.digest(p.read_bytes()))))
    def forbidden(*args):
        raise AssertionError("opened archive before external approval")
    monkeypatch.setattr(r, "ArchivedBacking", forbidden)
    with pytest.raises(ValueError, match="ROOT has not approved"):
        r.receive(p, r.digest(p.read_bytes()), approval, r.digest(approval.read_bytes()))
    with pytest.raises(ValueError, match="external seal digest differs"):
        r.receive(p, "0" * 64, approval, r.digest(approval.read_bytes()))
