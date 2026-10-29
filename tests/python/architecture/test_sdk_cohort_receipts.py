"""ROM-17: source, installed SDK and cohort identities, installation leases and receipts.

SOURCE-ONLY tests (no ``pops`` / ``_pops`` import).  They build a throwaway Git checkout and a fake
installed SDK beside it (pip ``RECORD``, ``direct_url.json``, installed headers, ``variants.json``)
and exercise ``scripts/sdk_cohort.py`` and ``scripts/qualification_receipt.py`` on them.  The real
compiled, loaded and advanced receipt is received by
``tests/python/integration/runtime/test_qualification_receipt_native.py``.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]


def _load(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cohort = _load("sdk_cohort")
receipts = _load("qualification_receipt")


# --------------------------------------------------------------------------- #
# A checkout and an installed SDK built from it                                #
# --------------------------------------------------------------------------- #
GIT = ("git", "-c", "user.name=PoPS test", "-c", "user.email=test@example.invalid",
       "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main")
SOURCES = {
    "python/pops/__init__.py": "VERSION = 1\n",
    "python/pops/model.py": "def rate(u):\n    return -u\n",
    "python/pops/py.typed": "",
    "include/pops_headers.manifest": "api pops/core.hpp\ntest-only pops/testing.hpp\n",
    "include/pops/core.hpp": "#pragma once\nint pops_core();\n",
    "include/pops/testing.hpp": "#pragma once\n",
    "src/native.cpp": "int pops_core() { return 1; }\n",
    "examples/case.py": "CASE = 'M01'\n",
    "tests/data.txt": "fixture\n",
}


def _git(repo, *arguments):
    return subprocess.run([*GIT, "-C", str(repo), *arguments], check=True,
                          capture_output=True).stdout.decode()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    for relative, content in SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run([*GIT, "init", "-q", str(root)], check=True)
    _commit(root, "source")
    return root


def _record_hash(path):
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def install(repo, site, *, dimension=2, leaf=b"native leaf", abi_headers=None):
    """Install the checkout's package, headers and one native leaf as pip would."""
    package = site / "pops"
    for relative in SOURCES:
        if relative.startswith("python/pops/"):
            target = package / relative[len("python/pops/"):]
        elif relative.startswith("include/") and not relative.endswith("testing.hpp"):
            target = package / relative
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    signature = cohort.header_signature(package / "include")
    extension = package / "_native" / f"dim{dimension}" / f"_pops{SUFFIX}"
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(leaf)
    row = {"dimension": dimension, "path": f"dim{dimension}/{extension.name}",
           "sha256": hashlib.sha256(leaf).hexdigest(), "version": "1.0.0",
           "abi_key": "compiler=13;headers=%s;dim=%d" % (abi_headers or signature, dimension),
           "has_mpi": False, "has_kokkos": True}
    (package / "_native" / "variants.json").write_text(
        json.dumps({"schema_version": 1, "variants": [row]}), encoding="utf-8")
    info = site / "pops-1.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Name: pops\nVersion: 1.0.0\n", encoding="utf-8")
    wheel = site.parent / "wheelhouse" / "pops-1.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"retained wheel " + leaf)
    (info / "direct_url.json").write_text(json.dumps({
        "url": wheel.as_uri(),
        "archive_info": {"hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
    }), encoding="utf-8")
    rows = [f"pops/{path.relative_to(package).as_posix()},{_record_hash(path)},1"
            for path in sorted(package.rglob("*")) if path.is_file()]
    rows += [f"{info.name}/METADATA,,", f"{info.name}/RECORD,,"]
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return package


@pytest.fixture
def site(tmp_path, repo):
    root = tmp_path / "site-packages"
    install(repo, root)
    return root


def _attest(repo, site, **options):
    return cohort.attest_native_build(repo, site / "pops", 2, **options)


# --------------------------------------------------------------------------- #
# Source identity                                                              #
# --------------------------------------------------------------------------- #
def test_source_identity_names_one_clean_commit_and_flags_every_modification(repo):
    clean = cohort.source_identity(repo)
    assert clean["clean"] and clean["sdk_inputs_clean"]
    assert clean["commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert clean["tree"] == _git(repo, "rev-parse", "HEAD^{tree}").strip()

    (repo / "docs-note.txt").write_text("untracked, outside the SDK inputs\n")
    assert cohort.source_identity(repo)["clean"]

    (repo / "tests/data.txt").write_text("edited\n")
    edited = cohort.source_identity(repo)
    assert not edited["clean"] and edited["sdk_inputs_clean"]
    assert edited["modified"] == ["tests/data.txt"]
    with pytest.raises(cohort.CohortError, match="not clean"):
        cohort.require_clean_source(edited)

    (repo / "python/pops/extra.py").write_text("X = 1\n")
    planted = cohort.source_identity(repo)
    assert not planted["sdk_inputs_clean"]
    assert planted["untracked_sdk_inputs"] == ["python/pops/extra.py"]


def test_sdk_inputs_digest_ignores_tests_and_examples(repo):
    before = cohort.source_identity(repo)["sdk_inputs_sha256"]
    (repo / "examples/case.py").write_text("CASE = 'M02'\n")
    _commit(repo, "case only")
    assert cohort.source_identity(repo)["sdk_inputs_sha256"] == before
    (repo / "src/native.cpp").write_text("int pops_core() { return 2; }\n")
    _commit(repo, "native change")
    assert cohort.source_identity(repo)["sdk_inputs_sha256"] != before


def test_renamed_tracked_file_is_reported_once(repo):
    _git(repo, "mv", "tests/data.txt", "tests/moved.txt")
    source = cohort.source_identity(repo)
    assert source["modified"] == ["tests/moved.txt"]
    assert not source["clean"]


# --------------------------------------------------------------------------- #
# Installed SDK identity and coherence                                         #
# --------------------------------------------------------------------------- #
def test_sdk_identity_is_the_exact_installed_bytes_authenticated_by_record(site):
    sdk = cohort.sdk_identity(site / "pops")
    assert sdk["installation"]["kind"] == "wheel" and sdk["installation"]["record_verified"]
    assert sdk["installation_root"] == str(site.resolve())
    assert [row["dimension"] for row in sdk["native"]] == [2]

    cache = site / "pops" / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-312.pyc").write_bytes(b"bytecode")
    assert cohort.sdk_identity(site / "pops")["sdk_id"] == sdk["sdk_id"]

    (site / "pops" / "model.py").write_text("def rate(u):\n    return u\n")
    changed = cohort.sdk_identity(site / "pops")
    assert changed["sdk_id"] != sdk["sdk_id"]
    assert changed["installation"]["kind"] == "unmanaged"
    assert changed["installation"]["mismatched"] == ["model.py"]


def test_native_leaf_must_match_its_variant_manifest(site):
    leaf = next((site / "pops" / "_native" / "dim2").iterdir())
    leaf.write_bytes(b"swapped leaf")
    with pytest.raises(cohort.CohortError, match="disagree with variants.json"):
        cohort.sdk_identity(site / "pops")


def test_coherence_compares_sources_headers_and_the_native_header_token(repo, site, tmp_path):
    check = cohort.coherence(repo, site / "pops")
    assert check["coherent"], check
    assert check["compared_files"] == 5  # 3 Python sources, 1 installed header, the manifest

    (site / "pops" / "include" / "pops" / "core.hpp").write_text("#pragma once\n")
    edited = cohort.coherence(repo, site / "pops")
    assert not edited["coherent"] and edited["differing"] == ["include/pops/core.hpp"]

    other = tmp_path / "other-site"
    install(repo, other, abi_headers="0" * 64)
    stale = cohort.coherence(repo, other / "pops")
    assert not stale["coherent"]
    assert stale["native_headers"][0]["matches_installed_headers"] is False

    (other / "pops" / "stray.py").write_text("")
    assert cohort.coherence(repo, other / "pops")["unexpected"] == ["stray.py"]


# --------------------------------------------------------------------------- #
# Native build attestation and the immutable cohort record                     #
# --------------------------------------------------------------------------- #
def test_cohort_needs_every_native_leaf_attested_by_a_build_of_the_same_inputs(repo, site):
    with pytest.raises(cohort.CohortError, match="no official build attests it"):
        cohort.publish_cohort(repo, site / "pops")

    attestation = _attest(repo, site, command=["bash", "scripts/build_python.sh", "--dim", "2"])
    assert attestation["source"]["sdk_inputs_clean"]
    record = cohort.publish_cohort(repo, site / "pops")
    path = cohort.cohort_path(site)
    assert path.stat().st_mode & 0o777 == 0o444
    assert record["cohort_id"] == cohort.cohort_id(record["source"], record["sdk"])
    assert record["native_provenance"]["leaves"][0]["built_from_commit"] \
        == attestation["source"]["commit"]
    assert cohort.read_cohort(site) == record
    assert cohort.verify_cohort(site / "pops", source_root=repo)["verified"]
    assert cohort.publish_cohort(repo, site / "pops") == record  # idempotent


def test_a_published_cohort_never_changes_and_tampering_is_detected(repo, site):
    _attest(repo, site)
    record = cohort.publish_cohort(repo, site / "pops")

    (repo / "examples/case.py").write_text("CASE = 'M03'\n")
    _commit(repo, "a later source with the same SDK inputs")
    with pytest.raises(cohort.CohortError, match="never changes"):
        cohort.publish_cohort(repo, site / "pops")
    moved = cohort.verify_cohort(site / "pops", source_root=repo)
    assert not moved["verified"] and "cohort source is" in moved["problems"][0]

    (site / "pops" / "model.py").write_text("def rate(u):\n    return 0\n")
    tampered = cohort.verify_cohort(site / "pops")
    assert not tampered["verified"]
    assert any("installed bytes changed" in item for item in tampered["problems"])

    forged = dict(record, source=dict(record["source"], commit="f" * 40))
    with pytest.raises(cohort.CohortError, match="does not authenticate"):
        cohort.validate_cohort(forged)


def test_a_leaf_restored_from_other_sdk_inputs_cannot_join_the_cohort(repo, site):
    _attest(repo, site)
    (repo / "src/native.cpp").write_text("int pops_core() { return 3; }\n")
    _commit(repo, "native sources changed, leaf not rebuilt")
    with pytest.raises(cohort.CohortError, match="built from other SDK inputs"):
        cohort.publish_cohort(repo, site / "pops")


def test_a_build_from_modified_sdk_inputs_attests_nothing(repo, site):
    (repo / "src/native.cpp").write_text("int pops_core() { return 4; }\n")
    dirty = _attest(repo, site)
    assert not dirty["source"]["sdk_inputs_clean"]
    _git(repo, "checkout", "--", "src/native.cpp")
    with pytest.raises(cohort.CohortError, match="built from modified SDK inputs"):
        cohort.publish_cohort(repo, site / "pops")


def test_a_cohort_needs_a_clean_source_and_a_wheel_installation(repo, site):
    _attest(repo, site)
    (repo / "tests/data.txt").write_text("edited\n")
    with pytest.raises(cohort.CohortError, match="not clean"):
        cohort.publish_cohort(repo, site / "pops")
    _git(repo, "checkout", "--", "tests/data.txt")
    (site / "pops-1.0.0.dist-info" / "direct_url.json").write_text(
        json.dumps({"url": repo.as_uri(), "dir_info": {}}))
    with pytest.raises(cohort.CohortError, match="needs a wheel installation"):
        cohort.publish_cohort(repo, site / "pops")


# --------------------------------------------------------------------------- #
# Installation leases                                                          #
# --------------------------------------------------------------------------- #
def test_qualification_lease_is_shared_by_one_source_and_refuses_another(site):
    first = cohort.acquire_qualification_lease(site, "a" * 40, purpose="worker 1")
    second = cohort.acquire_qualification_lease(site, "a" * 40, purpose="worker 2")
    with pytest.raises(cohort.InstallationBusy, match="needs its own installation"):
        cohort.acquire_qualification_lease(site, "b" * 40, purpose="worker 3")
    with pytest.raises(cohort.InstallationBusy, match="in use"):
        cohort.acquire_mutation_lease(site, purpose="pip install")
    first.release()
    with pytest.raises(cohort.InstallationBusy):
        cohort.acquire_qualification_lease(site, "b" * 40)
    second.release()
    other = cohort.acquire_qualification_lease(site, "b" * 40)
    other.release()


def test_mutation_lease_excludes_every_qualification(site):
    with cohort.mutation_lease(site, purpose="build_python.sh --dim 2"):
        with pytest.raises(cohort.InstallationBusy, match="being modified"):
            cohort.acquire_qualification_lease(site, "a" * 40)
        with pytest.raises(cohort.InstallationBusy):
            cohort.acquire_mutation_lease(site)
    cohort.acquire_qualification_lease(site, "a" * 40).release()


def _cli(*arguments, env=None):
    return subprocess.run([sys.executable, "-I", str(SCRIPTS / "sdk_cohort.py"), *arguments],
                          capture_output=True, text=True, env=env, check=False)


def test_two_processes_cannot_qualify_two_sources_in_one_installation(repo, site, tmp_path):
    other = tmp_path / "other-checkout"
    subprocess.run([*GIT, "clone", "-q", str(repo), str(other)], check=True)
    (other / "examples/case.py").write_text("CASE = 'M04'\n")
    _commit(other, "another source")
    script = (
        "import subprocess, sys; "
        "r = subprocess.run([sys.executable, '-I', %r, 'hold', '--installation', %r, "
        "'--source', %r, '--', sys.executable, '-c', 'pass'], capture_output=True, text=True); "
        "print(r.returncode); print(r.stderr)"
        % (str(SCRIPTS / "sdk_cohort.py"), str(site), str(other)))
    held = _cli("hold", "--installation", str(site), "--source", str(repo), "--",
                sys.executable, "-c", script)
    assert held.returncode == 0, held.stderr
    code, message = held.stdout.split("\n", 1)
    assert code == "3" and "needs its own installation" in message

    same = ("import subprocess, sys; raise SystemExit(subprocess.run([sys.executable, '-I', %r, "
            "'hold', '--installation', %r, '--source', %r, '--', sys.executable, '-c', "
            "'pass']).returncode)" % (str(SCRIPTS / "sdk_cohort.py"), str(site), str(repo)))
    shared = _cli("hold", "--installation", str(site), "--source", str(repo), "--",
                  sys.executable, "-c", same)
    assert shared.returncode == 0, shared.stderr


def test_a_build_holds_the_installation_for_its_whole_duration(site):
    probe = ("import subprocess, sys; "
             "held = subprocess.run([sys.executable, '-I', %r, 'held', '--installation', %r]); "
             "busy = subprocess.run([sys.executable, '-I', %r, 'mutate', '--installation', %r, "
             "'--', sys.executable, '-c', 'pass'], capture_output=True); "
             "raise SystemExit(10 * held.returncode + busy.returncode)"
             % (str(SCRIPTS / "sdk_cohort.py"), str(site), str(SCRIPTS / "sdk_cohort.py"),
                str(site)))
    result = _cli("mutate", "--installation", str(site), "--purpose", "test build", "--",
                  sys.executable, "-c", probe)
    assert result.returncode == 3, result.stderr  # held (0) inside, a second build refused (3)
    alone = _cli("held", "--installation", str(site))
    assert alone.returncode == 1


def test_build_python_runs_under_the_lease_and_attests_retained_wheels():
    build = (ROOT / "scripts" / "build_python.sh").read_text(encoding="utf-8")
    activate = build.index('conda activate "$ENV_NAME"')
    lease = build.index('sdk_cohort.py" mutate --installation "$POPS_SITE_PACKAGES"')
    install_command = build.index('python -m pip "${pip_args[@]}"')
    verify = build.index('verify_installed_native.py" "${native_verify_args[@]}"')
    attest = build.index('sdk_cohort.py" attest-build --source "$HERE"')
    assert activate < build.index('sdk_cohort.py" held') < lease < install_command
    assert install_command < verify < attest
    assert 'ORIGINAL_ARGS=("$@")' in build
    assert '-- bash "$HERE/scripts/build_python.sh" "${ORIGINAL_ARGS[@]}"' in build
    assert '--wheel "${built_wheels[0]}"' in build[attest:]


# --------------------------------------------------------------------------- #
# Receipts                                                                     #
# --------------------------------------------------------------------------- #
def _evidence(stage):
    return {
        "E": {"files": [{"path": "examples/case.py", "sha256": "0" * 64, "committed": True}]},
        "G": {"verified": True, "semantic_identity": "plan"},
        "C": {"same_plan_as_G": True, "dsos": [
            {"role": "program", "path": "/p.so", "sha256": "1" * 64, "artifact_sidecar": {}},
            {"role": "model", "path": "/m.so", "sha256": "2" * 64}]},
        "loaded": {"all_mapped": True, "system": "System", "installed_program_matches": True},
        "advanced": {"same_artifact_as_C": True, "accepted_steps": 2},
        "S": {"passed": True, "name": "mass", "oracle": "exact", "domain": "M01"},
    }[stage]


def _receipt(source, sdk, *, stages=receipts.STAGES, overlay=None, failures=(), consensus=True,
             patch=None):
    data = {
        "schema": receipts.RECEIPT_SCHEMA, "receipt_id": None, "kind": "pops-runtime",
        "requirement": {"statement": "M01 compiled, loaded and advanced", "domain": "M01"},
        "status": "failed" if failures else "closed", "opened_utc": "t0", "closed_utc": "t1",
        "source": source, "sdk": sdk, "coherence": {"coherent": True}, "cohort": None,
        "lease": {"held": True}, "process": {"harness": []}, "backend": {"native_loaded": True},
        "ranks": {"size": 1, "rank": 0, "consensus": consensus},
        "parameters": {}, "stages": [{"stage": name, "at_utc": "t", "evidence": _evidence(name)}
                                     for name in stages],
        "dsos": [], "overlay": overlay or {"clean": True, "findings": []},
        "levels": {}, "level": "none", "limits": [], "failures": list(failures),
    }
    if patch is not None:
        patch(data)
    data["levels"], data["level"], data["limits"] = receipts.derive_levels(data)
    data["receipt_id"] = receipts.receipt_id(data)
    receipts.validate_receipt(data)
    return data


@pytest.fixture
def published(repo, site):
    _attest(repo, site)
    record = cohort.publish_cohort(repo, site / "pops")
    return record, cohort.source_identity(repo), cohort.sdk_identity(site / "pops")


def test_every_k6_level_reads_off_the_recorded_stages(published):
    _, source, sdk = published
    full = _receipt(source, sdk)
    assert full["levels"] == dict.fromkeys(receipts.LADDER, True) and full["level"] == "S"
    assert _receipt(source, sdk, stages=("E", "G", "C"))["level"] == "C"
    unloaded = _receipt(source, sdk, stages=("E", "G", "C", "advanced"))
    assert unloaded["level"] == "C" and any(item.startswith("R:") for item in unloaded["limits"])

    def other_program(data):
        data["stages"][3]["evidence"]["installed_program_matches"] = False
    assert _receipt(source, sdk, patch=other_program)["level"] == "C"

    def uncommitted(data):
        data["stages"][0]["evidence"]["files"][0]["committed"] = False
    edited = _receipt(source, sdk, patch=uncommitted)
    assert edited["level"] == "none" and edited["levels"]["S"] is False


def test_an_r_receipt_of_the_cohort_source_and_sdk_is_admitted(published):
    record, source, sdk = published
    verdict = receipts.admit(_receipt(source, sdk), record, require="R")
    assert verdict["admitted"], verdict
    assert verdict["cohort_id"] == record["cohort_id"]


def test_syntax_or_overlay_receipts_never_receive_the_cohort(published):
    record, source, sdk = published
    syntax = _receipt(source, sdk, stages=("E", "G"))
    assert not receipts.admit(syntax, record, require="R")["admitted"]
    overlay = _receipt(source, sdk, overlay={"clean": False, "findings": [
        "POPS_NATIVE_VARIANTS_ROOT selects native leaves outside the SDK: /tmp/x"]})
    refused = receipts.admit(overlay, record, require="E")
    assert not refused["admitted"]
    assert refused["reasons"] == [
        "overlay: POPS_NATIVE_VARIANTS_ROOT selects native leaves outside the SDK: /tmp/x"]
    disagreeing = _receipt(source, sdk, consensus=None)
    assert "ranks did not agree" in receipts.admit(disagreeing, record)["reasons"][0]
    failed = _receipt(source, sdk, failures=[{"stage": "advanced", "message": "boom"}])
    assert not receipts.admit(failed, record)["admitted"]


def test_a_receipt_of_another_source_or_sdk_is_refused(published):
    record, source, sdk = published
    other_source = dict(source, commit="c" * 40)
    reasons = receipts.admit(_receipt(other_source, sdk), record)["reasons"]
    assert reasons == ["receipt source %s is not the cohort source %s"
                       % ("c" * 40, source["commit"])]
    other_sdk = dict(sdk, sdk_id="d" * 64)
    assert "is not the cohort SDK" in receipts.admit(_receipt(source, other_sdk),
                                                      record)["reasons"][0]
    assert receipts.admit(_receipt(source, sdk), None)["reasons"] == [
        "no cohort record names this installation"]


def test_a_receipt_authenticates_its_identity_and_its_levels(published, tmp_path):
    _, source, sdk = published
    data = _receipt(source, sdk)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    assert receipts.read_receipt(path) == data
    inflated = copy.deepcopy(data)
    inflated["stages"] = inflated["stages"][:3]
    with pytest.raises(receipts.ReceiptError, match="do not follow from its stages"):
        receipts.validate_receipt(inflated)
    edited = copy.deepcopy(data)
    edited["parameters"] = {"t_end": 1.0}
    path.write_text(json.dumps(edited))
    with pytest.raises(receipts.ReceiptError, match="does not authenticate"):
        receipts.read_receipt(path)


def test_a_standalone_cpp_reception_is_finite_and_never_admitted(published, repo, tmp_path):
    record, _, _ = published
    data = receipts.record_standalone(
        "standalone M01 kernel", source_root=repo, binary=Path(sys.executable),
        command=[sys.executable, "-c", "pass"], directory=tmp_path)
    assert data["level"] == "R-finite"
    assert data["levels"] == dict.fromkeys(receipts.LADDER, False)
    assert data["limits"] == ["a standalone C++ program is C/R-finite, never R-PoPS (K6)"]
    verdict = receipts.admit(data, record, require="C")
    assert not verdict["admitted"]
    assert "never R-PoPS (K6); it qualifies no cohort" in verdict["reasons"][0]
    written = next(tmp_path.glob("standalone-m01-kernel-*.json"))
    assert receipts.read_receipt(written)["kind"] == "standalone-cpp"


class _Identity(SimpleNamespace):
    pass


def _fake_artifact(tmp_path, plan_identity):
    program = tmp_path / "program.so"
    program.write_bytes(b"program image")
    (tmp_path / "program.so.pops-artifact.json").write_text(json.dumps({"artifact": "a"}))
    model = tmp_path / "model.so"
    model.write_bytes(b"model image")
    return SimpleNamespace(
        verify=lambda: None, artifact_identity=_Identity(token="artifact"),
        semantic_identity=_Identity(token=plan_identity), target="system",
        backend="production", abi_key="abi",
        layout_program_paths={"layout": str(program)},
        blocks=(SimpleNamespace(name="fluid", model=SimpleNamespace(so_path=str(model))),))


def test_receipt_records_each_stage_and_never_counts_unmapped_dsos(repo, site, tmp_path):
    plan = SimpleNamespace(verify=lambda: None, plan_identity=_Identity(token="plan"),
                           snapshot=SimpleNamespace(semantic_identity=_Identity(token="sem")),
                           resolved_dimension=2, target="system", backend="production")
    report = SimpleNamespace(to_data=lambda: {"accepted_steps": 2, "rejected_steps": 0,
                                              "final_time": .1, "stop_reason": "t_end"},
                             artifact_identity=_Identity(token="artifact"))
    receipt = receipts.Receipt("fake case", source_root=repo, package=site / "pops",
                               directory=tmp_path / "receipts", domain="source-only")
    with receipt:
        with pytest.raises(receipts.ReceiptError, match="only after the stage C"):
            receipt.loaded(SimpleNamespace())
        receipt.equations(repo / "examples/case.py")
        receipt.plan(plan)
        receipt.compiled(_fake_artifact(tmp_path, "sem"))
        receipt.parameters(t_end=.1, cells=(16, 8))
        with pytest.raises(receipts.ReceiptError, match="only after the stage loaded"):
            receipt.advanced(report)
        receipt.loaded(SimpleNamespace(_executor=SimpleNamespace(_s=SimpleNamespace())))
        receipt.advanced(report)
    data = receipts.read_receipt(receipt.path)
    assert [row["stage"] for row in data["stages"]] == ["E", "G", "C", "loaded", "advanced"]
    assert data["levels"] == {"E": True, "G": True, "C": True, "R": False, "S": False}
    assert data["lease"]["held"] and data["lease"]["source"] == data["source"]["commit"]
    assert data["parameters"] == {"t_end": .1, "cells": [16, 8]}
    assert any("not all found mapped" in item for item in data["limits"])
    cohort.acquire_mutation_lease(site).release()  # closing the receipt released its lease


def test_receipt_fails_fast_while_another_source_qualifies(repo, site):
    with cohort.qualification_lease(site, "e" * 40, purpose="another worker"):
        with pytest.raises(cohort.InstallationBusy, match="needs its own installation"):
            receipts.Receipt("blocked", source_root=repo, package=site / "pops").open()


def test_receipt_records_a_failed_stage_and_reaches_no_level_past_it(repo, site, tmp_path):
    receipt = receipts.Receipt("failing case", source_root=repo, package=site / "pops",
                               directory=tmp_path)
    with pytest.raises(RuntimeError, match="compile failed"):
        with receipt:
            receipt.equations(repo / "examples/case.py")
            receipt._current = "C"
            raise RuntimeError("compile failed")
    data = receipts.read_receipt(receipt.path)
    assert data["status"] == "failed" and data["failures"][0]["stage"] == "C"
    assert data["level"] == "E"
    assert data["limits"][-1].startswith("failed in stage C: compile failed")


def test_an_observation_failing_at_close_is_recorded_and_the_receipt_written(
        repo, site, monkeypatch, tmp_path):
    def unavailable():
        raise RuntimeError("runtime report unavailable")
    monkeypatch.setattr(receipts, "backend_observation", unavailable)
    receipt = receipts.Receipt("close probe", source_root=repo, package=site / "pops",
                               directory=tmp_path)
    with receipt:
        receipt.equations(repo / "examples/case.py")
    data = receipts.read_receipt(receipt.path)
    assert data["status"] == "failed" and data["backend"] is None
    assert [row["stage"] for row in data["failures"]] == ["close:backend"]
    assert not receipts.admit(data, None)["admitted"]
    cohort.acquire_mutation_lease(site).release()


def test_overlay_findings_name_environment_overrides(repo, site, monkeypatch, tmp_path):
    receipt = receipts.Receipt("overlay probe", source_root=repo, package=site / "pops",
                               lease=False)
    receipt.open()
    monkeypatch.setenv("POPS_NATIVE_VARIANTS_ROOT", str(tmp_path))
    monkeypatch.setenv("POPS_INCLUDE", str(tmp_path / "missing"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "libshim.so"))
    findings = receipts.overlay_findings(receipt.data, receipt._sdk_files)["findings"]
    assert "the installation lease was not held" in findings
    assert any(item.startswith("POPS_NATIVE_VARIANTS_ROOT selects") for item in findings)
    assert any(item.startswith("POPS_INCLUDE=") for item in findings)
    assert any(item.startswith("LD_PRELOAD injects code") for item in findings)
    receipt.close()


def test_harness_must_be_committed_in_the_source_it_qualifies(repo):
    source = cohort.source_identity(repo)
    rows = receipts.committed_files(source, receipts.HARNESS)
    assert all(not row["committed"] for row in rows)  # the harness is not in this checkout
    case = receipts.committed_files(source, [repo / "examples/case.py"])
    assert case[0]["committed"] and case[0]["path"] == "examples/case.py"
    (repo / "examples/case.py").write_text("CASE = 'edited'\n")
    assert not receipts.committed_files(source, [repo / "examples/case.py"])[0]["committed"]


def test_lease_files_live_beside_the_package_and_survive_reinstall(site):
    lease = cohort.acquire_qualification_lease(site, "a" * 40)
    lease.release()
    state = cohort.state_directory(site)
    assert sorted(path.name for path in state.iterdir()) == ["guard.lock", "lease.json",
                                                             "lease.lock"]
    assert not (site / "pops" / cohort.STATE_DIRECTORY).exists()
    assert json.loads((state / "lease.json").read_text())["mode"] == "qualify"
    assert os.access(state, os.W_OK)
