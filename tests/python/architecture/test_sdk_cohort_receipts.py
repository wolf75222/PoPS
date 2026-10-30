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
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

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
#: Stand-ins for the SDK's lifecycle types, installed with the fake SDK: the receipt accepts an
#: object only when its class is defined by an SDK file that still has its installed bytes.
LIFECYCLE = """class _Record:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def verify(self):
        return None


class ResolvedSimulationPlan(_Record):
    pass


class CompiledSimulationArtifact(_Record):
    pass


class RuntimeInstance(_Record):
    pass


class RunReport(_Record):
    def to_data(self):
        return dict(self.data)
"""
#: Committed oracles of the fake checkout: one plain, one closing over its bound.
ORACLES = """def mass(runtime):
    return {"passed": runtime.mass == 0.0, "mass": runtime.mass}


def bounded(limit):
    def check(runtime):
        return {"passed": abs(runtime.mass) <= limit, "mass": runtime.mass}
    return check
"""
SOURCES = {
    "python/pops/__init__.py": "VERSION = 1\n",
    "python/pops/model.py": "def rate(u):\n    return -u\n",
    "python/pops/lifecycle.py": LIFECYCLE,
    "python/pops/py.typed": "",
    "include/pops_headers.manifest": "api pops/core.hpp\ntest-only pops/testing.hpp\n",
    "include/pops/core.hpp": "#pragma once\nint pops_core();\n",
    "include/pops/testing.hpp": "#pragma once\n",
    "src/native.cpp": "int pops_core() { return 1; }\n",
    "examples/case.py": "CASE = 'M01'\n",
    "examples/oracle.py": ORACLES,
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


def _wheel(path, leaf, row):
    """A retained wheel holding the leaf and its manifest, as the official build retains it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pops/_native/" + row["path"], leaf)
        archive.writestr("pops/_native/variants.json",
                         json.dumps({"schema_version": 1, "variants": [row]}))
    return path


def install(repo, site, *, dimension=2, leaf=b"native leaf", abi_headers=None, wheel=None):
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
    if wheel is None:
        wheel = _wheel(site.parent / "wheelhouse" / "pops-1.0.0-cp312-cp312-linux_x86_64.whl",
                       leaf, row)
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


def _verified(package, dimension):
    """Stands for ``verify_native_leaf``, which loads the leaf: a fake leaf cannot be loaded."""
    leaf = next((Path(package) / "_native" / f"dim{dimension}").iterdir())
    return {"dimension": dimension, "native_extension": str(leaf),
            "sha256": hashlib.sha256(leaf.read_bytes()).hexdigest()}


def _attest(repo, site, *, before=None, **options):
    """Attest as build_python.sh does: under the build's lease, with the source it started from."""
    before = cohort.source_identity(repo) if before is None else before
    with cohort.mutation_lease(site, purpose="build_python.sh --dim 2") as lease:
        return cohort.attest_native_build(repo, site / "pops", 2, source_before=before,
                                          lease=lease, verify=_verified, **options)


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


def test_a_rename_names_both_paths_and_a_move_out_of_the_sdk_inputs_dirties_them(repo):
    _git(repo, "mv", "tests/data.txt", "tests/moved.txt")
    source = cohort.source_identity(repo)
    assert source["modified"] == ["tests/data.txt", "tests/moved.txt"]
    assert not source["clean"] and source["sdk_inputs_clean"]
    _git(repo, "mv", "python/pops/model.py", "tests/model.py")
    moved = cohort.source_identity(repo)
    assert "python/pops/model.py" in moved["modified"]
    assert not moved["sdk_inputs_clean"]


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
    assert check["compared_files"] == 6  # 4 Python sources, 1 installed header, the manifest

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


def test_a_build_whose_source_changed_while_it_ran_attests_nothing(repo, site):
    before = cohort.source_identity(repo)
    (repo / "examples/case.py").write_text("CASE = 'M02'\n")
    _commit(repo, "a commit lands while the build runs")
    changed = _attest(repo, site, before=before)
    assert changed["source_changed_during_build"] == ["commit", "tree"]
    with pytest.raises(cohort.CohortError, match="its build saw the source change"):
        cohort.publish_cohort(repo, site / "pops")


def test_only_the_build_attests_a_leaf_and_only_one_that_loads(repo, site, tmp_path):
    before = cohort.source_identity(repo)
    with pytest.raises(cohort.CohortError, match="attested only by the build"):
        cohort.attest_native_build(repo, site / "pops", 2, source_before=before,
                                   verify=_verified)
    snapshot = tmp_path / "source-before.json"
    snapshot.write_text(json.dumps({"source": before}))
    attest = [sys.executable, "-I", str(SCRIPTS / "sdk_cohort.py"), "attest-build",
              "--source", str(repo), "--package", str(site / "pops"), "--dimension", "2",
              "--source-before", str(snapshot)]
    by_hand = _cli(*attest[3:])
    assert by_hand.returncode == 2 and "never attested by hand" in by_hand.stderr
    other = _cli("mutate", "--installation", str(site), "--purpose", "pip install", "--", *attest)
    assert other.returncode == 2 and "never attested by hand" in other.stderr
    # Inside the build's lease, the fake leaf is loaded in a child interpreter and refused.
    build = _cli("mutate", "--installation", str(site), "--purpose", "build_python.sh --dim 2",
                 "--", *attest)
    assert build.returncode == 2 and "does not verify" in build.stderr, build.stderr
    assert not cohort.native_build_directory(site).exists()


def test_a_prefix_installed_from_the_retained_wheel_finds_its_attestation(repo, site, tmp_path):
    wheel = next((site.parent / "wheelhouse").glob("*.whl"))
    attestation = _attest(repo, site, wheel=wheel)
    beside = cohort.wheel_attestation_path(wheel)
    assert json.loads(beside.read_text()) == attestation
    assert attestation["wheel"]["leaf"]["sha256"] == attestation["leaf_sha256"]
    assert attestation["environment"]["python"]["executable"] == sys.executable
    prefix = tmp_path / "qualification-prefix"
    install(repo, prefix, wheel=wheel)  # a fresh prefix: no attestation of its own
    record = cohort.publish_cohort(repo, prefix / "pops")
    leaf = record["native_provenance"]["leaves"][0]
    assert leaf["attestation"] == str(beside) and leaf["problem"] is None
    assert leaf["environment"]["python"]["executable"] == sys.executable
    assert cohort.verify_cohort(prefix / "pops", source_root=repo)["verified"]

    again = tmp_path / "second-prefix"
    install(repo, again, wheel=wheel)
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(cohort.CohortError, match="no official build attests it"):
        cohort.publish_cohort(repo, again / "pops")


def test_wheel_tags_and_dimensions_are_read_from_the_retained_wheel(site):
    wheel = next((site.parent / "wheelhouse").glob("*.whl"))
    assert cohort._wheel_tag(wheel) == "cp312-cp312-linux_x86_64"
    assert cohort._wheel_dimensions(wheel) == [2]
    assert cohort.build_tree_record(site, 2, wheel)["found"] is False


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


def test_lease_files_take_the_installation_mode_and_lock_read_only(site, monkeypatch):
    site.chmod(0o2775)
    cohort.acquire_qualification_lease(site, "a" * 40).release()
    state = cohort.state_directory(site)
    assert stat.S_IMODE(state.stat().st_mode) == 0o2775
    for name in ("guard.lock", "lease.lock"):
        assert stat.S_IMODE((state / name).stat().st_mode) == 0o664
        (state / name).chmod(0o444)  # as another user's lock file looks to this one
    cohort.acquire_qualification_lease(site, "a" * 40).release()

    def denied(*_arguments, **_options):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(cohort, "write_json_atomic", denied)
    with pytest.raises(cohort.CohortError, match="cannot write the lease record"):
        cohort.acquire_mutation_lease(site)
    monkeypatch.undo()
    cohort.acquire_mutation_lease(site).release()  # the refused attempt released its lock


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


def test_no_build_or_install_mutates_an_installation_that_holds_a_cohort(repo, site, tmp_path):
    _attest(repo, site)
    cohort.publish_cohort(repo, site / "pops")
    marker = tmp_path / "ran"
    build = _cli("mutate", "--installation", str(site), "--purpose", "pip install", "--",
                 sys.executable, "-c", "open(%r, 'w').close()" % str(marker))
    assert build.returncode == 2 and "a cohort never changes" in build.stderr
    assert not marker.exists()
    cohort.acquire_mutation_lease(site).release()  # the refusal released the lease


def test_build_python_runs_under_the_lease_and_attests_retained_wheels():
    build = (ROOT / "scripts" / "build_python.sh").read_text(encoding="utf-8")
    activate = build.index('conda activate "$ENV_NAME"')
    lease = build.index('sdk_cohort.py" mutate --installation "$POPS_SITE_PACKAGES"')
    install_command = build.index('python -m pip "${pip_args[@]}"')
    verify = build.index('verify_installed_native.py" "${native_verify_args[@]}"')
    attest = build.index('sdk_cohort.py" attest-build --source "$HERE"')
    before = build.index('sdk_cohort.py" identify --source "$HERE" > "$POPS_BUILD_SOURCE"')
    assert activate < build.index('sdk_cohort.py" held') < lease < before < install_command
    assert install_command < verify < attest
    assert 'ORIGINAL_ARGS=("$@")' in build
    assert '-- bash "$HERE/scripts/build_python.sh" "${ORIGINAL_ARGS[@]}"' in build
    assert '--source-before "$POPS_BUILD_SOURCE" --wheel "${built_wheels[0]}"' in build[attest:]
    assert '--purpose "build_python.sh --dim $NATIVE_DIM"' in build


# --------------------------------------------------------------------------- #
# Receipts                                                                     #
# --------------------------------------------------------------------------- #
def _evidence(stage):
    return {
        "E": {"files": [{"path": "examples/case.py", "sha256": "0" * 64, "committed": True}]},
        "G": {"verified": True, "semantic_identity": "plan",
              "plan_type": {"type": "pops.codegen._plans.ResolvedSimulationPlan", "in_sdk": True,
                            "expected": "ResolvedSimulationPlan", "is_expected": True}},
        "C": {"same_plan_as_G": True,
              "artifact_type": {"in_sdk": True, "expected": "CompiledSimulationArtifact",
                                "is_expected": True},
              "dsos": [
                  {"role": "program", "path": "/p.so", "sha256": "1" * 64, "artifact_sidecar": {}},
                  {"role": "model", "path": "/m.so", "sha256": "2" * 64}]},
        "loaded": {"all_mapped": True, "system": "System", "installed_program_matches": True,
                   "runtime_type": {"in_sdk": True}, "executor_type": {"in_sdk": True},
                   "native": {"sdk_leaf": True, "leaf_mapped": True, "system_of_leaf": True}},
        "advanced": {"same_artifact_as_C": True, "accepted_steps": 2, "same_runtime_as_loaded": True,
                     "report_type": {"in_sdk": True, "expected": "RunReport",
                                     "is_expected": True}},
        "S": {"passed": True, "name": "mass", "domain": "M01",
              "oracle": {"qualname": "check", "origin": "source", "committed": True},
              "arguments": [{"type": "pops.runtime._runtime_instance.RuntimeInstance",
                             "run_object": "runtime"}]},
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


@pytest.mark.parametrize(("stage", "key", "level", "limit"), [
    ("G", ("plan_type", "is_expected"), "E", "G: the plan is not the SDK's"),
    ("C", ("artifact_type", "is_expected"), "G", "C: the artifact is not the SDK's"),
    ("loaded", ("native", "sdk_leaf"), "C", "the SDK's native leaf is not the loaded pops._pops"),
    ("loaded", ("native", "leaf_mapped"), "C", "not mapped with its installed bytes"),
    ("loaded", ("native", "system_of_leaf"), "C", "the System is not a class of the SDK's"),
    ("loaded", ("runtime_type", "in_sdk"), "C", "the runtime is not an SDK object"),
    ("advanced", ("report_type", "is_expected"), "C", "did not advance through pops.run"),
    ("advanced", ("same_runtime_as_loaded",), "C", "did not advance through pops.run"),
    ("S", ("oracle", "committed"), "R", "is not a committed file of the source"),
])
def test_r_and_s_need_the_sdk_runtime_and_a_committed_oracle(published, stage, key, level, limit):
    _, source, sdk = published

    def falsify(data):
        evidence = next(row["evidence"] for row in data["stages"] if row["stage"] == stage)
        for name in key[:-1]:
            evidence = evidence[name]
        evidence[key[-1]] = False
    data = _receipt(source, sdk, patch=falsify)
    assert data["level"] == level
    assert any(limit in item for item in data["limits"]), data["limits"]


def test_r_needs_the_runtime_that_advanced_to_be_the_one_the_stage_loaded_saw(published):
    """A report recorded without its runtime is no R: the oracle could be called on another."""
    _, source, sdk = published
    assert inspect.signature(receipts.Receipt.advanced).parameters["runtime"].default \
        is inspect.Parameter.empty

    def unnamed(data):
        data["stages"][4]["evidence"]["same_runtime_as_loaded"] = None
    data = _receipt(source, sdk, patch=unnamed)
    assert data["level"] == "C"
    assert any("did not advance through pops.run" in item for item in data["limits"])


def test_an_oracle_called_on_anything_but_the_run_is_no_s(published):
    _, source, sdk = published

    def driver_value(data):
        data["stages"][-1]["evidence"]["arguments"] = [{"type": "builtins.dict",
                                                        "run_object": None}]
    data = _receipt(source, sdk, patch=driver_value)
    assert data["level"] == "R"
    assert data["limits"] == ["S: oracle check was not called on the runtime or report of the run"]


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


def test_a_receipt_is_self_consistent_and_its_levels_follow_from_its_stages(published, tmp_path):
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
    with pytest.raises(receipts.ReceiptError, match="not self-consistent"):
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


def test_ranks_agree_on_the_overlay_and_the_level_not_only_on_the_identities(repo, site,
                                                                              tmp_path):
    peer = {}
    receipt = receipts.Receipt("two ranks", source_root=repo, package=site / "pops",
                               directory=tmp_path, domain="source-only",
                               gather=lambda local: [local, peer.get("identity", local)])
    with receipt:
        receipt.equations(repo / "examples/case.py")
    receipt.data["backend"] = {"native_loaded": True, "runtime_environment": {
        "mpi_ranks": 2, "mpi_rank": 0, "communicator": "world"}}
    agreed = receipt._ranks()
    assert agreed["size"] == 2 and agreed["consensus"] is True
    seen = copy.deepcopy(receipt.data)
    # Rank 1 saw the same source, SDK and DSOs, but an overlay ...
    receipt.data["overlay"] = {"clean": False,
                               "findings": ["LD_PRELOAD injects code into the process: x"]}
    peer["identity"] = receipt._ranks()["local_identity"]
    receipt.data["overlay"] = seen["overlay"]
    assert receipt._ranks()["consensus"] is False
    # ... or reached another level.
    receipt.data["stages"] = []
    peer["identity"] = receipt._ranks()["local_identity"]
    receipt.data["stages"] = seen["stages"]
    assert receipt._ranks()["consensus"] is False


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


def _module_from(path, name):
    """Import one file as a module named ``name``, as a process imports its own code."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_a_run_without_pops_reaches_no_level_past_e(repo, site, tmp_path):
    """The reviewer's probe: plain objects and unmapped DSOs, no SDK and no native leaf."""
    plan = SimpleNamespace(verify=lambda: None, plan_identity=_Identity(token="plan"),
                           snapshot=SimpleNamespace(semantic_identity=_Identity(token="sem")),
                           resolved_dimension=2, target="system", backend="production")
    report = SimpleNamespace(to_data=lambda: {"accepted_steps": 2, "rejected_steps": 0,
                                              "final_time": .1, "stop_reason": "t_end"},
                             artifact_identity=_Identity(token="artifact"))
    runtime = SimpleNamespace(_executor=SimpleNamespace(_s=SimpleNamespace(
        installed_program_hash=lambda: "h0")), mass=0.0)
    oracles = _module_from(repo / "examples" / "oracle.py", "rom17_fake_oracles_a")
    receipt = receipts.Receipt("fake case", source_root=repo, package=site / "pops",
                               directory=tmp_path / "receipts", domain="source-only")
    with receipt:
        with pytest.raises(receipts.ReceiptError, match="only after the stage C"):
            receipt.loaded(runtime)
        receipt.equations(repo / "examples/case.py")
        receipt.plan(plan)
        receipt.compiled(_fake_artifact(tmp_path, "sem"))
        receipt.parameters(t_end=.1, cells=(16, 8))
        with pytest.raises(receipts.ReceiptError, match="only after the stage loaded"):
            receipt.advanced(report, runtime)
        receipt.loaded(runtime)
        receipt.advanced(report, runtime)
        receipt.property("mass", oracles.mass, runtime, domain="source-only")
    data = receipts.read_receipt(receipt.path)
    assert [row["stage"] for row in data["stages"]] == list(receipts.STAGES)
    assert data["level"] == "E", data["limits"]
    assert data["limits"][0] == "G: the plan is not the SDK's verified ResolvedSimulationPlan"
    assert any("the SDK's native leaf is not the loaded pops._pops" in item
               for item in data["limits"])
    assert data["lease"]["held"] and data["lease"]["source"] == data["source"]["commit"]
    assert data["parameters"] == {"t_end": .1, "cells": [16, 8]}
    findings = data["overlay"]["findings"]
    assert any(item.startswith("the receipt is not created by a committed file") for item in findings)
    assert not receipts.admit(data, None, require="E")["admitted"]
    cohort.acquire_mutation_lease(site).release()  # closing the receipt released its lease


def test_sdk_lifecycle_objects_reach_c_and_r_still_needs_the_native_leaf(repo, site, tmp_path):
    lifecycle = _module_from(site / "pops" / "lifecycle.py", "rom17_fake_sdk_lifecycle")
    oracles = _module_from(repo / "examples" / "oracle.py", "rom17_fake_oracles_b")
    plan = lifecycle.ResolvedSimulationPlan(
        plan_identity=_Identity(token="plan"), resolved_dimension=2, target="system",
        backend="production", snapshot=SimpleNamespace(semantic_identity=_Identity(token="sem")))
    fake = _fake_artifact(tmp_path, "sem")
    artifact = lifecycle.CompiledSimulationArtifact(**vars(fake))
    runtime = lifecycle.RuntimeInstance(_executor=lifecycle.RuntimeInstance(
        _s=SimpleNamespace(installed_program_hash=lambda: "h0")), mass=0.0)
    report = lifecycle.RunReport(data={"accepted_steps": 2}, artifact_identity=_Identity(
        token="artifact"))
    receipt = receipts.Receipt("fake SDK case", source_root=repo, package=site / "pops",
                               directory=tmp_path, domain="source-only")
    with receipt:
        receipt.equations(repo / "examples/case.py")
        receipt.plan(plan)
        receipt.compiled(artifact)
        receipt.loaded(runtime)
        receipt.advanced(report, runtime)
        receipt.property("mass", oracles.bounded(1.e-12), runtime, domain="source-only")
        receipt.property("mass of a value", oracles.mass, SimpleNamespace(mass=0.0),
                         domain="source-only")
        with pytest.raises(receipts.ReceiptError, match="returned no boolean verdict"):
            receipt.property("no verdict", lambda run: {"mass": 0.0}, runtime, domain="x")
    data = receipts.read_receipt(receipt.path)
    stages = {row["stage"]: row["evidence"] for row in data["stages"] if row["stage"] != "S"}
    assert stages["G"]["plan_type"] == {"type": "rom17_fake_sdk_lifecycle.ResolvedSimulationPlan",
                                        "in_sdk": True, "expected": "ResolvedSimulationPlan",
                                        "is_expected": True}
    assert stages["C"]["artifact_type"]["is_expected"]
    assert stages["loaded"]["runtime_type"]["in_sdk"]
    assert stages["advanced"]["report_type"]["is_expected"]
    assert stages["advanced"]["same_runtime_as_loaded"] is True
    assert data["level"] == "C"
    assert data["limits"][0].startswith("R: not loaded by the SDK's runtime (the SDK's native leaf")
    bounded, value = [row["evidence"] for row in data["stages"] if row["stage"] == "S"]
    assert bounded["passed"] and bounded["oracle"]["origin"] == "source"
    assert bounded["oracle"]["committed"] and bounded["oracle"]["file"] == "examples/oracle.py"
    assert bounded["oracle"]["closure"] == {"limit": 1.e-12}
    assert bounded["arguments"] == [{"type": "rom17_fake_sdk_lifecycle.RuntimeInstance",
                                     "run_object": "runtime"}]
    assert receipts._property_problems(bounded) == []
    assert receipts._property_problems(value) == [
        "oracle mass was not called on the runtime or report of the run"]
    assert data["failures"] == []  # the refused oracle raised in the driver, which caught it


def _main_module(path=None, spec=None):
    """A ``__main__`` as the interpreter leaves it: a file (or none) and the module ``-m`` ran."""
    main = ModuleType("__main__")
    if path is not None:
        main.__file__ = str(path)
    main.__spec__ = None if spec is None else SimpleNamespace(name=spec)
    return main


def test_the_code_of_the_process_is_judged_where_it_comes_from(repo, site, monkeypatch):
    source = cohort.source_identity(repo)
    package = site / "pops"
    interpreter = json.__file__
    case = str(repo / "examples" / "case.py")
    monkeypatch.setitem(sys.modules, "__main__", _main_module(case))
    clean = receipts.code_observation(source, package, [case, interpreter, case])
    assert [frame["origin"] for frame in clean["driver"]] == ["source", "interpreter", "source"]
    assert clean["entry"] == {"file": case, "module": None, "origin": "source"}
    # This test process also runs the real checkout's modules, foreign to the fake one: only the
    # entry and the driver are judged clean here.
    assert [item for item in receipts._code_findings(clean) if not item.startswith("module ")] \
        == []

    stray = repo.parent / "driver.py"
    stray.write_text("pass\n")
    planted = ModuleType("rom17_planted")
    planted.__file__ = str(stray)
    monkeypatch.setitem(sys.modules, "rom17_planted", planted)
    edited = ModuleType("rom17_edited_case")
    edited.__file__ = case
    monkeypatch.setitem(sys.modules, "rom17_edited_case", edited)
    (repo / "examples" / "case.py").write_text("CASE = 'edited'\n")
    judged = receipts.code_observation(
        source, package, [interpreter, "<string>", "<stdin>", "<frozen runpy>", str(stray),
                          str(package / "model.py")])
    findings = receipts._code_findings(judged)
    assert "the receipt is driven by code compiled from a string (python -c, exec): <string>" \
        in findings
    assert "the receipt is driven by code read from standard input: <stdin>" in findings
    assert not any(item.endswith("<frozen runpy>") for item in findings)
    assert any(item.startswith("the receipt is driven by code outside the source") and
               item.endswith(str(stray)) for item in findings)
    assert any(item.startswith("the receipt is driven by the SDK it qualifies") for item in findings)
    assert any(item.startswith("the receipt is not created by a committed file") for item in findings)
    assert "module rom17_planted loaded from %s, outside the source, the SDK and the interpreter" \
        % stray in findings
    assert any(item.startswith("module rom17_edited_case (examples/case.py) is not committed")
               for item in findings)
    assert receipts._code_findings(None) == ["the code of the process was not judged"]


def test_the_most_specific_root_names_the_origin_of_code(tmp_path):
    checkout = tmp_path / "checkout"
    venv = checkout / ".venv"
    package = venv / "lib" / "site-packages" / "pops"
    assert receipts.code_origin(str(checkout / "examples" / "m.py"), checkout, package) == "source"
    assert receipts.code_origin(str(package / "model.py"), checkout, package) == "sdk"
    assert receipts.code_origin("<frozen runpy>", checkout, package) == "internal"
    assert receipts.code_origin(str(tmp_path / "elsewhere.py"), checkout, package) == "foreign"


def test_a_receiver_process_starts_from_a_committed_file_or_python_m_pytest(repo, site,
                                                                           monkeypatch):
    """The entry is judged: code that runs before the receipt cannot hide outside its frames."""
    source = cohort.source_identity(repo)
    package = site / "pops"
    case = repo / "examples" / "case.py"

    def entry_findings(main):
        monkeypatch.setitem(sys.modules, "__main__", main)
        code = receipts.code_observation(source, package, [str(case)])
        return [item for item in receipts._code_findings(code)
                if item.startswith("the process was")]
    assert entry_findings(_main_module(case)) == []
    assert entry_findings(_main_module(pytest.__file__.replace("__init__", "__main__"),
                                       spec="pytest.__main__")) == []
    # python -c, standard input or an interactive prompt: no file, nothing judged before the receipt.
    assert entry_findings(_main_module()) == [
        "the process was not started from a file (python -c, standard input or an interactive "
        "prompt): the code it ran first is not judged"]
    # An interpreter entry that runs the code it is given (python -m cProfile driver.py).
    import cProfile
    assert entry_findings(_main_module(cProfile.__file__, spec="cProfile")) == [
        "the process was started by the interpreter's cProfile, which runs the code it is given; "
        "a receiver is started from a committed file or by python -m pytest"]
    assert entry_findings(_main_module(package / "model.py", spec="pops.model")) == [
        "the process was started by the SDK it qualifies: %s" % (package / "model.py")]


def test_only_the_standard_library_and_the_site_directories_are_the_interpreter(repo, site):
    import site as interpreter_site
    import sysconfig

    roots = receipts._interpreter_roots()
    base = sysconfig.get_paths(vars={"base": sys.base_prefix, "platbase": sys.base_exec_prefix,
                                     "installed_base": sys.base_prefix,
                                     "installed_platbase": sys.base_exec_prefix})
    assert {Path(base["stdlib"]), *map(Path, interpreter_site.getsitepackages())} <= set(roots)
    package = site / "pops"

    def origin(path):
        return receipts.code_origin(str(path), repo, package)
    assert origin(json.__file__) == "interpreter" and origin(pytest.__file__) == "interpreter"
    standard = receipts._standard_library()
    assert Path(base["stdlib"]) in standard and set(standard) <= set(roots)
    assert receipts.interpreter_part(json.__file__, roots, standard) == {
        "library": "stdlib", "package": "json"}
    assert receipts.interpreter_part(pytest.__file__, roots, standard) == {
        "library": "site", "package": "pytest"}
    # The rest of a prefix is no library: its root, bin/ and a stray script are foreign.
    for stray in (Path(sys.prefix) / "driver.py",
                  Path(sysconfig.get_paths()["scripts"]) / "pytest"):
        assert all(stray != root and not stray.is_relative_to(root) for root in roots), stray
        assert origin(stray) == "foreign"


def test_the_interpreter_drives_a_receipt_only_through_its_standard_library_and_pytest(
        repo, site, tmp_path, monkeypatch):
    """The final review's O8: a module placed in a site directory is no driver of a receipt."""
    library = tmp_path / "env" / "lib" / "python3"
    packages = library / "site-packages"  # inside the standard library, as in a conda environment
    roots, standard = (library, packages), (library,)

    def part(path):
        return receipts.interpreter_part(str(path), roots, standard)
    assert part(library / "runpy.py") == {"library": "stdlib", "package": "runpy"}
    assert part(library / "importlib" / "util.py") == {"library": "stdlib", "package": "importlib"}
    assert part(library / "adhoc.py") == {"library": None, "package": "adhoc"}
    assert part(packages / "_pytest" / "main.py") == {"library": "site", "package": "_pytest"}
    assert part(packages / "adhoc_plugin.py") == {"library": "site", "package": "adhoc_plugin"}
    assert part(tmp_path / "elsewhere.py") == {"library": None, "package": None}

    source = cohort.source_identity(repo)
    case = str(repo / "examples" / "case.py")
    monkeypatch.setitem(sys.modules, "__main__", _main_module(case))

    def findings(*driver):
        code = receipts.code_observation(source, site / "pops", [case, *map(str, driver), case],
                                         interpreter=roots, standard=standard, prefixes=())
        # This process's own modules lie outside the fake interpreter, whose files no manifest
        # records: only the frames are judged here.
        return [item for item in receipts._code_findings(code)
                if not item.startswith(("module ", "interpreter file "))]
    assert findings(library / "runpy.py", packages / "pytest" / "__main__.py",
                    packages / "_pytest" / "python.py", packages / "pluggy" / "_callers.py") == []
    planted = ("the receipt is driven by code of the interpreter that is neither its standard "
               "library nor pytest (a file placed in a site directory or beside the standard "
               "library): %s")
    assert findings(packages / "_pytest" / "main.py", packages / "adhoc_plugin.py") == [
        planted % (packages / "adhoc_plugin.py")]
    assert findings(library / "adhoc.py") == [planted % (library / "adhoc.py")]
    # A frame's file is judged against the manifests even when no module holds it.
    code = receipts.code_observation(source, site / "pops", [case, str(packages / "pluggy.py"),
                                                             case],
                                     interpreter=roots, standard=standard, prefixes=())
    assert [(row["path"], row["module"], row["verdict"])
            for row in code["interpreter_files"]["refused"]] == [
        (str(packages / "pluggy.py"), None, "unrecorded")]


def test_a_receipt_is_created_under_the_entry_of_its_process():
    """atexit, a thread or another callback of the interpreter leaves the entry off the frames."""
    entry = {"file": "/env/site/pytest/__main__.py", "module": "pytest.__main__",
             "origin": "interpreter"}

    def frame(path, library="site", package="_pytest"):
        return {"file": path, "origin": "interpreter", "library": library, "package": package}

    def findings(*frames):
        return receipts._code_findings({"entry": entry, "driver": list(frames),
                                        "interpreter_files": {"refused": []}})
    runpy = {"file": "<frozen runpy>", "origin": "internal"}
    main = frame(entry["file"], package="pytest")
    receiver = {"file": "/checkout/tests/test_receiver.py", "origin": "source", "committed": True}
    assert findings(runpy, runpy, main, frame("/env/site/_pytest/python.py"), receiver) == []
    # Without frozen modules, runpy runs from the standard library.
    stdlib_runpy = frame("/env/lib/runpy.py", "stdlib", "runpy")
    assert findings(stdlib_runpy, stdlib_runpy, main, receiver) == []
    elsewhere = ("the receipt is not created under the process entry %s: its outermost frame is "
                 "%s (a receiver called from atexit, a thread or another callback of the "
                 "interpreter)")
    assert findings(receiver) == [elsewhere % (entry["file"], receiver["file"])]  # from atexit
    thread = frame("/env/lib/threading.py", "stdlib", "threading")
    assert findings(thread, thread, receiver) == [elsewhere % (entry["file"], thread["file"])]


def _distribution(site, name, rows=None, *, top_level=(), installed=None, sources=None):
    """A distribution installed in ``site``: a dist-info whose RECORD has ``rows`` (bytes are
    written as they are), or, without rows, a legacy egg-info with the ``installed-files.txt``
    and ``SOURCES.txt`` lines given, if any."""
    if rows is None:
        info = site / f"{name}-1.0.egg-info"
        info.mkdir(parents=True)
        (info / "PKG-INFO").write_text(f"Metadata-Version: 1.1\nName: {name}\nVersion: 1.0\n")
        (info / "top_level.txt").write_text("".join(f"{top}\n" for top in top_level))
        for listing, lines in (("installed-files.txt", installed), ("SOURCES.txt", sources)):
            if lines is not None:
                (info / listing).write_text("".join(f"{line}\n" for line in lines))
        return info
    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    record = rows if isinstance(rows, bytes) else "".join(
        f"{path},{digest},\n" for path, digest in [*rows, (f"{info.name}/RECORD", "")]).encode()
    (info / "RECORD").write_bytes(record)
    return info


def test_the_files_of_the_interpreter_are_judged_against_the_manifests_of_the_environment(
        tmp_path):
    """The O9 review: a module of a site directory can drive a receiver without a frame of its own
    (a ``functools.partial`` bound as a pytest hook), so the file itself is judged.  Every file the
    interpreter runs from a site directory is one that a RECORD or conda-meta records with its
    bytes (a legacy egg's installed-files.txt lists files without digests), or one of a
    distribution that lists no installed file; in a conda environment the standard library is
    judged too, even when conda's record of it cannot be read."""
    env = tmp_path / "env"
    library = env / "lib" / "python3"
    packages = library / "site-packages"  # inside the standard library, as in a conda environment
    system = tmp_path / "usr" / "dist-packages"
    roots, standard = (library, packages, system), (library,)

    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    stdlib = {name: write(library / relative, f"# {relative}\n") for name, relative in (
        ("os", "os.py"), ("json", "json/__init__.py"), ("_sysconfigdata", "_sysconfigdata.py"),
        ("adhoc", "adhoc.py"))}
    good = write(packages / "good" / "__init__.py", "GOOD = 1\n")
    dropped = write(packages / "good" / "dropped.py", "pass\n")  # inside an installed package
    bad = write(packages / "bad.py", "BAD = 2\n")
    plain = write(packages / "plain.py", "PLAIN = 1\n")
    plugin = write(packages / "adhoc_plugin.py", "pytest_configure = print\n")
    broken = write(packages / "broken.py", "BROKEN = 1\n")
    frame = write(packages / "frame_only.py", "pass\n")
    legacy = write(system / "legacy" / "__init__.py", "LEGACY = 1\n")
    egg = write(system / "egg" / "__init__.py", "EGG = 1\n")
    sourced = write(system / "sourced" / "__init__.py", "SOURCED = 1\n")
    orphan = write(system / "orphan.py", "ORPHAN = 1\n")
    namesake = write(packages / "legacy" / "__init__.py", "LEGACY = 2\n")  # the egg's name
    _distribution(packages, "good", [("good/__init__.py", _record_hash(good))])
    bad.write_text("BAD = 1\n")
    _distribution(packages, "bad", [("bad.py", _record_hash(bad))])
    bad.write_text("BAD = 2\n")  # changed after it was installed
    _distribution(packages, "plain", [("plain.py", "")])
    _distribution(packages, "broken", b"broken.py,\xff\xfe,\n")  # no RECORD can be read
    _distribution(system, "legacy", top_level=("legacy",))
    # A legacy egg lists its installed files from its egg-info, a script outside the site among
    # them; SOURCES.txt names the files of a source tree (src/ layout here), not installed ones.
    _distribution(system, "egg", top_level=("egg",),
                  installed=["../egg/__init__.py", "../../bin/egg-script"])
    _distribution(system, "sourced", top_level=("sourced",),
                  sources=["setup.py", "src/sourced/__init__.py"])
    modules = {**stdlib, "good": good, "good.dropped": dropped, "bad": bad, "plain": plain,
               "adhoc_plugin": plugin, "broken": broken, "legacy": legacy, "egg": egg,
               "sourced": sourced, "orphan": orphan, "shadow_legacy": namesake}
    files = {os.path.realpath(path): {"module": name, "path": str(path)}
             for name, path in modules.items()}
    files[os.path.realpath(frame)] = {"module": None, "path": str(frame)}

    def judge(*prefixes):
        return receipts.judge_interpreter_files(
            files, receipts.interpreter_manifest(roots, standard, prefixes), roots, standard)

    def rows(judged, key):
        return {(row["path"], row["verdict"], row["library"], row.get("by"))
                for row in judged[key]}
    unrecorded = {(str(path), "unrecorded", "site", None)
                  for path in (dropped, plugin, broken, frame, orphan, namesake)}
    unverified = {(str(plain), "listed", "site", "plain 1.0"),
                  (str(egg), "listed", "site", "egg 1.0"),
                  (str(legacy), "unmanaged", "site", "legacy 1.0"),
                  (str(sourced), "unmanaged", "site", "sourced 1.0")}
    judged = judge()
    assert judged["counts"] == {"recorded": 1, "listed": 2, "unmanaged": 2, "believed": 4,
                                "altered": 1, "unrecorded": 6}
    assert judged["standard_library"] == "believed"
    assert judged["manifests"] == {"conda": [], "distributions": 4,
                                   "unmanaged": ["legacy 1.0", "sourced 1.0"],
                                   "unreadable": ["broken 1.0"]}
    assert rows(judged, "unverified") == unverified
    assert rows(judged, "refused") == unrecorded | {(str(bad), "altered", "site", "bad 1.0")}
    code = {"entry": {"file": "/checkout/case.py", "module": None, "origin": "source"},
            "driver": [{"file": "/checkout/case.py", "origin": "source", "committed": True}],
            "interpreter_files": judged}
    placed = ("interpreter file %s (%s) is recorded by no manifest of the environment (no "
              "RECORD, no conda-meta): code placed %s")
    assert sorted(receipts._code_findings(code)) == sorted([
        placed % (dropped, "module good.dropped", "in a site directory"),
        placed % (plugin, "module adhoc_plugin", "in a site directory"),
        placed % (broken, "module broken", "in a site directory"),
        placed % (frame, "a frame of the driver", "in a site directory"),
        placed % (orphan, "module orphan", "in a site directory"),
        placed % (namesake, "module shadow_legacy", "in a site directory"),
        "interpreter file %s (module bad) does not have the bytes bad 1.0 records: an installed "
        "file was changed" % bad])
    assert receipts._code_findings({**code, "interpreter_files": None}) == [
        "the interpreter's files were not judged against the environment's manifests"]
    assert receipts._code_findings({**code, "interpreter_files": {"error": "OSError: denied"}}) \
        == ["the interpreter's files were not judged against the environment's manifests "
            "(OSError: denied)"]

    # A conda environment's conda-meta records its standard library and what conda installed.
    placeholder = "/opt/anaconda1anaconda2anaconda3"
    write(env / "conda-meta" / "python-3.12.0-h0_0.json", json.dumps({
        "name": "python", "version": "3.12.0", "build": "h0_0", "paths_data": {
            "paths_version": 1, "paths": [
                {"_path": "lib/python3/os.py", "path_type": "hardlink",
                 "sha256": sha256(stdlib["os"])},
                {"_path": "lib/python3/json/__init__.py", "path_type": "hardlink",
                 "sha256": "0" * 64, "prefix_placeholder": placeholder, "file_mode": "text",
                 "sha256_in_prefix": sha256(stdlib["json"])},
                {"_path": "lib/python3/_sysconfigdata.py", "path_type": "hardlink",
                 "sha256": "0" * 64, "prefix_placeholder": placeholder, "file_mode": "text"},
                {"_path": "lib/python3/__pycache__/os.cpython-312.pyc", "path_type": "pyc_file"},
                {"_path": "lib/python3/site-packages/bad.py", "path_type": "hardlink",
                 "sha256": sha256(bad)}]}}))
    write(env / "conda-meta" / "unreadable-1.0-0.json", "{")
    judged = judge(env)
    assert judged["counts"] == {"recorded": 4, "listed": 3, "unmanaged": 2, "believed": 0,
                                "altered": 0, "unrecorded": 7}
    assert judged["standard_library"] == "judged"
    assert judged["manifests"]["conda"] == [str(env)]
    assert judged["manifests"]["unreadable"] == [
        str(env / "conda-meta" / "unreadable-1.0-0.json"), "broken 1.0"]
    assert rows(judged, "unverified") == unverified | {
        (str(stdlib["_sysconfigdata"]), "listed", "stdlib", "python 3.12.0 (conda)")}
    # bad.py has the bytes conda recorded; adhoc.py lies beside a standard library conda lists.
    assert rows(judged, "refused") == unrecorded | {
        (str(stdlib["adhoc"]), "unrecorded", "stdlib", None)}
    assert placed % (stdlib["adhoc"], "module adhoc", "in or beside the standard library") in \
        receipts._code_findings({**code, "interpreter_files": judged})

    # Without conda's record of python, the standard library of the conda prefix records nothing.
    (env / "conda-meta" / "python-3.12.0-h0_0.json").unlink()
    judged = judge(env)
    assert judged["standard_library"] == "judged"
    assert judged["counts"] == {"recorded": 1, "listed": 2, "unmanaged": 2, "believed": 0,
                                "altered": 1, "unrecorded": 10}
    assert rows(judged, "refused") == unrecorded | {(str(bad), "altered", "site", "bad 1.0")} | {
        (str(path), "unrecorded", "stdlib", None) for path in stdlib.values()}


def test_files_whose_manifests_cannot_be_read_are_recorded_as_not_judged(repo, site, monkeypatch):
    def unreadable(*arguments):
        raise RuntimeError("no manifest")
    monkeypatch.setattr(receipts, "interpreter_manifest", unreadable)
    code = receipts.code_observation(cohort.source_identity(repo), site / "pops", [])
    assert code["interpreter_files"] == {"error": "RuntimeError: no manifest"}
    assert "the interpreter's files were not judged against the environment's manifests " \
        "(RuntimeError: no manifest)" in receipts._code_findings(code)


#: A receiver committed in the fake checkout: it writes one receipt (stage E) under a false label.
RECEIVER = """from pathlib import Path


def qualify(repo, package, directory):
    import qualification_receipt as receipts
    receipt = receipts.Receipt("M05 compiled, loaded and advanced (label only)",
                               source_root=Path(repo), package=Path(package),
                               directory=Path(directory), domain="M01")
    with receipt:
        receipt.equations(Path(repo) / "examples" / "case.py")
    return receipt.path
"""
#: A driver of that receiver (the reviewer's probe): run by ``python -c``, from standard input,
#: as a script placed in a prefix, or committed and run directly or under an interpreter entry.
DRIVER = """import sys
sys.path[:0] = [{scripts!r}, {examples!r}]
import receiver
{call}
"""
#: The same driver as a pytest plugin placed in a site directory (the final review's O8), loaded
#: by ``python -m pytest -p`` and calling the receiver from ``pytest_configure``.
PLUGIN = """import sys
sys.path[:0] = [{scripts!r}, {examples!r}]
import receiver


def pytest_configure(config):
    {call}
"""
#: The same driver handed to pytest as a hook that opens no frame of its own (the O9 review): a
#: bound ``functools.partial`` of the receiver, which pluggy calls from its own frame.
FRAMELESS_PLUGIN = """import functools
import sys
import types
sys.path[:0] = [{scripts!r}, {examples!r}]
import receiver

pytest_configure = types.MethodType(functools.partial(receiver.qualify, package={package!r},
                                                      directory={directory!r}), {repo!r})
"""
#: A committed test that runs the receiver, as a lot's receiver runs under ``python -m pytest``.
RECEIVER_TEST = """import sys
sys.path[:0] = [{scripts!r}, {examples!r}]
import receiver


def test_receiver():
    {call}
"""


def _venv_with_these_site_packages(prefix, *, copied=("_pytest",)):
    """A fresh venv whose site directory links this interpreter's packages (pytest among them).

    ``--system-site-packages`` would expose the base interpreter's packages, not those of the venv
    this test may run in.  The packages named in ``copied`` are copies, which a probe may change
    without changing this interpreter's own; their RECORD, linked, still records them.
    """
    import shutil
    import site as interpreter_site

    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True)
    python = str(prefix / "bin" / "python")
    packages = Path(subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_path("
                                    "'purelib'))"], capture_output=True, text=True,
                                   check=True).stdout.strip())
    for directory in filter(os.path.isdir, interpreter_site.getsitepackages()):
        for item in Path(directory).iterdir():
            if item.suffix != ".pth" and item.name != "__pycache__" \
                    and not (packages / item.name).exists():
                if item.name in copied:
                    shutil.copytree(item, packages / item.name,
                                    ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    (packages / item.name).symlink_to(item)
    return python, packages


@pytest.mark.parametrize("entry", ("committed file", "python -c", "standard input",
                                   "script in a prefix", "python -m cProfile",
                                   "python -c through atexit", "pytest plugin in a site directory",
                                   "pytest plugin through atexit", "pytest plugin without a frame",
                                   "pytest plugin inside pytest's package",
                                   "committed test under python -m pytest",
                                   "python -m pytest with a changed pytest file"))
def test_an_ad_hoc_driver_of_a_committed_receiver_is_refused_at_every_entry(repo, site, tmp_path,
                                                                            entry):
    """The reviews' probes: a committed receiver writes an admitted receipt only for its own driver.

    Only a committed file or python -m pytest starts a receiver, and no ad hoc code drives it:
    not from a string, standard input or a prefix, nor from a plugin placed in a site directory
    or inside an installed package, directly, through atexit or through a hook that opens no frame
    of its own.  An installed file whose bytes changed is refused too.
    """
    scripts = repo / "scripts"
    scripts.mkdir()
    for path in receipts.HARNESS:  # the receipt's own harness, committed in this checkout
        (scripts / path.name).write_bytes(path.read_bytes())
    (repo / "examples" / "receiver.py").write_text(RECEIVER)
    (repo / "examples" / "test_probe.py").write_text("def test_probe():\n    pass\n")
    out = tmp_path / "receipts"
    arguments = "%r, %r, %r" % (str(repo), str(site / "pops"), str(out))
    # Through atexit, the receiver runs with none of the driver's frames above it.
    call = ("import atexit; atexit.register(receiver.qualify, %s)" % arguments
            if entry.endswith("through atexit") else "receiver.qualify(%s)" % arguments)
    driver = DRIVER.format(scripts=str(scripts), examples=str(repo / "examples"), call=call)
    committed = repo / "examples" / "drive.py"
    committed.write_text(driver)
    (repo / "examples" / "test_receiver.py").write_text(RECEIVER_TEST.format(
        scripts=str(scripts), examples=str(repo / "examples"), call=call))
    _commit(repo, "a committed receiver and its driver")
    _attest(repo, site)
    record = cohort.publish_cohort(repo, site / "pops")
    python = sys.executable
    plugin = packages = None
    if entry == "script in a prefix":
        prefix = tmp_path / "prefix"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages",
                        str(prefix)], check=True)
        python = str(prefix / "bin" / "python")
        (prefix / "driver.py").write_text(driver)
    elif "pytest" in entry:
        python, packages = _venv_with_these_site_packages(tmp_path / "prefix")
        plugin = packages / ("_pytest" if entry.endswith("package") else ".") / \
            "rom17_adhoc_plugin.py"
        template = FRAMELESS_PLUGIN if entry.endswith("without a frame") else PLUGIN
        if entry.startswith("pytest plugin"):
            plugin.write_text(template.format(
                scripts=str(scripts), examples=str(repo / "examples"), call=call,
                repo=str(repo), package=str(site / "pops"), directory=str(out)))
        if entry.endswith("changed pytest file"):
            with open(packages / "_pytest" / "__init__.py", "a") as stream:
                stream.write("# changed after it was installed\n")
    plugin_name = "_pytest.rom17_adhoc_plugin" if entry.endswith("package") else \
        "rom17_adhoc_plugin"
    pytest_run = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", plugin_name,
                  str(repo / "examples" / "test_probe.py")]
    receiver_test = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     str(repo / "examples" / "test_receiver.py")]
    command = {
        "committed file": [python, str(committed)],
        "python -c": [python, "-c", driver],
        "python -c through atexit": [python, "-c", driver],
        "standard input": [python, "-"],
        "script in a prefix": [python, str(tmp_path / "prefix" / "driver.py")],
        "python -m cProfile": [python, "-m", "cProfile", str(committed)],
        "pytest plugin in a site directory": pytest_run,
        "pytest plugin through atexit": pytest_run,
        "pytest plugin without a frame": pytest_run,
        "pytest plugin inside pytest's package": pytest_run,
        "committed test under python -m pytest": receiver_test,
        "python -m pytest with a changed pytest file": receiver_test,
    }[entry]
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
                   and not key.startswith("POPS_")}
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"  # only the plugin named by -p
    result = subprocess.run(command, input=driver if entry == "standard input" else None,
                            capture_output=True, text=True, env=environment, cwd=tmp_path,
                            check=False)
    assert result.returncode == 0, result.stderr
    (path,) = out.glob("*.json")
    data = receipts.read_receipt(path)
    findings = data["overlay"]["findings"]
    started = data["process"]["code"]["entry"]
    placed = ("interpreter file %s (module %s) is recorded by no manifest of the environment (no "
              "RECORD, no conda-meta): code placed in a site directory" % (plugin, plugin_name))
    changed = packages and (
        "interpreter file %s (module _pytest) does not have the bytes pytest %s records: an "
        "installed file was changed" % (packages / "_pytest" / "__init__.py",
                                        importlib.metadata.version("pytest")))
    expected = {
        "committed file": [],
        "python -c": [
            "the process was not started from a file (python -c, standard input or an "
            "interactive prompt): the code it ran first is not judged",
            "the receipt is driven by code compiled from a string (python -c, exec): <string>"],
        "python -c through atexit": [
            "the process was not started from a file (python -c, standard input or an "
            "interactive prompt): the code it ran first is not judged"],
        "standard input": [
            "the process was not started from a file (python -c, standard input or an "
            "interactive prompt): the code it ran first is not judged",
            "the receipt is driven by code read from standard input: <stdin>"],
        "script in a prefix": [
            "the receipt is driven by code outside the source, the SDK and the interpreter: %s"
            % (tmp_path / "prefix" / "driver.py"),
            "module __main__ loaded from %s, outside the source, the SDK and the interpreter"
            % (tmp_path / "prefix" / "driver.py")],
        "python -m cProfile": [
            "the process was started by the interpreter's cProfile, which runs the code it is "
            "given; a receiver is started from a committed file or by python -m pytest"],
        "pytest plugin in a site directory": [
            "the receipt is driven by code of the interpreter that is neither its standard "
            "library nor pytest (a file placed in a site directory or beside the standard "
            "library): %s" % plugin, placed],
        "pytest plugin through atexit": [
            "the receipt is not created under the process entry %s: its outermost frame is %s "
            "(a receiver called from atexit, a thread or another callback of the interpreter)"
            % (started["file"], repo / "examples" / "receiver.py"), placed],
        # pluggy calls the bound partial: no frame of the plugin, only its file is refused.
        "pytest plugin without a frame": [placed],
        # A frame of _pytest drives the receipt; the file is no part of pytest's RECORD.
        "pytest plugin inside pytest's package": [placed],
        "committed test under python -m pytest": [],
        "python -m pytest with a changed pytest file": [changed],
    }[entry]
    assert findings == expected, findings
    if "pytest" in entry:
        assert started["module"] == "pytest.__main__" and started["origin"] == "interpreter"
    assert data["process"]["code"]["driver"][-1]["file"] == str(repo / "examples" / "receiver.py")
    assert data["level"] == "E"
    assert receipts.admit(data, record, require="E")["admitted"] is (
        entry in ("committed file", "committed test under python -m pytest"))


def test_the_cohort_says_whether_each_leaf_was_built_in_an_official_environment(
        repo, site, monkeypatch):
    """ROM-17 criterion 2 reads off the cohort: a stand-in over a venv is not official."""
    official = {"conda": {"prefix": "/envs/pops", "conda_meta": True,
                          "list_explicit": {"sha256": "a" * 64, "error": None}},
                "python": {"executable": "/envs/pops/bin/python"}}
    assert cohort.official_environment(official) == {"official": True, "reasons": []}
    stand_in = {"conda": {"prefix": "/venv", "conda_meta": False,
                          "list_explicit": {"sha256": None, "error": "not a conda prefix"}},
                "python": {"executable": "/usr/bin/python3"}}
    assert cohort.official_environment(stand_in)["reasons"] == [
        "CONDA_PREFIX /venv has no conda-meta: not a conda environment",
        "conda list --explicit did not answer: not a conda prefix",
        "the build's interpreter /usr/bin/python3 is not the environment's"]
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    _attest(repo, site)
    record = cohort.publish_cohort(repo, site / "pops")
    provenance = record["native_provenance"]
    assert provenance["attested"] and provenance["official_environment"] is False
    assert provenance["leaves"][0]["official_environment"] == {"official": False, "reasons": [
        "the build ran outside a conda environment (no CONDA_PREFIX)",
        "conda list --explicit did not answer: not run"]}
    assert cohort.verify_cohort(site / "pops")["official_environment"] is False


def test_receipt_fails_fast_while_another_source_qualifies(repo, site):
    with cohort.qualification_lease(site, "e" * 40, purpose="another worker"):
        with pytest.raises(cohort.InstallationBusy, match="needs its own installation"):
            receipts.Receipt("blocked", source_root=repo, package=site / "pops").open()


def test_a_receipt_takes_its_lease_before_it_identifies_the_sdk(repo, site, tmp_path,
                                                                 monkeypatch):
    held = []
    identify = cohort.sdk_identity

    def identify_under_the_lease(package_dir, *arguments, **options):
        with pytest.raises(cohort.InstallationBusy):
            cohort.acquire_mutation_lease(site, purpose="a build racing the receipt")
        held.append(str(package_dir))
        return identify(package_dir, *arguments, **options)
    monkeypatch.setattr(cohort, "sdk_identity", identify_under_the_lease)
    receipt = receipts.Receipt("lease first", source_root=repo, package=site / "pops",
                               directory=tmp_path, domain="source-only")
    with receipt:
        receipt.equations(repo / "examples/case.py")
    assert held and receipt.data["lease"]["held"]


def test_receipt_on_a_checkout_package_takes_no_lease_and_writes_nothing_there(repo, tmp_path):
    receipt = receipts.Receipt("editable probe", source_root=repo,
                               package=repo / "python" / "pops", directory=tmp_path)
    with receipt:
        receipt.equations(repo / "examples/case.py")
    data = receipts.read_receipt(receipt.path)
    assert data["lease"] == {"held": False, "installation_root": str(repo / "python"),
                             "skipped": "the SDK is no pip installation (a checkout)"}
    assert not (repo / "python" / ".pops-sdk").exists()
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    findings = data["overlay"]["findings"]
    assert "installation is unmanaged, not a wheel authenticated by RECORD" in findings
    assert "the installation lease was not held" in findings
    assert not receipts.admit(data, None, require="E")["admitted"]


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


def test_every_mapped_sdk_file_must_have_the_bytes_of_the_sdk_the_receipt_names(repo, site):
    package = site / "pops"
    sdk = cohort.sdk_identity(package)
    row = sdk["native"][0]
    leaf = package / "_native" / row["path"]
    data = {"source": cohort.source_identity(repo), "sdk": sdk, "coherence": {"coherent": True},
            "lease": {"held": True}, "process": None, "dsos": [
                {"path": str(leaf), "role": "sdk", "deleted": False, "sha256": row["sha256"]},
                {"path": str(leaf), "role": "sdk", "deleted": False, "sha256": "f" * 64},
                {"path": str(package / "model.py"), "role": "sdk", "deleted": True}]}
    findings = [item for item in receipts.overlay_findings(data, cohort.sdk_file_table(package))[
        "findings"] if item.startswith(("mapped SDK file", "SDK file"))]
    assert findings == [
        "mapped SDK file %s does not have the bytes of the SDK this receipt names"
        % leaf.relative_to(package).as_posix(),
        "SDK file %s is mapped but was deleted or replaced after it was loaded"
        % (package / "model.py")]


def test_a_source_changed_during_the_qualification_is_an_overlay(repo, site, tmp_path):
    receipt = receipts.Receipt("changing source", source_root=repo, package=site / "pops",
                               directory=tmp_path)
    with receipt:
        receipt.equations(repo / "examples/case.py")
        (repo / "tests/data.txt").write_text("edited while qualifying\n")
    data = receipts.read_receipt(receipt.path)
    assert data["source"]["clean"] and not data["process"]["source_at_close"]["clean"]
    assert any(item.startswith("the source changed during the qualification")
               for item in data["overlay"]["findings"])


def test_the_harness_is_judged_on_the_bytes_it_was_imported_from():
    source = cohort.source_identity(ROOT)
    head = receipts.committed_files(source, receipts.HARNESS, contents=receipts._HARNESS_BYTES)
    for path, row in zip(receipts.HARNESS, head, strict=True):
        unchanged = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", str(path)],
            check=False).returncode == 0
        assert row["committed"] is unchanged, row
    tampered = dict(receipts._HARNESS_BYTES)
    tampered[receipts.HARNESS[0]] = b"# not the code that runs\n"
    rows = receipts.committed_files(source, receipts.HARNESS, contents=tampered)
    assert rows[0]["committed"] is False


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
