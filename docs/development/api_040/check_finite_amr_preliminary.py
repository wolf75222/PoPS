"""Verify the immutable preliminary finite/AMR archive, including failed runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> None:
    root = Path(__file__).resolve().parent / "evidence/finite-amr-preliminary"
    manifest = json.loads((root / "manifest.json").read_text())
    inventory = json.loads((root / "inventory.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == record["bytes"], name
        assert digest(path) == record["sha256"], name
    checksum_names = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert name not in checksum_names, name
        checksum_names.add(name)
        assert digest(root / name) == expected, name
    assert checksum_names == set(manifest["files"]) | {"manifest.json"}
    assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == (
        checksum_names | {"SHA256SUMS"})
    outcomes = {
        "installed-finite-amr-units": (119, 1, 118),
        "installed-amr-flat-authority-repaired": (13, 0, 13),
        "installed-finite-amr-native-serial": (9, 7, 2),
    }
    source_digests = set()
    native_digests = set()
    for name, (total, failed, passed) in outcomes.items():
        directory = root / name
        identity = json.loads((directory / "identity.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        assert digest(directory / "identity.json") == result["identity_sha256"]
        assert digest(directory / "pytest.log") == result["log_sha256"]
        assert digest(directory / "source-files.json") == identity["source_files_sha256"]
        assert len(json.loads((directory / "source-files.json").read_text())) == 1089
        assert identity["verified_source_files"] == 1089
        cases = ET.parse(directory / "pytest.xml").findall(".//testcase")
        counts = {"tests": len(cases),
                  "failures": sum(c.find("failure") is not None for c in cases),
                  "errors": sum(c.find("error") is not None for c in cases),
                  "skipped": sum(c.find("skipped") is not None for c in cases)}
        assert counts == result["counts"] == inventory[name]["counts"]
        assert counts == {"tests": total, "failures": failed, "errors": 0, "skipped": 0}
        assert total - failed == passed == inventory[name]["passed"]
        assert result["returncode"] == (1 if failed else 0)
        assert result["status"] == ("failed" if failed else "passed")
        assert [c.attrib["name"] for c in cases] == [c["name"] for c in inventory[name]["cases"]]
        assert inventory[name]["source_commit"] == identity["source_commit"]
        assert inventory[name]["native_sha256"] == identity["native_sha256"]
        assert manifest["verified_git_source_snapshots"][identity["source_commit"]][
            "verified_git_blob_count"] == 1089
        source_digests.add(identity["source_files_sha256"])
        native_digests.add(identity["native_sha256"])
        assert ";dim=2" in identity["abi_key"]
        assert "headers=396719235acdb96435dba0dd27c5e5579aefbda4936f657262b9b2553d13bb4c;" in identity["abi_key"]
    assert source_digests == {"6c432f8e7005a26400f852e24d3f11b9443eeac0d79cac2985fc549f869c13fc"}
    assert native_digests == {"aed8c1582c3344bd5afb19ab784adec260bcdbd4bf944b26a29242b53889c862"}
    failed_sources = list((root / "failed-cpp").rglob("*.failed.cpp"))
    assert len(failed_sources) == manifest["failed_cpp_count"] == 6
    assert all("expression_active_" in p.read_text() for p in failed_sources)
    print("finite/AMR preliminary evidence verified: 118/119, 13/13, 2/9; six failed C++ sources retained")


if __name__ == "__main__":
    verify()
