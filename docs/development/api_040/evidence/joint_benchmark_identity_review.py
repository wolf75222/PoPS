"""Independent snapshot-refusal tests. No PoPS import, build, or benchmark run.

Run with POPS_BENCHMARK_SCRIPT pointing to the reviewed benchmark script.
The tiny snapshots are fixtures for filesystem authentication, not native artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SnapshotIdentityReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["POPS_BENCHMARK_SCRIPT"]).resolve()
        spec = importlib.util.spec_from_file_location("benchmark_under_review", path)
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.root = self.parent / "snapshot"
        (self.root / "pops/_native/dim2").mkdir(parents=True)
        self.source = self.root / "pops/__init__.py"
        self.source.write_text("# fixture only\n")
        self.native = self.root / "pops/_native/dim2/_pops.fixture.so"
        self.native.write_bytes(b"not an executable: authentication fixture")
        self.members = {"python/pops/__init__.py": digest(self.source)}
        self.write_receipt()

    def write_receipt(self):
        manifest = self.root / "source-files.json"
        manifest.write_text(json.dumps(self.members))
        identity = {
            "schema_version": 1, "verified_source_files": len(self.members),
            "source_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "source_files_sha256": digest(manifest),
            "native_sha256": digest(self.native),
        }
        (self.root / "identity.json").write_text(json.dumps(identity))

    def test_unmodified_fixture_reaches_identity_gate(self):
        identity = self.bench._load_identity(self.root)
        self.assertEqual(identity["native_sha256"], digest(self.native))

    def test_changed_source_refused(self):
        self.source.write_text("# altered after receipt\n")
        with self.assertRaises(ValueError):
            self.bench._load_identity(self.root)

    def test_changed_native_refused(self):
        self.native.write_bytes(b"different fixture")
        with self.assertRaises(ValueError):
            self.bench._load_identity(self.root)

    def test_changed_manifest_refused(self):
        with (self.root / "source-files.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            self.bench._load_identity(self.root)

    def test_parent_escape_refused_even_with_matching_digest(self):
        outside = self.parent / "outside.py"
        outside.write_text("# outside snapshot\n")
        self.members["python/pops/../../outside.py"] = digest(outside)
        self.write_receipt()
        with self.assertRaises(ValueError):
            self.bench._load_identity(self.root)

    def test_symlink_escape_refused_even_with_matching_digest(self):
        outside = self.parent / "outside.py"
        outside.write_text("# outside snapshot\n")
        (self.root / "pops/linked.py").symlink_to(outside)
        self.members["python/pops/linked.py"] = digest(outside)
        self.write_receipt()
        with self.assertRaises(ValueError):
            self.bench._load_identity(self.root)

    def driver_fixture(self):
        baseline = {
            "source_commit": "3d06cabee9db4a31c4d07fa4e00dc5a40d164155",
            "native_sha256": "4574ed6096650aa708ad040fd6ee056414a3cfa1735a0440bd74220170265beb",
            "source_files_sha256": "2a76043ebff3bb5405d3408b39867856b9dd635d67a72ca9040d6535794300ce",
            "verified_source_files": 1079,
        }
        candidate = dict(baseline, source_commit="candidate-expected")
        args = SimpleNamespace(
            baseline_root=self.root, candidate_root=self.root,
            expected_candidate_commit="candidate-expected",
            output=self.parent / "results", execute=False,
        )
        return baseline, candidate, args

    def test_candidate_revision_is_independent_of_snapshot_receipt(self):
        baseline, candidate, args = self.driver_fixture()
        candidate["source_commit"] = "stale-candidate"
        with patch.object(self.bench, "_load_identity", side_effect=[baseline, candidate]):
            with self.assertRaises(ValueError):
                self.bench._driver(args)
        self.assertFalse(args.output.exists())

    def test_baseline_manifest_is_pinned_outside_its_receipt(self):
        baseline, candidate, args = self.driver_fixture()
        baseline["source_files_sha256"] = "replaced-manifest"
        with patch.object(self.bench, "_load_identity", side_effect=[baseline, candidate]):
            with self.assertRaises(ValueError):
                self.bench._driver(args)
        self.assertFalse(args.output.exists())

    def test_existing_receipt_is_not_overwritten(self):
        baseline, candidate, args = self.driver_fixture()
        args.output.mkdir()
        receipt = args.output / "plan.json"
        receipt.write_text("prior result must survive\n")
        with patch.object(self.bench, "_load_identity", side_effect=[baseline, candidate]):
            with self.assertRaises(ValueError):
                self.bench._driver(args)
        self.assertEqual(receipt.read_text(), "prior result must survive\n")

    def test_unsupported_host_refused_before_loading_packages(self):
        with patch.object(self.bench.sys, "platform", "linux"):
            with patch.object(self.bench, "_load_identity") as loader:
                with self.assertRaises(ValueError):
                    self.bench._driver(SimpleNamespace())
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
