"""Independent tool-only counter tests; no PoPS import, worker or native campaign.

Set POPS_RESOURCE_PROBE to the companion script being reviewed.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ResourceProbeReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["POPS_RESOURCE_PROBE"]).resolve()
        spec = importlib.util.spec_from_file_location("resource_probe_under_review", path)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def rows(self, left, right):
        return {
            lane: {"samples": [{"selected_counters": self.probe._selected_counters(counters)}
                               for _ in range(self.probe.PROFILED_RUNS)]}
            for lane, counters in (("baseline", left), ("candidate", right))
        }

    def test_missing_is_not_a_measured_zero(self):
        absent = self.probe._counter({}, "scratch_allocs")
        present = self.probe._counter({"scratch_allocs": 0}, "scratch_allocs")
        self.assertIs(absent["available"], False)
        self.assertIsNone(absent["value"])
        self.assertIs(present["available"], True)
        self.assertEqual(present["value"], 0)

    def test_partial_memory_and_kernel_names_are_not_inferred(self):
        observed = self.probe._selected_counters({"scratch_allocs": 3, "kernels": 7})
        self.assertEqual(observed["scratch_allocs"]["value"], 3)
        self.assertIs(observed["scratch_peak_bytes"]["available"], False)
        self.assertEqual(observed["kernels"]["value"], 7)
        self.assertIs(observed["kernel_launches"]["available"], False)
        self.assertIs(observed["mpi_messages"]["available"], False)

    def test_largest_buffer_is_not_total_memory(self):
        observed = self.probe._counter({"scratch_peak_bytes": 4096}, "scratch_peak_bytes")
        self.assertEqual(observed["value"], 4096)
        self.assertEqual(observed["unit"], "bytes_largest_single_scratch_buffer")

    def test_noninteger_and_negative_counters_refuse(self):
        for value in (True, False, -1, 1.5, "2", None, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.probe._counter({"kernels": value}, "kernels")

    def test_symmetric_explicit_zeros_remain_measured(self):
        result = self.probe._aggregate(self.rows({"kernels": 0}, {"kernels": 0}))
        self.assertIs(result["kernels"]["available"], True)
        self.assertEqual(result["kernels"]["values"],
                         {lane: [0] * self.probe.PROFILED_RUNS
                          for lane in ("baseline", "candidate")})
        self.assertIs(result["scratch_peak_bytes"]["available"], False)

    def test_missing_lane_counter_prevents_comparison(self):
        result = self.probe._aggregate(self.rows({"scratch_allocs": 9}, {}))
        self.assertIs(result["scratch_allocs"]["available"], False)
        self.assertEqual(result["scratch_allocs"]["observed_values"],
                         {"baseline": [9] * self.probe.PROFILED_RUNS})

    def test_one_missing_sample_prevents_comparison(self):
        rows = self.rows({"kernels": 9}, {"kernels": 10})
        rows["candidate"]["samples"][1]["selected_counters"] = self.probe._selected_counters({})
        result = self.probe._aggregate(rows)
        self.assertIs(result["kernels"]["available"], False)

    def test_frozen_physics_script_change_refused_before_import(self):
        with patch.object(self.probe, "_sha", return_value="different"):
            with self.assertRaisesRegex(ValueError, "frozen v1"):
                self.probe._frozen_v1()

    def test_wrong_candidate_identity_refused(self):
        identity = {
            "source_commit": "stale",
            "native_sha256": self.probe.CANDIDATE_NATIVE,
            "source_files_sha256": self.probe.CANDIDATE_SOURCES,
            "verified_source_files": 1080,
        }
        helper = SimpleNamespace(_load_identity=lambda root: identity)
        with self.assertRaisesRegex(ValueError, "authenticated snapshot"):
            self.probe._identity(helper, Path("unused"), "candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
