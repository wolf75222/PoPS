"""Independent source-only architectural witness; no PoPS/native import."""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FILES = (
    "python/pops/codegen/moment_path_kernel.py",
    "python/pops/codegen/module_emit_path.py",
    "python/pops/moments/fan_li.py",
    "python/pops/time/_program/affine_moments.py",
    "python/pops/codegen/program_emit_affine_moments.py",
    "python/pops/moments/closures/discrete_entropy.py",
    "python/pops/mesh/physical_mapping.py",
    "python/pops/mesh/native_physical_mapping.py",
    "include/pops/runtime/dynamic/physical_support_transfer.hpp",
    "python/pops/moments/closures/hyqmom15.py",
    "python/pops/moments/model_builder.py",
)

class GenericCoreCompositionWitness(unittest.TestCase):
    def test_equivalent_permuted_basis_is_refused_and_closure_is_selected_centrally(self):
        # Bypass package initialization deliberately: this witnesses authoring,
        # never a compiled module, a numerical result or backend availability.
        sys.path.insert(0, str(ROOT / "python"))
        spec = importlib.util.spec_from_file_location(
            "independent_moment_path_emitter", ROOT / FILES[0])
        emitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emitter)
        basis = tuple((p, q) for q in range(3) for p in range(3-q))
        plan = dict(order=2, closure_order=2, indices=basis,
                    bound_terms=(6, 10), regularization=())
        emitted = "\n".join(emitter.emit_moment_path_kernel(plan, "IndependentWitness"))
        self.assertIn("hermite_raw_edge<2, 3>", emitted)
        permuted = tuple(reversed(basis))
        self.assertEqual(set(permuted), set(basis))
        plan["indices"] = permuted
        with self.assertRaises(ValueError) as refusal:
            emitter.emit_moment_path_kernel(plan, "IndependentWitness")
        self.assertIn("exact q-outer raw moment ordering", str(refusal.exception))
        self.assertFalse(any(name == "_pops" or name.endswith("._pops") for name in sys.modules))
        print(json.dumps({
            "status": "SOURCE_ONLY",
            "checkout": str(ROOT), "python_path": str(ROOT / "python"),
            "python_executable": sys.executable,
            "source_head": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "native_loaded": False,
            "central_closure": "hermite_raw_edge<2, 3>",
            "equivalent_permuted_basis_refusal": str(refusal.exception),
            "inspected_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                  for name in FILES},
        }, sort_keys=True))

if __name__ == "__main__":
    unittest.main()
