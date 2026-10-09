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
    def test_closure_free_authored_arithmetic_accepts_permuted_storage(self):
        spec = importlib.util.spec_from_file_location(
            "independent_moment_path_emitter", ROOT / FILES[0])
        emitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emitter)
        basis = tuple((p, q) for q in range(3) for p in range(3-q))
        old_recipe = dict(order=2, closure_order=2, indices=basis,
                          bound_terms=(6, 10), regularization=())
        with self.assertRaises(ValueError):
            emitter.emit_moment_path_kernel(old_recipe, "RefusedRecipe")
        nodes = tuple(dict(op="literal", args=(value,), degree=0, polynomial=False) for value in (0, 1))
        plan = dict(schema_version=1, order=2, polynomial_degree=0, indices=tuple(reversed(basis)), nodes=nodes,
                    flux=(0,)*6, integrands=(0,)*6, factors=(0,)*6, speed=1,
                    integral_mode="density_weighted_polynomial", integral=None)
        emitted = "\n".join(emitter.emit_moment_path_kernel(plan, "IndependentWitness"))
        self.assertNotIn("hermite", emitted)
        self.assertIn("slots[] = {5, 4, 3, 2, 1, 0}", emitted)
        self.assertFalse(any(name == "_pops" or name.endswith("._pops") for name in sys.modules))
        print(json.dumps({"status": "SOURCE_ONLY", "checkout": str(ROOT),
                          "python_path": str(ROOT / "python"), "native_loaded": False,
                          "source_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                          "central_closure": False, "permuted_basis": "accepted",
                          "inspected_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}}, sort_keys=True))

if __name__ == "__main__":
    unittest.main()
