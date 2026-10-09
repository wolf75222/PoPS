"""SOURCE_ONLY lifecycle reception. No PoPS import, CompiledModel or native execution.

Execute actual installer statement slices; substitute only the external authority,
configuration and native lifecycle seams by explicitly named host adapters. This
receives ordering/argument identity, not DSO/package/ExecutionContext authentication.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BASE = "dbc9e1cd725c894f615b76a459d47986a8febd03"
EXECUTOR = "python/pops/runtime/_runtime_executor.py"
INSTALL = "python/pops/runtime/_amr_system_install.py"


def method(source, name):
    return next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def run_nodes(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<actual-source-installer-slice>", "exec"), namespace)


class NativeLifecycleHostAdapter:
    def __init__(self, config):
        self.config = config
        self._s = self
        self._execution_context = None
        self.lane = False
        self.blocks = []
        self.geometry = False
        self.hierarchy = False
        self.events = []

    def add_equation(self, name, model, **kwargs):
        if not self.lane:
            raise RuntimeError("package install requires its authentic staged lane")
        self.blocks.append((name, model))
        self.events.append("block")

    def embedded(self, normalized):
        if not self.lane:
            raise RuntimeError("AMR native package requires a pre-staged RuntimeInstance lane")
        if not self.blocks:
            raise RuntimeError("AmrSystem embedded geometry requires an installed exact generated block provider")
        if self.hierarchy:
            raise RuntimeError("AmrSystem embedded geometry must be authored before hierarchy materialization")
        self.geometry = True
        self.events.append("embedded")

    def _finish_program_install(self, *args):
        if not self.geometry:
            raise RuntimeError("embedded geometry was not bound before materialization")
        self.hierarchy = True
        self.events.append("materialize")


class LifecycleReception(unittest.TestCase):
    def setUp(self):
        self.layout = SimpleNamespace(native_spatial_layout=object(), transition_ratios=(2,))
        self.context = object()  # Opaque already-authenticated caller resource; no forged native handle.
        self.plan = SimpleNamespace(layout=object(), resolved_hierarchy=object(),
            initial_condition_plan=object(), bootstrap_plan=object(), execution_context=self.context,
            artifact=SimpleNamespace(program=object(), layout_plan=SimpleNamespace(layouts=(self.layout,))))
        self.created = []

    def prefix(self, source, *, authority_failure=False):
        node = method(source, "_install_adaptive_native_engine")
        # Stop after real lane/runtime-authorities staging, before unrelated bootstrap argument lowering.
        stop = next(i for i, stmt in enumerate(node.body)
                    if isinstance(stmt, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "schema" for target in stmt.targets))
        body = node.body[:stop]
        fn = ast.FunctionDef(name="prefix", args=node.args, body=body + [ast.Return(ast.Name("engine", ast.Load()))],
                             decorator_list=[])

        def authenticate(plan):
            self.assertIs(plan, self.plan)
            return plan

        def config(layout, *, hierarchy, native_layout):
            self.assertIs(layout, self.plan.layout)
            self.assertIs(hierarchy, self.plan.resolved_hierarchy)
            self.assertIs(native_layout, self.layout.native_spatial_layout)
            return object()

        def create(cfg):
            engine = NativeLifecycleHostAdapter(cfg)
            self.created.append(engine)
            return engine

        def authorities(engine, plan):
            self.assertIs(plan, self.plan)
            self.assertIs(engine._execution_context, self.context)
            if authority_failure:
                raise ValueError("named host authority refusal")
            engine.lane = True
            engine.events.append("lane")

        def embedded(engine, layout):
            self.assertIs(layout, self.layout)
            engine.embedded(layout)

        adapters = {
            "pops.runtime._layout_install_projection": SimpleNamespace(require_install_authority=authenticate),
            "pops.runtime._amr_bind_lowering": SimpleNamespace(amr_config_from_layout=config),
            "pops.runtime._system": SimpleNamespace(AmrSystem=create),
            "pops.runtime._checkpoint_spatial": SimpleNamespace(install_checkpoint_spatial_contract=lambda *_a, **_k: None),
            "pops.runtime._runtime_mesh_lowering": SimpleNamespace(install_embedded_boundary=embedded),
            "pops.runtime._runtime_authorities": SimpleNamespace(install_runtime_authorities=authorities),
        }
        namespace = {"_require_native_geometry": lambda plan: self.assertIs(plan, self.plan)}
        run_nodes([fn], namespace)
        with patch.dict(sys.modules, adapters):
            engine = namespace["prefix"](self.plan)
        return engine, adapters

    def stage_blocks_geometry_and_materialize(self, engine, adapters):
        node = method((ROOT / INSTALL).read_text(), "_install_compiled")
        def calls(stmt, attribute):
            return any(isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                       and v.func.attr == attribute for v in ast.walk(stmt))
        selected = [stmt for stmt in node.body if calls(stmt, "add_equation")
                    or (isinstance(stmt, ast.If) and any(
                        isinstance(v, ast.Name) and v.id == "install_embedded_boundary" for v in ast.walk(stmt)))
                    or calls(stmt, "_finish_program_install")]
        # No fake CompiledModel/package is created. Opaque provider tokens are only handed through
        # the actual call-site sequence to the named native-lifecycle adapter.
        token = object()
        namespace = dict(self=engine, install_plan=self.plan, lowered_instances={"rho": (token, None, None)},
                         per_block_params={}, compiled=None, so_path=None, bind_schema=None, params={})
        with patch.dict(sys.modules, adapters):
            run_nodes(selected, namespace)
        self.assertIs(engine.blocks[0][1], token)

    def test_parent_exact_source_exhibits_original_native_lane_refusal(self):
        source = subprocess.check_output(["git", "show", f"{BASE}:{EXECUTOR}"], cwd=ROOT, text=True)
        with self.assertRaisesRegex(RuntimeError, "^AMR native package requires a pre-staged RuntimeInstance lane$"):
            self.prefix(source)
        self.assertEqual(self.created[0].events, [])

    def test_candidate_stages_lane_then_blocks_geometry_and_materialization(self):
        engine, adapters = self.prefix((ROOT / EXECUTOR).read_text())
        self.stage_blocks_geometry_and_materialize(engine, adapters)
        self.assertEqual(engine.events, ["lane", "block", "embedded", "materialize"])
        self.assertIs(engine._execution_context, self.context)

    def test_authority_failure_precedes_any_block_or_embedded_consumer(self):
        with self.assertRaisesRegex(ValueError, "^named host authority refusal$"):
            self.prefix((ROOT / EXECUTOR).read_text(), authority_failure=True)
        self.assertEqual(self.created[0].events, [])

    def test_multiple_native_layouts_are_not_inferred_as_one(self):
        engine, adapters = self.prefix((ROOT / EXECUTOR).read_text())
        self.plan.artifact.layout_plan.layouts = (self.layout, self.layout)
        with self.assertRaises(ValueError):
            self.stage_blocks_geometry_and_materialize(engine, adapters)
        self.assertEqual(engine.events, ["lane", "block"])

    def test_embedded_requires_blocks_and_unmaterialized_carrier(self):
        engine = NativeLifecycleHostAdapter(None)
        engine.lane = True
        with self.assertRaisesRegex(RuntimeError, "installed exact generated block provider"):
            engine.embedded(self.layout)
        engine.add_equation("rho", object())
        engine.hierarchy = True
        with self.assertRaisesRegex(RuntimeError, "before hierarchy materialization"):
            engine.embedded(self.layout)

    def test_host_lifecycle_conditions_are_real_current_native_guards(self):
        text = (ROOT / "src/runtime/amr/amr_system.cpp").read_text()
        start = text.index("void AmrSystem<Dim>::set_analytic_level_set(")
        end = text.index("template <int Dim>", start + 1)
        source = text[start:end]
        self.assertLess(source.index("require_package_assembly_lane()"), source.index("prepared_blocks.empty()"))
        self.assertLess(source.index("prepared_blocks.empty()"), source.index("p_->engine || p_->prepared_hierarchy"))
        self.assertIn("all_reduce_max(local_failure, lane)", source)
        self.assertIn("all_ranks_agree_exact_ordered_byte_pairs", source)


if __name__ == "__main__":
    unittest.main()
