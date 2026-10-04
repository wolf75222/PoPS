"""Actual parent Git Python emission, no installation or PoPS Native loading."""
from pathlib import Path
import importlib.abc
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = "d1b293215b91a2ac57dfe6fb2c44235a3ba1372f"


class Parent(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self.sources = {"pops.codegen." + name: subprocess.check_output([
            "git", "show", BASE + ":python/pops/codegen/" + name + ".py"], cwd=ROOT)
            for name in ("cpp_symbols", "program_emit_ops")}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(ROOT / "python" / (module.__name__.replace(".", "/") + ".py"))
        exec(compile(self.sources[module.__name__], module.__file__, "exec"), module.__dict__)


def main():
    mode, output = sys.argv[1:]
    sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
    if mode.startswith("parent"):
        sys.meta_path.insert(0, Parent())
    import pops
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen._compiler_lowering import require_compiler_lowering
    from tests.python.support.public_coupled_block_name_case import build
    from tests.python.support.local_residual_product_case import make_case
    profiles = []
    if mode.endswith("legacy"):
        profiles.append(("explicit", build(("first", "second"))))
        case, layout, _ = make_case()
        profiles.append(("implicit", pops.resolve(pops.validate(case), layout=layout)))
    elif mode.endswith("explicit-red"):
        profiles.append(("explicit", build()))
    elif mode.endswith("implicit-red"):
        case, layout, _ = make_case(block_names=("a-b", "a_b"))
        profiles.append(("implicit", pops.resolve(pops.validate(case), layout=layout)))
    else:
        raise ValueError("unknown replay mode")
    out = Path(output)
    out.mkdir(exist_ok=False)
    for label, resolved in profiles:
        directory = out / label
        directory.mkdir()
        graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
        (directory / "program.cpp").write_text(emit_cpp_program(resolved.time, model_graph=graph))
        for i, block in enumerate(resolved.blocks):
            (directory / ("model-%d.cpp" % i)).write_text(require_compiler_lowering(
                graph.model_for_block(block.name)).native_loader_source(name="ScratchWitness%d" % i,
                consumer_owner_qid=block.instance_owner_qid,
                declare_auxiliary_providers=block.declares_auxiliary_providers))
    assert not any(name.startswith("pops.") and
        Path(getattr(module, "__file__", "") or "").suffix in (".so", ".dylib", ".pyd")
        for name, module in tuple(sys.modules.items()))
    print(mode, len(profiles), "actual Source profiles emitted without PoPS Native")


if __name__ == "__main__":
    main()
