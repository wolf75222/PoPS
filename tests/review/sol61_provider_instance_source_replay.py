"""Pure Source emission replay; baseline modules are loaded from the actual Git base.

No PoPS native extension is loaded. The baseline hook replaces modified production
Python modules only; fixture/modules outside that delta remain byte-identical.
"""
from pathlib import Path
import importlib.abc
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = 'b0a9409fa412575a0e5bd47b26705e6afe88f1a6'


class Baseline(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        paths = subprocess.check_output(['git', 'diff', '--name-only', BASE, '--', 'python'], cwd=ROOT, text=True).splitlines()
        original = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, '--', 'python'], cwd=ROOT, text=True).splitlines())
        self.sources = {p[7:-3].replace('/', '.'): subprocess.check_output(['git', 'show', BASE + ':' + p], cwd=ROOT)
                        for p in paths if p.endswith('.py') and p in original}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(ROOT / 'python' / (module.__name__.replace('.', '/') + '.py'))
        exec(compile(self.sources[module.__name__], module.__file__, 'exec'), module.__dict__)


def main():
    mode, output = sys.argv[1:]
    sys.path.insert(0, str(ROOT / 'python'))
    sys.path.insert(1, str(ROOT))
    if mode.startswith('baseline'):
        sys.meta_path.insert(0, Baseline())
    import pops
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen._compiler_lowering import require_compiler_lowering
    if mode.endswith('negative'):
        from tests.python.support.field_publication_instance_case import build
        build()
        raise AssertionError('the real baseline cardinality refusal did not occur')
    if mode.endswith('legacy'):
        from tests.python.unit.fields.test_field_publication import publication_case
        case, layout, *_ = publication_case()
        resolved = pops.resolve(pops.validate(case), layout=layout)
    else:
        from tests.python.support.field_publication_instance_case import build
        resolved = build(different_values=mode.endswith('joint'), first_beta=mode.endswith('joint'))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    (out / 'program.cpp').write_text(emit_cpp_program(resolved.time, model_graph=graph))
    for index, block in enumerate(resolved.blocks):
        emitter = graph.model_for_block(block.name)
        (out / f'model-{index}.cpp').write_text(require_compiler_lowering(emitter).native_loader_source(
            name=f'Witness_{index}', consumer_owner_qid=block.instance_owner_qid,
            declare_auxiliary_providers=block.declares_auxiliary_providers))
    loaded = [name for name,module in tuple(sys.modules.items()) if name.startswith('pops.')
              and Path(getattr(module,'__file__','') or '').suffix in ('.so','.dylib','.pyd')]
    if loaded:
        raise RuntimeError('Source replay loaded a PoPS native extension: '+repr(loaded))
    print(mode, len(resolved.blocks), 'real Program/Model emission complete')


if __name__ == '__main__':
    main()
