"""Spec 4: the intra-pops import graph is acyclic and respects the layering.

The sub-packages form a directed acyclic dependency stack:

    _ir       -> identity                        (canonical scalar identity codec)
    identity  imports nothing else in pops
    frames    -> identity
    analytic  -> frames                         (data-only coordinate expressions)
    domain    -> frames, identity
    model     -> _ir, identity, params
    problem   -> _ir, identity, model
    physics   -> _ir, identity, model, problem
    time      -> _ir, identity, model, params
    initial   -> identity, model                 (layout-plan consumer protocol)
    mesh      -> _ir, analytic, domain, frames, identity, model, params
    amr       -> _ir, identity, mesh, model, time
    layouts   -> amr, mesh
    boundary  -> _ir, analytic, domain, identity, model, representations
    numerics  -> _ir, identity, model, params
    linalg    -> (nothing)                       (Spec 5: abstract algebra descriptors)
    solvers   -> identity                        (typed solver descriptor sink)
    moments   -> _ir                             (Spec 5: moment-model toolkit)
    diagnostics -> linalg                        (Spec 5: Norm takes a typed norm kind)
    params    -> (nothing)                       (typed parameter dependency sink)
    output    -> model, time                     (qualified selections and schedules)
    external  -> model                           (authenticated component manifests)
    lib       -> identity, frames, time, physics, moments, fields, params, solvers
    codegen   -> _ir, model, physics, time, lib, solvers, params,
                 external, fields
    runtime   -> authoring/lowering contracts, including resolved fields

This test builds the import-time cross-layer edges from module-scope imports (``ast``,
``col_offset == 0``) between sub-packages and asserts (a) the graph has no cycle and
(b) every edge points to an allowed lower layer. The flat root files and
``pops/__init__.py`` (the exact public lifecycle facade) are not layered sub-packages and are
excluded from the graph.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# Allowed downstream targets for each layer (what it MAY import within pops).
ALLOWED = {
    "_ir": {"identity"},
    "identity": set(),
    "representations": set(),
    "spaces": set(),
    "projection": set(),
    "params": set(),
    "linalg": set(),
    "frames": {"identity"},
    "analytic": {"frames"},
    "domain": {"frames", "identity"},
    "model": {"_ir", "identity", "params"},
    "problem": {"_ir", "identity", "model"},
    "physics": {"_ir", "identity", "model", "problem"},
    "time": {"_ir", "identity", "model", "params"},
    "initial": {"identity", "model"},
    # Physical maps consume the same immutable support/unit leaves as their quantity ports.
    # The IR remains a lower layer; compiler and runtime dependencies are still forbidden.
    "mesh": {"_ir", "analytic", "domain", "frames", "identity", "model", "params"},
    "amr": {"_ir", "identity", "mesh", "model", "time"},
    "layouts": {"amr", "mesh"},
    "boundary": {"_ir", "analytic", "domain", "identity", "model", "representations"},
    "numerics": {"_ir", "identity", "model", "params"},
    "solvers": {"identity"},
    "fields": {"_ir", "identity", "model", "time"},
    "moments": {"_ir"},
    "diagnostics": {"linalg"},
    "output": {"identity", "model", "time"},
    "external": {"identity", "model"},
    # Ready implementations may mint canonical semantic identities, but identity is a strict sink:
    # this edge cannot introduce a cycle or pull compiler/runtime authority into pops.lib.
    "lib": {"fields", "frames", "identity", "moments", "params", "physics", "solvers", "time"},
    "codegen": {"_ir", "fields", "identity", "model", "params", "solvers", "time"},
    "runtime": {"_ir", "codegen", "fields", "identity", "mesh", "model", "output", "time"},
}
LAYERS = set(ALLOWED)

NATIVE_SELECTOR_CONSUMERS = frozenset({
    "pops._native_collectives",
    "pops._paraview_python_bootstrap",
    "pops._platform_contracts",
    "pops.codegen._compiled_artifact",
    "pops.codegen._checkpoint_migration_uniform_v2",
    "pops.external.artifacts",
    "pops.external.compiler",
    "pops.output._writers.hdf5",
    "pops.runtime._amr_package_lane",
    "pops.runtime._platform_manifest",
    "pops.runtime._runtime_authorities",
    "pops.runtime._threading",
    "pops.runtime.doctor",
    "pops.runtime.fallbacks",
})


def _layer_of(modname):
    """Return the sub-package layer for a dotted ``pops.<layer>...`` name, else None."""
    parts = modname.split(".")
    if len(parts) >= 2 and parts[0] == "pops" and parts[1] in LAYERS:
        return parts[1]
    return None


def _module_name(path):
    rel = path.relative_to(POPS.parent).with_suffix("")
    return ".".join(rel.parts)


def _intra_targets(tree):
    """Yield module-scope (col_offset==0) import targets that name some pops module."""
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pops" or alias.name.startswith("pops."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and (
                node.module == "pops" or node.module.startswith("pops.")
            ):
                yield node.module


def _source_paths():
    """Yield importable source modules, excluding editor/cache copy artifacts.

    Local synchronization tools can leave untracked names such as ``module 2.py`` beside the real
    source. Those files are not Python modules and must not change an architecture result. A valid
    untracked module is still scanned, while ``test_file_sizes.py`` separately refuses a
    non-importable path if it is ever committed.
    """
    for path in sorted(POPS.rglob("*.py")):
        module_parts = path.relative_to(POPS).with_suffix("").parts
        if all(part.isidentifier() for part in module_parts):
            yield path


def test_layer_map_covers_every_top_level_package():
    actual = {
        path.name for path in POPS.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert LAYERS == actual, "layer map drift: missing=%s extra=%s" % (
        sorted(actual - LAYERS), sorted(LAYERS - actual))


def _native_import_lines(tree):
    """Yield every direct or importlib native-extension load at any lexical scope."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name in {"_pops", "pops._pops"} for alias in node.names):
                yield node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"_pops", "pops._pops"} or any(
                    alias.name == "_pops" and (module == "pops" or node.level)
                    for alias in node.names):
                yield node.lineno
        elif isinstance(node, ast.Call) and node.args \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "import_module" \
                and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value in {"_pops", "pops._pops"}:
            yield node.lineno


def test_no_module_bypasses_the_native_selector_with_a_direct_extension_import():
    violations = []
    for path in _source_paths():
        module = _module_name(path)
        lines = tuple(_native_import_lines(ast.parse(path.read_text(), str(path))))
        if lines:
            violations.append("%s:%s" % (module, ",".join(map(str, lines))))
    assert not violations, (
        "direct native-extension load bypasses pops._native_selector: " + ", ".join(violations)
    )


def test_native_consumers_import_the_process_selector_explicitly():
    observed = set()
    for path in _source_paths():
        module = _module_name(path)
        tree = ast.parse(path.read_text(), str(path))
        if any(
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "pops._native_selector"
            for node in ast.walk(tree)
        ):
            observed.add(module)
    missing = sorted(NATIVE_SELECTOR_CONSUMERS - observed)
    assert not missing, "native consumer(s) bypass or lost the process selector: " + ", ".join(missing)


def _build_edges():
    """Return {src_layer: {(dst_layer, "src_module -> dst_target"), ...}}."""
    edges = {}
    for path in _source_paths():
        src_layer = _layer_of(_module_name(path))
        if src_layer is None:
            continue  # root facade / flat files are not layered sub-packages.
        tree = ast.parse(path.read_text(), str(path))
        for target in _intra_targets(tree):
            dst_layer = _layer_of(target)
            if dst_layer is None or dst_layer == src_layer:
                continue
            why = "%s -> %s" % (_module_name(path), target)
            edges.setdefault(src_layer, set()).add((dst_layer, why))
    return edges


def test_layering_respected():
    edges = _build_edges()
    violations = []
    for src_layer, deps in edges.items():
        for dst_layer, why in sorted(deps):
            if dst_layer not in ALLOWED[src_layer]:
                violations.append("%s may not import %s (%s)" % (src_layer, dst_layer, why))
    assert not violations, "layering violations:\n  " + "\n  ".join(sorted(violations))


def test_graph_is_acyclic():
    edges = _build_edges()
    adjacency = {layer: {d for d, _ in deps} for layer, deps in edges.items()}

    # Iterative DFS with three-color marking; record the back-edge that closes a cycle.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {layer: WHITE for layer in LAYERS}
    cycle_edge = []

    def visit(start):
        stack = [(start, iter(sorted(adjacency.get(start, ()))))]
        color[start] = GRAY
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if color[child] == GRAY:
                    cycle_edge.append("%s -> %s" % (node, child))
                    return True
                if color[child] == WHITE:
                    color[child] = GRAY
                    stack.append((child, iter(sorted(adjacency.get(child, ())))))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
        return False

    for layer in sorted(LAYERS):
        if color[layer] == WHITE and visit(layer):
            break
    assert not cycle_edge, "import cycle through edge(s): " + ", ".join(cycle_edge)


def test_params_remains_a_dependency_sink():
    """ADC-654 consumers may depend on params; params must never depend back on them."""
    dependencies = sorted(dst for dst, _ in _build_edges().get("params", set()))
    assert not dependencies, (
        "pops.params is the central ParamKind x ParamUse sink and must have no layered "
        "module-scope dependencies; got %s" % dependencies)


def test_internal_ir_remains_a_dependency_sink():
    """The IR depends only on the foundational canonical scalar identity codec."""
    dependencies = {dst for dst, _ in _build_edges().get("_ir", set())}
    assert dependencies == {"identity"}, (
        "pops._ir may depend only on pops.identity canonical scalars; got %s"
        % sorted(dependencies))


def test_solver_catalog_remains_a_dependency_sink():
    """The inert descriptor catalog may depend only on foundational exact identities."""
    dependencies = {dst for dst, _ in _build_edges().get("solvers", set())}
    assert dependencies == {"identity"}, (
        "pops.solvers must depend exactly on pops.identity and no other layer; got %s"
        % sorted(dependencies))


# A registered compatibility class alias is API routing, not an imperative layer edge.
# ALLOWED above still governs every ordinary module-scope import unchanged.
def _public_alias_route_errors(module, tree, aliases):
    registered = tuple(alias for alias in aliases.values() if alias.public_module == module)
    hooks = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "__getattr__"]
    if not hooks:
        return ["registered alias has no route"] if registered else []
    if len(hooks) != 1:
        return ["duplicate lazy export hooks"]
    hook = hooks[0]
    body = hook.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    rejection = ast.parse("raise AttributeError(name)").body
    if not registered and ast.dump(ast.Module(body=body, type_ignores=[])) == ast.dump(
            ast.Module(body=rejection, type_ignores=[])):
        return []
    expected = ast.parse(
        "from pops.public_api_exports import resolve_public_library_alias\n"
        "return resolve_public_library_alias(__name__, name)"
    ).body
    valid_args = (len(hook.args.args) == 1 and hook.args.args[0].arg == "name"
                  and not hook.args.posonlyargs and not hook.args.kwonlyargs
                  and not hook.args.vararg and not hook.args.kwarg and not hook.args.defaults)
    if (not registered or not valid_args or hook.decorator_list
            or isinstance(hook, ast.AsyncFunctionDef)
            or ast.dump(ast.Module(body=body, type_ignores=[])) != ast.dump(
                ast.Module(body=expected, type_ignores=[]))):
        return ["unknown or imperative lazy export route"]
    if any(isinstance(node, ast.Name) and node.id == "__name__"
           and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)):
        return ["lazy export module identity is overwritten"]
    exports = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "__all__" for target in node.targets):
            exports = ast.literal_eval(node.value)
    return ["registered alias missing from __all__: " + alias.public_name
            for alias in registered if alias.public_name not in exports]


def _public_alias_contract():
    import runpy
    return runpy.run_path(str(POPS / "public_api_exports.py"))


def test_public_library_alias_routes_are_exact_and_registered():
    contract = _public_alias_contract()
    aliases = contract["PUBLIC_LIBRARY_ALIASES"]
    failures = []
    observed = set()
    for path in _source_paths():
        module = _module_name(path).removesuffix(".__init__")
        if _layer_of(module) is None:
            continue
        observed.add(module)
        tree = ast.parse(path.read_text(), str(path))
        failures.extend(module + ": " + error
                        for error in _public_alias_route_errors(module, tree, aliases))
    for key, alias in aliases.items():
        assert key == (alias.public_module, alias.public_name)
        assert alias.contract_version == contract["PUBLIC_LIBRARY_ALIAS_VERSION"] == 1
        assert alias.public_module in observed
        target = REPO_ROOT / alias.source_path
        assert target.is_file(), alias.source_path
        owner_tree = ast.parse(target.read_text(), str(target))
        assert any(isinstance(node, ast.ClassDef) and node.name == alias.canonical_name
                   for node in owner_tree.body), "alias must name the owner's own class declaration"
    assert not failures, "public compatibility alias violation(s): " + "; ".join(failures)


def test_public_alias_fence_rejects_fake_unregistered_and_imperative_routes():
    aliases = _public_alias_contract()["PUBLIC_LIBRARY_ALIASES"]
    good = ast.parse('def __getattr__(name):\n    from pops.public_api_exports import resolve_public_library_alias\n    return resolve_public_library_alias(__name__, name)\n__all__ = ["FanLi15RawMomentPath"]\n')
    assert not _public_alias_route_errors("pops.numerics", good, aliases)
    assert _public_alias_route_errors("pops.numerics", good, {})
    assert _public_alias_route_errors("pops.fields", good, aliases)
    for body in (
        "from pops.moments.fan_li_path import FanLi15RawMomentPath\nreturn FanLi15RawMomentPath",
        "from pops.public_api_exports import resolve_public_library_alias\nreturn resolve_public_library_alias(__name__, name)()",
        "from pops.runtime import Runtime\nreturn Runtime()",
        "from pops.public_api_exports import resolve_public_library_alias\nreturn resolve_public_library_alias('pops.numerics', name)",
    ):
        fake = ast.parse("def __getattr__(name):\n    " + body.replace("\n", "\n    "))
        assert _public_alias_route_errors("pops.numerics", fake, aliases)
    missing = ast.parse("__all__ = ['FanLi15RawMomentPath']")
    assert _public_alias_route_errors("pops.numerics", missing, aliases)
    ordinary = ast.parse("from pops.moments.fan_li_path import FanLi15RawMomentPath")
    assert list(_intra_targets(ordinary)) == ["pops.moments.fan_li_path"]
    assert "moments" not in ALLOWED["numerics"]


def test_public_library_alias_registry_has_only_stdlib_dependencies():
    tree = ast.parse((POPS / "public_api_exports.py").read_text())
    allowed = {"__future__", "dataclasses", "enum", "importlib", "pathlib", "types", "typing"}
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(isinstance(node, ast.ImportFrom) and node.level == 0
               and node.module in allowed for node in imports)


def _public_alias_import_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
                node.module == "pops.public_api_exports"
                or node.level and node.module == "public_api_exports"
                or (node.module == "pops" or node.level and node.module is None)
                and any(alias.name == "public_api_exports" for alias in node.names)):
            yield node
        elif isinstance(node, ast.Import) and any(
                alias.name == "pops.public_api_exports" for alias in node.names):
            yield node
        elif (isinstance(node, ast.Call) and node.args
                and (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                     or isinstance(node.func, ast.Name) and node.func.id in {"import_module", "__import__"})
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {"pops.public_api_exports", ".public_api_exports"}):
            yield node


def test_public_alias_contract_is_consumed_only_by_registered_facade_hooks():
    aliases = _public_alias_contract()["PUBLIC_LIBRARY_ALIASES"]
    violations = []
    for path in _source_paths():
        module = _module_name(path).removesuffix(".__init__")
        tree = ast.parse(path.read_text(), str(path))
        imports = list(_public_alias_import_nodes(tree))
        if not imports:
            continue
        hooks = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "__getattr__"]
        if (_public_alias_route_errors(module, tree, aliases) or len(hooks) != 1
                or imports != [hooks[0].body[-2]]):
            violations.append(module)
    assert not violations, "compatibility routing is not compiler/runtime authority: " + ", ".join(violations)


def test_public_alias_contract_import_scanner_covers_core_bypass_spellings():
    for source in (
        "from pops.public_api_exports import resolve_public_library_alias",
        "from pops import public_api_exports",
        "import pops.public_api_exports as routes",
        "importlib.import_module('pops.public_api_exports')",
        "from ..public_api_exports import resolve_public_library_alias",
        "from .. import public_api_exports",
        "import_module('.public_api_exports', 'pops')",
        "__import__('pops.public_api_exports')",
    ):
        assert len(list(_public_alias_import_nodes(ast.parse(source)))) == 1


def _numerics_ir_imports(module, tree):
    """Resolve every lexical import spelling, including literal dynamic routes."""
    from importlib.util import resolve_name
    package = module.removesuffix(".__init__") if module.endswith(".__init__") else module.rsplit(".", 1)[0]
    result = []
    import_callables = {"import_module": "import_module", "__import__": "__import__"}
    for item in ast.walk(tree):
        if isinstance(item, ast.ImportFrom) and item.module in {"importlib", "builtins"}:
            import_callables.update((alias.asname or alias.name, alias.name) for alias in item.names
                                    if alias.name in {"import_module", "__import__"})
    for item in ast.walk(tree):
        if isinstance(item, ast.Assign):
            original = item.value.id if isinstance(item.value, ast.Name) else (
                item.value.attr if isinstance(item.value, ast.Attribute) else "")
            if original in import_callables:
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        import_callables[target.id] = import_callables[original]
    def add(node, target, names=()):
        if target == "pops._ir" or target.startswith("pops._ir."):
            result.append((node, target, names))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(node, alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = (resolve_name("." * node.level + (node.module or ""), package)
                      if node.level else node.module or "")
            add(node, target, tuple(alias.name for alias in node.names))
            if target == "pops":
                for alias in node.names:
                    add(node, target + "." + alias.name)
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if name not in import_callables:
                continue
            argument = node.args[0] if node.args else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"), None)
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            target = argument.value
            if target.startswith("."):
                context = node.args[1] if len(node.args) > 1 else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "package"), None)
                if not isinstance(context, ast.Constant) or not isinstance(context.value, str):
                    raise AssertionError("relative dynamic import has no literal package authority")
                target = resolve_name(target, context.value)
            add(node, target)
            if target == "pops" and import_callables[name] == "__import__":
                fromlist = next((k.value for k in node.keywords if k.arg == "fromlist"), None)
                if len(node.args) > 3:
                    fromlist = node.args[3]
                if isinstance(fromlist, (ast.Tuple, ast.List)):
                    for value in fromlist.elts:
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            add(node, "pops." + value.value)
    return result


def _numerics_ir_route_errors(module, tree):
    errors = []
    for node, target, names in _numerics_ir_imports(module, tree):
        allowed = (module == "pops.numerics.normalized_polynomial_path"
                   and isinstance(node, ast.ImportFrom) and node in tree.body
                   and node.level == 0 and target == "pops._ir.path_arithmetic"
                   and names == ("PathArithmeticComposition",)
                   and all(alias.asname is None for alias in node.names))
        if not allowed:
            errors.append((target, names))
    return errors


def test_numerics_lower_ir_edge_has_exact_generic_path_marker_owner():
    observed = []
    for path in _source_paths():
        module = _module_name(path)
        if _layer_of(module) != "numerics":
            continue
        tree = ast.parse(path.read_text(), str(path))
        assert not _numerics_ir_route_errors(module, tree), module
        observed.extend((module, target, names)
                        for _, target, names in _numerics_ir_imports(module, tree))
    assert observed == [("pops.numerics.normalized_polynomial_path",
                         "pops._ir.path_arithmetic", ("PathArithmeticComposition",))]
    assert ALLOWED["_ir"] == {"identity"}
    assert "moments" not in ALLOWED["numerics"]
    marker = ast.parse((POPS / "_ir/path_arithmetic.py").read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(marker))


def test_numerics_marker_fence_refuses_alternate_absolute_relative_and_nested_spellings():
    sources = (
        "import pops._ir.expr as expr", "from .._ir.expr import Expr",
        "from pops import _ir", "from .. import _ir",
        "def route():\n    from pops._ir.expr import Expr",
        "importlib.import_module('pops._ir.expr')", "import_module('.expr', 'pops._ir')",
        "__import__('pops._ir.expr')", "__import__('pops', fromlist=['_ir'])",
        "importlib.import_module(name='pops._ir.expr')",
        "from importlib import import_module as load\nload('pops._ir.expr')",
        "from builtins import __import__ as load\nload('pops._ir.expr')",
        "from builtins import __import__ as load\nload('pops', fromlist=['_ir'])",
        "load = importlib.import_module\nload('pops._ir.expr')",
        "from pops._ir.path_arithmetic import PathArithmeticComposition as Other",
        "def route():\n    from pops._ir.path_arithmetic import PathArithmeticComposition",
    )
    for source in sources:
        assert _numerics_ir_route_errors("pops.numerics.normalized_polynomial_path", ast.parse(source)), source
    canonical = ast.parse("from pops._ir.path_arithmetic import PathArithmeticComposition")
    assert not _numerics_ir_route_errors("pops.numerics.normalized_polynomial_path", canonical)
    assert _numerics_ir_route_errors("pops.numerics.other", canonical)


def test_pointwise_boundary_leaf_has_no_upper_layer_authority():
    leaf = POPS / "model/pointwise_boundary.py"
    tree = ast.parse(leaf.read_text(encoding="utf-8"), str(leaf))
    imports = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports == {"__future__", "typing", "pops._ir.expr", "pops.model.handles"}
    source = (POPS / "boundary/interior_trace.py").read_text(encoding="utf-8")
    assert "from pops.model.pointwise_boundary import BoundaryValue" in source
    assert "pops.fields" not in source


def test_field_and_boundary_consumers_share_exact_pointwise_read_identity():
    from pops.model.pointwise_boundary import BoundaryValue, resolve_handle
    from pops.fields.boundary_values import BoundaryValue as FieldBoundaryValue
    from pops.fields._references import resolve_handle as FieldResolveHandle
    from pops.boundary.interior_trace import InteriorTrace

    assert FieldBoundaryValue is BoundaryValue
    assert FieldResolveHandle is resolve_handle
    assert InteriorTrace.__mro__[1] is BoundaryValue



def test_numerical_library_expression_surface_preserves_exact_language_owners():
    import importlib
    from pops.model import expression_language as language
    assert language.EXPRESSION_LANGUAGE_VERSION == 1
    source = POPS / "model/expression_language.py"
    tree = ast.parse(source.read_text())
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
    aliases = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        assert node.level == 0 and node.module.startswith("pops._ir.")
        original = importlib.import_module(node.module)
        for alias in node.names:
            assert alias.asname is None
            assert getattr(language, alias.name) is getattr(original, alias.name)
            aliases[alias.name] = node.module
    assert set(aliases) == set(language.__all__)
    assert not hasattr(language, "__getattr__")
