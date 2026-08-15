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
    mesh      -> analytic, domain, frames, identity, model, params
    amr       -> _ir, identity, mesh, model, time
    layouts   -> amr, mesh
    boundary  -> _ir, analytic, domain, identity, model, representations
    numerics  -> identity, model, params
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
    "mesh": {"analytic", "domain", "frames", "identity", "model", "params"},
    "amr": {"_ir", "identity", "mesh", "model", "time"},
    "layouts": {"amr", "mesh"},
    "boundary": {"_ir", "analytic", "domain", "identity", "model", "representations"},
    "numerics": {"identity", "model", "params"},
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
    "pops._capabilities_report",
    "pops._native_collectives",
    "pops._paraview_python_bootstrap",
    "pops._platform_contracts",
    "pops.codegen._compiled_artifact",
    "pops.codegen.inspect_compiled",
    "pops.external.artifacts",
    "pops.external.compiler",
    "pops.output._writers.hdf5",
    "pops.runtime._platform_manifest",
    "pops.runtime._runtime_authorities",
    "pops.runtime._threading",
    "pops.runtime.defaults",
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


_MAX_STATIC_IMPORT_VALUES = 32


def _possible_string_values(node, bindings):
    """Return statically enumerable string values, or ``None`` for a dynamic expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({node.value})
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values = set()
        for element in node.elts:
            element_values = _possible_string_values(element, bindings)
            if element_values is None:
                return None
            values.update(element_values)
            if len(values) > _MAX_STATIC_IMPORT_VALUES:
                return None
        return frozenset(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _possible_string_values(node.left, bindings)
        right = _possible_string_values(node.right, bindings)
        if left is None or right is None \
                or len(left) * len(right) > _MAX_STATIC_IMPORT_VALUES:
            return None
        return frozenset(prefix + suffix for prefix in left for suffix in right)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            expression = value.value if isinstance(value, ast.FormattedValue) else value
            possible = _possible_string_values(expression, bindings)
            if possible is None:
                return None
            parts.append(possible)
        values = {""}
        for part in parts:
            if len(values) * len(part) > _MAX_STATIC_IMPORT_VALUES:
                return None
            values = {prefix + suffix for prefix in values for suffix in part}
        return frozenset(values)
    return None


def _static_string_bindings(tree):
    """Over-approximate names assigned from finite string expressions at any lexical scope."""
    bindings = {}
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            candidates.extend((target, value) for target in targets)
        elif isinstance(node, ast.For):
            candidates.append((node.target, node.iter))

    dynamic_targets = {
        target.id
        for target, value in candidates
        if isinstance(target, ast.Name)
        and any(
            isinstance(dependency, ast.Name) and dependency.id == target.id
            for dependency in ast.walk(value)
        )
    }
    changed_names = set()
    for _ in range(8):
        changed_names = set()
        for target, value in candidates:
            if not isinstance(target, ast.Name) or target.id in dynamic_targets:
                continue
            possible = _possible_string_values(value, bindings)
            if possible is None:
                continue
            merged = bindings.get(target.id, frozenset()) | possible
            if len(merged) > _MAX_STATIC_IMPORT_VALUES:
                dynamic_targets.add(target.id)
                bindings.pop(target.id, None)
                continue
            if merged != bindings.get(target.id):
                bindings[target.id] = merged
                changed_names.add(target.id)
        if not changed_names:
            break
    else:
        dynamic_targets.update(changed_names)

    changed = True
    while changed:
        changed = False
        for target, value in candidates:
            if not isinstance(target, ast.Name) or target.id in dynamic_targets:
                continue
            if any(
                isinstance(dependency, ast.Name) and dependency.id in dynamic_targets
                for dependency in ast.walk(value)
            ):
                dynamic_targets.add(target.id)
                changed = True
    for name in dynamic_targets:
        bindings.pop(name, None)
    return bindings


def _native_import_lines(tree):
    """Yield direct native loads and unresolved dynamic imports at any lexical scope."""
    importlib_aliases = {"importlib"}
    import_module_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name in {"_pops", "pops._pops"} for alias in node.names):
                yield node.lineno
            importlib_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"_pops", "pops._pops"} or any(
                    alias.name == "_pops" and (module == "pops" or node.level)
                    for alias in node.names):
                yield node.lineno
            if node.level == 0 and module == "importlib":
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )

    bindings = _static_string_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        is_import = (
            isinstance(node.func, ast.Name)
            and node.func.id in import_module_aliases | {"__import__"}
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        )
        if not is_import:
            continue
        possible = _possible_string_values(node.args[0], bindings)
        if possible is None or possible & {"_pops", "pops._pops"}:
            yield node.lineno


def test_no_module_bypasses_the_native_selector_with_a_direct_extension_import():
    violations = []
    for path in _source_paths():
        module = _module_name(path)
        lines = tuple(_native_import_lines(ast.parse(path.read_text(), str(path))))
        if lines:
            violations.append("%s:%s" % (module, ",".join(map(str, lines))))
    assert not violations, (
        "direct or unresolved dynamic native-extension load bypasses pops._native_selector: "
        + ", ".join(violations)
    )


def test_native_import_fence_tracks_indirection_and_importlib_aliases():
    indirect = ast.parse("""
import importlib as loader
for candidate in ("_pops", "pops._pops"):
    loader.import_module(candidate)
""")
    imported_alias = ast.parse("""
from importlib import import_module as load
load("pops." + "_pops")
""")
    unresolved = ast.parse("""
from importlib import import_module
def load(candidate):
    return import_module(candidate)
""")
    external = ast.parse("""
import importlib
for candidate in ("catalyst_conduit", "conduit"):
    importlib.import_module(candidate)
""")

    assert tuple(_native_import_lines(indirect)) == (4,)
    assert tuple(_native_import_lines(imported_alias)) == (3,)
    assert tuple(_native_import_lines(unresolved)) == (4,)
    assert tuple(_native_import_lines(external)) == ()


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
