"""Spec 4: the intra-pops import graph is acyclic and respects the layering.

The sub-packages form a directed acyclic dependency stack:

    _ir       -> identity                        (canonical scalar identity codec)
    identity  imports nothing else in pops
    frames    -> identity
    domain    -> frames, identity
    model     -> _ir, identity, params
    problem   -> _ir, identity, model
    physics   -> _ir, identity, model, problem
    time      -> _ir, identity, model, params
    mesh      -> domain, frames, identity, model, params
    amr       -> _ir, identity, mesh, model, time
    layouts   -> amr, mesh
    boundary  -> _ir, domain, identity, model, representations
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
    "domain": {"frames", "identity"},
    "model": {"_ir", "identity", "params"},
    "problem": {"_ir", "identity", "model"},
    "physics": {"_ir", "identity", "model", "problem"},
    "time": {"_ir", "identity", "model", "params"},
    "initial": {"model"},
    "mesh": {"domain", "frames", "identity", "model", "params"},
    "amr": {"_ir", "identity", "mesh", "model", "time"},
    "layouts": {"amr", "mesh"},
    "boundary": {"_ir", "domain", "identity", "model", "representations"},
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

NATIVE_IMPORT_PHASE_OWNERS = {
    "pops._bootstrap": "package-bootstrap",
    "pops._native_collectives": "runtime-collective-call",
    "pops._platform_contracts": "platform-contract-resolution",
    "pops.codegen._compiled_artifact": "compiled-artifact-sealing",
    "pops.codegen.toolchain": "runtime-compiler-probe",
    "pops.external.artifacts": "external-artifact-authentication",
    "pops.external.compiler": "external-component-compilation",
    "pops.output._writers.hdf5": "collective-output-write",
    "pops.runtime_environment": "runtime-environment-resolution",
}


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


def test_native_extension_loads_have_explicit_phase_owners_at_every_scope():
    violations = []
    observed_declared_owners = set()
    for path in _source_paths():
        module = _module_name(path)
        lines = tuple(_native_import_lines(ast.parse(path.read_text(), str(path))))
        allowed = module == "pops.runtime" or module.startswith("pops.runtime.") \
            or module in NATIVE_IMPORT_PHASE_OWNERS
        if lines and module in NATIVE_IMPORT_PHASE_OWNERS:
            observed_declared_owners.add(module)
        if lines and not allowed:
            violations.append("%s:%s" % (module, ",".join(map(str, lines))))
    assert not violations, "unowned native-extension load(s): " + ", ".join(violations)
    stale = sorted(set(NATIVE_IMPORT_PHASE_OWNERS) - observed_declared_owners)
    assert not stale, "native phase-owner declaration(s) without a native load: " + ", ".join(stale)


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
    assert dependencies <= {"identity"}, (
        "pops.solvers may depend only on pops.identity; got %s"
        % sorted(dependencies))
