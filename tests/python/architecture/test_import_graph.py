"""Spec 4: the intra-pops import graph is acyclic and respects the layering.

The sub-packages form a directed acyclic dependency stack:

    ir        imports nothing else in pops
    model     -> ir, params
    physics   -> ir, model, params
    time      -> ir, model, params
    mesh      -> model, params                    (owner-qualified layout subjects + parameters)
    numerics  -> model, params                    (typed subjects + parameter use sites)
    linalg    -> (nothing)                       (Spec 5: abstract algebra descriptors)
    solvers   -> (nothing)                       (typed solver descriptor sink)
    moments   -> ir                              (Spec 5: moment-model toolkit)
    diagnostics -> linalg                        (Spec 5: Norm takes a typed norm kind)
    params    -> (nothing)                       (typed parameter dependency sink)
    output    -> model, time                     (qualified selections and schedules)
    external  -> model                           (authenticated component manifests)
    lib       -> ir, model, time, physics, moments, numerics,
                 fields, params, solvers
    codegen   -> ir, model, physics, time, lib, solvers, params,
                 external, fields
    runtime   -> authoring/lowering contracts, including resolved fields,
                 and is the ONLY layer allowed to import _pops

This test builds the cross-layer edges from module-scope imports (``ast``,
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
    "ir": set(),
    # ADC-654: params is a dependency-free semantic sink.  Model owns ParamRegistry and the
    # authoring consumers call the central ParamKind x ParamUse validator instead of growing local
    # isinstance/int/float coercion branches.
    "model": {"ir", "params"},
    "physics": {"ir", "model", "params"},
    "time": {"ir", "model", "params"},
    # ADC-673: LayoutPlan assigns canonical model Handle subjects and therefore depends on the
    # dependency-free identity/data-model portion of pops.model. Model still has no mesh edge.
    "mesh": {"model", "params"},
    # Central descriptor packages remain runtime-free. Their allowed semantic dependencies are
    # explicit below rather than hidden through the root facade.
    "numerics": {"model", "params"},
    # Spec 5 sec.5.6: pops.linalg names the algebra (A x = b, operators, norms, reductions).
    # It imports only the flat pops.descriptors module (not a tracked layer) -> no edges.
    "linalg": set(),
    # Solver descriptors are a dependency sink: exact scalar serialization is loaded lazily when
    # an option is authored, so importing the catalog never reaches into IR, fields, codegen, lib,
    # or runtime. Custom-solver registry hooks are attached onto its namespace by
    # pops.codegen.solvers (codegen -> solvers, acyclic).
    "solvers": set(),
    # Resolved field contracts share the canonical symbolic/model/time values they authenticate.
    # BoundaryTopology remains a lazy, in-function dependency so fields does not create a
    # module-scope mesh edge and mesh stays independent of fields.
    "fields": {"ir", "model", "time"},
    "moments": {"ir"},
    # Spec 5 sec.5.13: pops.diagnostics.measures.Norm takes a typed pops.linalg.norms kind
    # (L1 / L2 / LInf), so diagnostics imports linalg (acyclic: linalg imports nothing).
    "diagnostics": {"linalg"},
    "params": set(),
    # Exact output selections authenticate canonical field Handles; writers remain runtime-free.
    "output": {"model", "time"},
    # External package manifests are semantic authoring inputs and consume the canonical
    # ComponentManifest value; platform/runtime imports remain lazy phase-boundary operations.
    "external": {"model"},
    # lib is presets-only: its implementations compose ordinary public descriptors and Programs.
    # Ready model/time presets therefore consume the central field-output, parameter, numerics,
    # and solver catalogs; they never define competing copies of those types and must not reach up
    # into codegen or runtime. Each allowed target is below lib and none imports lib, so preset
    # composition cannot create a cycle.
    "lib": {"ir", "model", "time", "physics", "moments", "numerics", "fields",
            "params", "solvers"},
    # codegen.solvers (the solver-gen DSL, criterion 19) imports pops.solvers at module scope to
    # attach custom-solver registry hooks. solvers has no layered dependencies, so the edge remains
    # acyclic.
    "codegen": {"ir", "model", "physics", "time", "lib", "solvers", "params", "external",
                "fields"},
    # Runtime consumes the resolved field layout/consumer contracts. fields remains runtime-free,
    # so this is a one-way execution-boundary dependency rather than a cycle.
    "runtime": {"ir", "model", "physics", "time", "lib", "mesh", "codegen", "params",
                "output", "fields"},
}
LAYERS = set(ALLOWED)


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


def test_ir_remains_a_dependency_sink():
    """Exact symbolic values may be consumed everywhere; the IR must never depend back."""
    dependencies = sorted(dst for dst, _ in _build_edges().get("ir", set()))
    assert not dependencies, (
        "pops.ir is the foundational symbolic sink and must have no layered module-scope "
        "dependencies; got %s" % dependencies)


def test_solver_catalog_remains_a_dependency_sink():
    """Preset/codegen consumers may depend on solvers; the descriptor catalog stays inert."""
    dependencies = sorted(dst for dst, _ in _build_edges().get("solvers", set()))
    assert not dependencies, (
        "pops.solvers is a descriptor dependency sink and must have no layered module-scope "
        "dependencies; got %s" % dependencies)
