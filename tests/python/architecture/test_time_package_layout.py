"""The temporal authoring package has one private DAG and one exact public façade."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / "python"
TIME = PYTHON / "pops" / "time"

PRIVATE_PACKAGES = {"_graph", "_history", "_methods", "_program", "_schedule", "_step"}
OLD_MODULES = (
    "_program_contract",
    "history",
    "history_persistence",
    "history_persistence_report",
    "history_persistence_validate",
    "method_properties",
    "method_tableau",
    "passes_facade",
    "program",
    "program_authoring",
    "program_base",
    "program_call",
    "program_clocks",
    "program_commit_validation",
    "program_condensed",
    "program_core",
    "program_detach",
    "program_diagnostics",
    "program_dt_bound",
    "program_dump",
    "program_freeze",
    "program_graph_conversion",
    "program_history",
    "program_inspect",
    "program_local",
    "program_passes",
    "program_rebuild",
    "program_region_validation",
    "program_rhs",
    "program_serialization",
    "program_solve",
    "program_temporal_manifest",
    "program_time_handles",
    "program_transaction",
    "program_value_validation",
    "schedule",
    "schedule_domains",
    "schedule_lowering",
    "schedule_protocol",
    "step_strategy",
    "step_transaction",
    "synchronization",
)

PUBLIC = (
    "Program", "ProgramValue", "StageStateSet", "ResidualSolution",
    "CoupledImplicitEuler", "LocalLinear", "LocalResidual",
    "SolveOutcome", "FieldSolveOutcome", "SolveAction", "FailRun", "RejectAttempt",
    "SOLVE_STATUSES", "Schedule",
    "StepStrategy", "FixedDt", "AdaptiveCFL", "ErrorControlledDt", "ExternalTimeGrid",
    "ALL_PROVISIONAL_STORES", "AcceptanceGuard", "BlockProjection", "GuardRole",
    "ProjectAndRecheck", "ProvisionalStore", "StepTransactionPlan", "StepTransactionReport",
    "ProgramGraph", "GraphProgramValue", "StateRead", "Unknown", "OperatorCall",
    "Solve", "Branch", "Loop", "Region", "RegionCapture",
    "Synchronize", "Commit", "ValueRef",
    "Clock", "TimePoint", "StagePoint",
    "RungeKuttaTableau", "AdditiveRungeKuttaTableau",
    "MethodCertificate", "MethodProperties", "AdditiveMethodCertificate",
    "AdditiveMethodProperties", "ProgramMethodCertificate", "SSPCertificate",
    "UnknownOrder", "certify_program_graph",
    "SampleAndHold", "SynchronizationRelation",
    "TimeState", "StageHandle", "HistoryHandle", "StateEndpointHandle", "CopyCurrent",
    "HistoryPersistence", "Dense", "Interval", "Revolve",
    "Domain", "AcceptedStep", "Attempt", "Stage", "ClockTick", "AMRLevel",
    "EventHandle", "Event", "WallOutput", "Trigger", "Always", "Every",
    "AtStart", "AtEnd", "When", "OffPolicy", "Hold", "Skip", "Zero",
    "AccumulateDt", "Error",
    "ScheduleTimeline", "ScheduleDueKind", "ScheduleAction", "ScheduleComment",
    "ScheduleDomainIR", "ScheduleDueIR", "ScheduleOffIR", "ScheduleLoweringIR",
    "UnresolvedScheduleCondition",
    "always", "every", "when", "on_start", "on_end",
    "eliminate_dead_nodes", "eliminate_common_subexpressions",
    "eliminate_redundant_field_solves", "optimize",
)


def _module(path: Path) -> str:
    parts = list(path.relative_to(PYTHON).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _sources() -> dict[str, Path]:
    return {
        _module(path): path
        for path in TIME.rglob("*.py")
        if " 2" not in path.name and " 3" not in path.name
    }


def _time_target(module: str, modules: set[str]) -> str | None:
    candidates = {
        candidate
        for candidate in modules
        if module == candidate or module.startswith(candidate + ".")
    }
    return max(candidates, key=len) if candidates else None


def _imports(path: Path) -> tuple[str, ...]:
    result = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


def _group(module: str) -> str:
    parts = module.split(".")
    leaf = parts[2] if len(parts) > 2 else "facade"
    if leaf in PRIVATE_PACKAGES or leaf in {"_authoring", "_rhs_terms"}:
        return leaf
    return "core"


def test_old_flat_modules_are_deleted_without_forwarders() -> None:
    for name in OLD_MODULES:
        assert not (TIME / (name + ".py")).exists(), name
        assert importlib.util.find_spec("pops.time." + name) is None, name


def test_public_time_facade_is_exact_and_identity_preserving() -> None:
    import pops
    import pops.time as time
    from pops.time._methods.tableau import RungeKuttaTableau
    from pops.time._program.api import Program
    from pops.time._schedule.api import Schedule
    from pops.time._step.strategy import StepStrategy

    assert tuple(time.__all__) == PUBLIC
    assert not hasattr(time, "__getattr__")
    assert pops.Program is time.Program is Program
    assert time.Schedule is Schedule
    assert time.StepStrategy is StepStrategy
    assert time.RungeKuttaTableau is RungeKuttaTableau
    assert all(hasattr(time, name) for name in PUBLIC)
    assert not hasattr(Program, "emit_cpp_program")
    assert not hasattr(Program, "_check_lowerable")
    assert not hasattr(Program, "_check_schedules_lowerable")
    assert not hasattr(Program, "scratch_plan")
    assert not hasattr(StepStrategy, "runtime_controller")


def test_private_package_initializers_do_not_create_parallel_authorities() -> None:
    for name in PRIVATE_PACKAGES - {"_graph"}:
        source = (TIME / name / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def test_time_sources_never_import_compiler_runtime_or_native_layers() -> None:
    forbidden = ("pops.codegen", "pops.runtime", "_pops", "pops._pops")
    violations = []
    for module, path in _sources().items():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            elif isinstance(node, ast.Call) and node.args \
                    and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "import_module" \
                    and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                targets = [node.args[0].value]
            for target in targets:
                if any(target == item or target.startswith(item + ".") for item in forbidden):
                    violations.append("%s:%d -> %s" % (module, node.lineno, target))
    assert not violations, "forbidden temporal consumer imports:\n  " + "\n  ".join(violations)


def test_time_module_graph_is_acyclic_and_respects_private_layering() -> None:
    sources = _sources()
    modules = set(sources)
    edges = {module: set() for module in modules}
    for module, path in sources.items():
        for imported in _imports(path):
            target = _time_target(imported, modules)
            if target is not None and target != module:
                edges[module].add(target)

    allowed = {
        "_authoring": set(),
        "core": {"_authoring"},
        "_graph": {"core"},
        "_history": {"core"},
        "_rhs_terms": {"core"},
        "_step": {"core"},
        "_methods": {"_graph", "core"},
        "_schedule": {"_graph", "core"},
        "_program": {
            "_authoring", "_graph", "_history", "_methods", "_rhs_terms",
            "_schedule", "_step", "core",
        },
    }
    violations = []
    for source, targets in edges.items():
        if source == "pops.time":
            continue
        source_group = _group(source)
        for target in targets:
            target_group = _group(target)
            if source_group != target_group and target_group not in allowed[source_group]:
                violations.append("%s -> %s" % (source, target))
    assert not violations, "forbidden temporal layer edges:\n  " + "\n  ".join(violations)

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        index[module] = low[module] = len(index)
        stack.append(module)
        active.add(module)
        for target in sorted(edges[module]):
            if target not in index:
                visit(target)
                low[module] = min(low[module], low[target])
            elif target in active:
                low[module] = min(low[module], index[target])
        if low[module] == index[module]:
            component = []
            while True:
                target = stack.pop()
                active.remove(target)
                component.append(target)
                if target == module:
                    break
            components.append(tuple(sorted(component)))

    for module in sorted(modules):
        if module not in index:
            visit(module)
    cycles = [component for component in components if len(component) > 1]
    assert not cycles, "temporal import SCCs: %s" % cycles


def test_method_coefficient_authority_is_one_way() -> None:
    tableau = (TIME / "_methods" / "tableau.py").read_text(encoding="utf-8")
    properties = (TIME / "_methods" / "properties.py").read_text(encoding="utf-8")
    assert "from pops.time._methods.coefficients import" in tableau
    assert "from pops.time._methods.coefficients import" in properties
    assert "from pops.time._methods.tableau import" not in properties
