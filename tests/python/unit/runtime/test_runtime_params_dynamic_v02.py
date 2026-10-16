"""Source authoring accepts finite parameter families beyond the former inline array."""
import pytest
from tests.python.unit.runtime.test_capacity_limits import _impl, _model_with_runtime_params


@pytest.mark.parametrize("count", [33, 65, 129])
def test_runtime_parameter_indices_are_complete_and_sorted(count):
    implementation = _impl(_model_with_runtime_params(count))
    nodes = implementation.assign_runtime_indices()
    assert len(nodes) == count
    assert [node.name for node in nodes] == sorted(node.name for node in nodes)
    assert [node.to_cpp() for node in nodes] == [f"params.get({index})" for index in range(count)]
    assert "RuntimeParams::declaration(%d)" % count in implementation._runtime_params_member()
    assert "pops::Real(0)" not in implementation._runtime_params_member()


def test_dynamic_parameter_storage_report_has_no_model_count_ceiling():
    from pops.physics.aux import max_runtime_params
    from pops.model.manifest import module_manifest_of
    assert max_runtime_params() is None
    manifest = module_manifest_of(_model_with_runtime_params(65))
    assert manifest.params_utilization == {"count": 65, "limit": None, "status": "dynamic"}


@pytest.mark.parametrize("capacity", [-1, True, 1.5, 1 << 64])
def test_parameter_capacity_requires_exact_uint64(capacity):
    from pops.codegen._plans import BindInputs
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        BindInputs(resources={"runtime_parameter_byte_capacity": capacity})


def test_bind_capacity_is_per_incarnation_and_initial_storage_deduplicates_bricks(monkeypatch):
    from types import SimpleNamespace
    from pops.runtime import _runtime_parameter_storage as storage

    # These are authenticated cardinalities at this unit boundary. Payload
    # allocation and native sizeof are exercised by the C++ receiver separately.
    monkeypatch.setattr(storage, "_cardinalities", lambda _: {"model:a": 65, "program:l:0": 65})
    result = storage.resolve_runtime_parameter_storage(
        SimpleNamespace(), {}, storage_bytes=lambda n: 96 + 16 * n
    )
    assert result["byte_capacity"] == 1136
    assert result["initial_storage_bytes"] == 2272
    assert result["scope"] == "immutable_incarnation"
    with pytest.raises(ValueError, match="requires 1136"):
        storage.resolve_runtime_parameter_storage(
            SimpleNamespace(), {"runtime_parameter_byte_capacity": 1135},
            storage_bytes=lambda n: 96 + 16 * n,
        )
    assert storage.resolve_runtime_parameter_storage(
        SimpleNamespace(), {"runtime_parameter_byte_capacity": 5000},
        storage_bytes=lambda n: 96 + 16 * n,
    )["byte_capacity"] == 5000


def _artifact_routes(count=65):
    from types import SimpleNamespace
    from pops import Case
    from pops.model import Module
    from pops.model.bind_schema import BindSchema
    from pops.params import RuntimeParam
    module = Module("parameter_storage")
    names = tuple("p_%03d" % index for index in range(count))
    for name in names:
        module.param(RuntimeParam(name, default=1.))
    case = Case("parameter_storage")
    case.block("fluid", module)
    schema = BindSchema.from_problem(case)
    program = SimpleNamespace(program=object(), program_block_routes=((0, "fluid"),),
        program_param_routes=tuple((0, name, index, 1.) for index, name in enumerate(names)))
    return SimpleNamespace(bind_schema=schema,
        blocks=(SimpleNamespace(name="fluid", model=SimpleNamespace(runtime_param_names=names)),),
        layout_programs=(SimpleNamespace(layout_id="layout", program=program),))


def test_cardinalities_authenticate_qualified_schema_and_count_each_carrier_once():
    from pops.runtime._runtime_parameter_storage import resolve_runtime_parameter_storage
    calls = []
    def native_size(count):
        calls.append(count)
        return 80+16*count
    result = resolve_runtime_parameter_storage(_artifact_routes(), {}, storage_bytes=native_size)
    assert result["cardinalities"] == {"model:fluid": 65, "program:layout:0": 65}
    assert calls == [65, 65]
    assert result["initial_storage_bytes"] == 2240
    assert result["byte_capacity"] == 1120


@pytest.mark.parametrize("fault, message", [
    ("hole", "hole"), ("duplicate", "duplicated"), ("foreign", "BindSchema/native artifact drift"),
    ("block", "block route is duplicated"), ("missing", "routing metadata"),
])
def test_bad_route_is_rejected_before_native_storage_query(fault, message):
    from pops.runtime._runtime_parameter_storage import resolve_runtime_parameter_storage
    artifact = _artifact_routes()
    program = artifact.layout_programs[0].program
    routes = list(program.program_param_routes)
    if fault == "hole":
        routes[-1] = (0, routes[-1][1], 65, 1.)
    elif fault == "duplicate":
        routes[-1] = (0, routes[-1][1], 0, 1.)
    elif fault == "foreign":
        routes[-1] = (0, "foreign", 64, 1.)
    elif fault == "block":
        program.program_block_routes += program.program_block_routes
    program.program_param_routes = tuple(routes)
    if fault == "missing":
        del program.program_param_routes
    def forbidden_query(_):
        pytest.fail("native storage queried before complete schema/route validation")
    with pytest.raises(ValueError, match=message):
        resolve_runtime_parameter_storage(artifact, {}, storage_bytes=forbidden_query)


def _platform():
    from pops._platform_contracts import proven_serial_manifest
    return proven_serial_manifest(backend="production", target="system", abi="headers|clang|c++23")


def _context():
    from pops._platform_contracts import ExecutionContext, ExecutionResource, proven_serial_manifest
    backend = proven_serial_manifest(backend="production", target="system", abi="headers|clang|c++23", runtime=True)
    return ExecutionContext(backend=backend, communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"), device=ExecutionResource("device", "host"))


def test_parameter_storage_budget_does_not_replace_execution_authority(monkeypatch):
    from pops.runtime import _platform_manifest as binding

    context = _context()
    monkeypatch.setattr(binding, "native_runtime_backend", lambda platform: context.backend)
    resources = {"execution_context": context, "runtime_parameter_byte_capacity": 4096}
    assert binding.execution_context_for_bind(_platform(), resources) is context
    assert resources["runtime_parameter_byte_capacity"] == 4096
    with pytest.raises(TypeError, match="competing keys"):
        binding.execution_context_for_bind(_platform(), {**resources, "device": "host"})


@pytest.mark.parametrize("capacity", (True, -1, 1 << 64, None))
def test_invalid_parameter_storage_budget_rejects_before_native_lookup(monkeypatch, capacity):
    from pops.runtime import _platform_manifest as binding

    def forbidden(platform):
        pytest.fail("invalid capacity reached native lookup")

    monkeypatch.setattr(binding, "native_runtime_backend", forbidden)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        binding.execution_context_for_bind(_platform(), {
            "execution_context": _context(), "runtime_parameter_byte_capacity": capacity,
        })



def test_parameter_storage_uses_only_the_authenticated_selected_variant(monkeypatch):
    from types import SimpleNamespace
    from pops import _native_selector as selector
    from pops.runtime._runtime_parameter_storage import resolve_runtime_parameter_storage

    calls = []
    def selected(*, required):
        calls.append(required)
        return SimpleNamespace(runtime_parameter_storage_bytes=lambda count: 64+16*count)
    monkeypatch.setattr(selector, "selected_native_module", selected)
    result = resolve_runtime_parameter_storage(_artifact_routes(count=3), {})
    assert calls == [True]
    assert result["carrier_bytes"] == {"model:fluid": 112, "program:layout:0": 112}
    assert result["initial_storage_bytes"] == 224


def test_parameter_storage_never_selects_a_default_native_dimension(monkeypatch):
    from pops import _native_selector as selector
    from pops.runtime._runtime_parameter_storage import resolve_runtime_parameter_storage

    def absent(*, required):
        assert required
        raise RuntimeError("no native dimension was selected")
    monkeypatch.setattr(selector, "selected_native_module", absent)
    with pytest.raises(RuntimeError, match="no native dimension was selected"):
        resolve_runtime_parameter_storage(_artifact_routes(count=3), {})
    # A parameter-free artifact needs no native sizeof or module selection.
    assert resolve_runtime_parameter_storage(_artifact_routes(count=0), {})["initial_storage_bytes"] == 0
