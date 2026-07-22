"""Collected native package test: compile, audit, install, load and call the real ABI consumer."""
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import pops
import numpy as np
from pops import interfaces
from pops.codegen.toolchain import (
    _probe_cxx_std,
    loader_cxx_std,
    pops_include,
    pops_loader_build_flags,
)
from pops.external import (
    ComponentPackageError,
    build_source_package_manifest,
    compile_component,
    load,
)
from pops.model import ComponentManifest
from pops.output import (
    CoarseOnly, ConsumerGraph, ExternalWriter, ParallelMode, ScientificOutput,
)
from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
from pops.time import every, on_start


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _manifest() -> ComponentManifest:
    interface = interfaces.NumericalFlux
    return ComponentManifest(
        uri="pops://external.test/fluxes/average", component_type="numerical_flux",
        version="1.0.0", facets=interface.facets,
        signature={
            "generic": True, "state_components": 2,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )


def _source(manifest: ComponentManifest) -> bytes:
    return f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {{
int evaluate(void*, const PopsNumericalFluxRequestV1* request,
             PopsNumericalFluxResultV1* result) {{
  if (!request || !result || request->left.dimension != 2 ||
      request->left.extents[0] != request->right.extents[0] ||
      request->left.extents[1] != request->right.extents[1] ||
      request->left.component_count != request->right.component_count ||
      request->left.component_count != result->normal_flux.component_count) return 2;
  const auto* left_values = static_cast<const double*>(request->left.data);
  const auto* right_values = static_cast<const double*>(request->right.data);
  const auto* normals = static_cast<const double*>(request->normals.data);
  auto* output = static_cast<double*>(result->normal_flux.data);
  const auto points = request->left.extents[0] * request->left.extents[1];
  for (std::size_t point = 0; point < points; ++point) {{
    double speed = 0.0;
    for (std::size_t component = 0; component < request->left.component_count; ++component) {{
      const auto li = point * request->left.axis_strides[1] + component * request->left.component_stride;
      const auto ri = point * request->right.axis_strides[1] + component * request->right.component_stride;
      const auto oi = point * result->normal_flux.axis_strides[1] + component * result->normal_flux.component_stride;
      const double left = left_values[li];
      const double right = right_values[ri];
      output[oi] = 0.5 * (left + right) * normals[point * request->normals.axis_strides[1]];
      speed = std::max(speed, std::max(std::abs(left), std::abs(right)));
    }}
    result->stability_bounds[point] = speed;
    result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
  }}
  result->status = {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

const PopsNumericalFluxApiV1 flux_table = {{
  {{sizeof(PopsNumericalFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, nullptr, nullptr}},
  &evaluate
}};
const PopsComponentInterfaceEntryV1 interface_entry = {{
  POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, sizeof(PopsNumericalFluxApiV1), &flux_table
}};
const PopsComponentApiV1 component_api = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &interface_entry
}};
}}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{
  return &component_api;
}}
'''.encode()


def _writer_manifest(name: str) -> ComponentManifest:
    interface = interfaces.Writer
    return ComponentManifest(
        uri="pops://external.test/writers/%s" % name, component_type="writer",
        version="1.0.0", facets=interface.facets,
        signature={
            "generic": True,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )


def _writer_source(manifest: ComponentManifest) -> bytes:
    return f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace {{
PopsComponentStatusV1 ok() {{
  return {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
}}

int fail(PopsWriterReceiptV1* receipt, const char* reason) {{
  if (receipt) receipt->status =
      {{sizeof(PopsComponentStatusV1), 1, POPS_COMPONENT_ABORT_RUN_V1, reason}};
  return 1;
}}

bool valid(const PopsWriterRequestV1* request) {{
  if (!request || request->struct_size < sizeof(PopsWriterRequestV1) ||
      !request->geometries || request->geometry_count == 0 ||
      (!request->fields && request->field_count != 0) ||
      (!request->diagnostics && request->diagnostic_count != 0) ||
      (request->field_count == 0 && request->diagnostic_count == 0) ||
      !request->metadata_json || !request->selection_identity ||
      !request->temporary_path || !request->published_path ||
      !request->snapshot_identity) return false;
  for (std::size_t index = 0; index < request->geometry_count; ++index) {{
    const auto& geometry = request->geometries[index];
    if (geometry.dimension != 2 || !geometry.cell_shape) return false;
    const auto cells = geometry.cell_shape[0] * geometry.cell_shape[1];
    if (!geometry.layout_identity || !geometry.layout_kind || !geometry.boxes ||
        geometry.box_count == 0 || geometry.valid_cells.size != cells ||
        geometry.coverage.size != cells ||
        geometry.cell_volumes.extents[0] * geometry.cell_volumes.extents[1] != cells)
      return false;
  }}
  for (std::size_t index = 0; index < request->field_count; ++index) {{
    const auto& field = request->fields[index];
    if (!field.field_identity || !field.reference_id ||
        !field.component_manifest_identity || !field.layout_identity ||
        !field.state_id || !field.centering || !field.units || field.dimension != 2 ||
        !field.pieces || field.piece_count == 0) return false;
    for (std::size_t piece = 0; piece < field.piece_count; ++piece)
      if (!field.pieces[piece].values.data ||
          field.pieces[piece].values.extents[0] == 0 ||
          field.pieces[piece].values.extents[1] == 0)
        return false;
  }}
  return true;
}}

std::string payload(const PopsWriterRequestV1* request) {{
  std::string result = "snapshot=" + std::string(request->snapshot_identity) + "\\n";
  result += "selection=" + std::string(request->selection_identity) + "\\n";
  result += "geometries=" + std::to_string(request->geometry_count) +
            " fields=" + std::to_string(request->field_count) +
            " diagnostics=" + std::to_string(request->diagnostic_count) + "\\n";
  for (std::size_t index = 0; index < request->field_count; ++index) {{
    const auto& field = request->fields[index];
    result += "field=" + std::string(field.field_identity) +
              " level=" + std::to_string(field.level) +
              " pieces=" + std::to_string(field.piece_count) + "\\n";
    double sum = 0.0;
    std::size_t values = 0;
    for (std::size_t piece_index = 0; piece_index < field.piece_count; ++piece_index) {{
      const auto& view = field.pieces[piece_index].values;
      const auto* data = static_cast<const double*>(view.data);
      const auto points = view.extents[0] * view.extents[1];
      for (std::size_t point = 0; point < points; ++point)
        for (std::size_t component = 0; component < view.component_count; ++component) {{
          sum += data[point + component * view.component_stride];
          ++values;
        }}
    }}
    result += "values=" + std::to_string(values) +
              " sum=" + std::to_string(sum) + "\\n";
  }}
  result += "metadata=" + std::string(request->metadata_json) + "\\n";
  return result;
}}

int verify(void*, const PopsWriterRequestV1* request, PopsWriterReceiptV1* receipt) {{
  if (!receipt || !valid(request)) return fail(receipt, "invalid complete snapshot");
  const auto body = payload(request);
  std::ofstream stream(request->temporary_path, std::ios::binary | std::ios::trunc);
  stream.write(body.data(), static_cast<std::streamsize>(body.size()));
  stream.close();
  if (!stream) return fail(receipt, "cannot stage writer output");
  receipt->bytes_written = body.size();
  receipt->content_digest = "writer-complete-snapshot-v1";
  receipt->status = ok();
  return 0;
}}

int publish(void*, const PopsWriterRequestV1* request, PopsWriterReceiptV1* receipt) {{
  if (!receipt || !valid(request)) return fail(receipt, "invalid publish snapshot");
  if (!std::filesystem::is_regular_file(request->temporary_path) ||
      std::filesystem::exists(request->published_path))
    return fail(receipt, "publish collision or missing temporary");
  const auto bytes = std::filesystem::file_size(request->temporary_path);
  std::filesystem::rename(request->temporary_path, request->published_path);
  receipt->bytes_written = bytes;
  receipt->content_digest = "writer-complete-snapshot-v1";
  receipt->status = ok();
  return 0;
}}

void discard(void*, const PopsWriterRequestV1* request) {{
  if (request && request->temporary_path)
    std::filesystem::remove(request->temporary_path);
}}

void rollback(void*, const PopsWriterRequestV1* request) {{
  if (!request) return;
  if (request->temporary_path) std::filesystem::remove(request->temporary_path);
  if (request->published_path) std::filesystem::remove(request->published_path);
}}

const PopsWriterApiV1 writer_table = {{
  {{sizeof(PopsWriterApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_WRITER_V1, 1, nullptr, nullptr}},
  &verify, &publish, &discard, &rollback
}};
const PopsComponentInterfaceEntryV1 interface_entry = {{
  POPS_NATIVE_INTERFACE_WRITER_V1, 1, sizeof(PopsWriterApiV1), &writer_table
}};
const PopsComponentApiV1 component_api = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &interface_entry
}};
}}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{
  return &component_api;
}}
'''.encode()


_CONSUMER = r'''#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
  if (argc != 7) return 90;
  pops::component::ExpectedNativeComponent expected{
    argv[2], argv[3], argv[4], argv[5], argv[6],
    {{POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, sizeof(PopsNumericalFluxApiV1)}}
  };
  auto loaded = pops::component::LoadedComponent::load(argv[1], expected);
  const auto& api = loaded.table<PopsNumericalFluxApiV1>(
      POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1);
  std::array<double, 4> left{2.0, 4.0, 1.0, 3.0};
  std::array<double, 4> right{6.0, 8.0, 3.0, 5.0};
  std::array<double, 4> normals{1.0, 0.0, 1.0, 0.0};
  std::array<double, 4> output{};
  std::array<double, 2> stability{};
  std::array<PopsComponentActionV1, 2> actions{};
  const PopsExecutionContextV1 execution{
    sizeof(PopsExecutionContextV1), 1, "test::execution-context",
    POPS_MEMORY_SPACE_HOST_V1,
    "test::backend", "test::cpu:0", POPS_SCALAR_FLOAT64_V1,
    POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
    POPS_PRECISION_FLOAT64_V1, POPS_PRECISION_FLOAT64_V1,
    0, "test::host-synchronous", 0, 0, "serial", "none"
  };
  const PopsLogicalTimeV1 logical_time{
    sizeof(PopsLogicalTimeV1), "test::clock", 7, 0, 0, 0,
    1, 1, 0.01, 0.25
  };
  const auto const_view = [](const double* data, std::size_t components) {
    return PopsConstFieldViewV1{
      sizeof(PopsConstFieldViewV1), data, 2, {1, 2, 1},
      {static_cast<std::ptrdiff_t>(2 * components),
       static_cast<std::ptrdiff_t>(components), 0},
      components, 1, POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
      POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
      "test::layout", "test::patch", POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1
    };
  };
  const auto mutable_view = [](double* data, std::size_t components) {
    return PopsFieldViewV1{
      sizeof(PopsFieldViewV1), data, 2, {1, 2, 1},
      {static_cast<std::ptrdiff_t>(2 * components),
       static_cast<std::ptrdiff_t>(components), 0},
      components, 1, POPS_FIELD_CENTERING_CELL_V1, 0, {0, 0, 0}, {0, 0, 0},
      POPS_SCALAR_FLOAT64_V1, POPS_MEMORY_SPACE_HOST_V1,
      "test::layout", "test::patch", POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1
    };
  };
  PopsNumericalFluxRequestV1 request{
    sizeof(PopsNumericalFluxRequestV1),
    const_view(left.data(), 2), const_view(right.data(), 2),
    const_view(normals.data(), 2), nullptr, logical_time, execution
  };
  PopsNumericalFluxResultV1 result{
    sizeof(PopsNumericalFluxResultV1), mutable_view(output.data(), 2),
    stability.data(), actions.data(), {}
  };
  if (pops::component::evaluate_faces(api, nullptr, request, result) != 0) return 91;
  if (output != std::array<double, 4>{4.0, 6.0, 2.0, 4.0}) return 92;
  if (std::abs(stability[0] - 8.0) > 1e-14 || std::abs(stability[1] - 5.0) > 1e-14)
    return 93;
  return 0;
}
'''


def test_source_component_executes_through_generic_native_loader_and_flux_consumer(tmp_path):
    manifest = _manifest()
    source = _source(manifest)
    (tmp_path / "average.cpp").write_bytes(source)
    package_data = build_source_package_manifest(
        components={"average": manifest}, payloads={"average.cpp": ("source", source)})
    package_path = tmp_path / "average.pops.json"
    package_path.write_text(json.dumps(package_data), encoding="utf-8")

    package = load(package_path)
    component = package.require("average", interface=interfaces.NumericalFlux)()
    artifact = compile_component(component)
    from pops import _pops
    assert artifact.platform_manifest.communicator.require("component communicator") == (
        "MPI_COMM_WORLD" if _pops.__has_mpi__ else "serial")
    install_root = tmp_path / "installed"
    installed = artifact.install(install_root)
    assert installed.path == install_root / (
        artifact.artifact_identity.hexdigest + artifact.suffix)
    same = artifact.install(install_root)
    assert same.path == installed.path
    assert same.native_handle is None and installed.native_handle is None
    assert not tuple(install_root.glob(".pops-component-*"))

    consumer_source = tmp_path / "consumer.cpp"
    consumer_source.write_text(_CONSUMER, encoding="utf-8")
    consumer = tmp_path / "consumer"
    compiler, cflags, lflags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    command = [compiler, "-std=" + standard, *cflags, "-I", pops_include(),
               str(consumer_source), "-o", str(consumer), *lflags]
    if sys.platform.startswith("linux"):
        command.append("-ldl")
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([
        str(consumer), str(installed.path), manifest.component_id,
        manifest.semantic_digest.token, manifest.manifest_digest.token,
        interfaces.NumericalFlux.to_data()["catalog_sha256"],
        _pops.abi_key(),
    ], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stderr
    loaded = installed.load()
    assert loaded.native_handle.report()["abi_key"] == _pops.abi_key()
    assert installed.to_data()["provenance"]["origin"] == "source"
    assert installed.runtime_contract.native_interface["name"] == "numerical_flux"

    # Never truncate a shared object that is still dlopen-ed: Linux may fault a later access to
    # the shortened mapping with SIGBUS.  Exercise the same authenticated-install collision on a
    # distinct, never-loaded copy so the test remains a valid filesystem-integrity check.
    collision_root = tmp_path / "collision"
    collision = artifact.install(collision_root)
    collision.path.write_bytes(b"tampered")
    with pytest.raises(ComponentPackageError, match="content-addressed path has other bytes"):
        artifact.install(collision_root)


def _compile_writer(tmp_path: Path, name: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = _writer_manifest(name)
    source = _writer_source(manifest)
    source_name = "%s.cpp" % name
    (tmp_path / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={name: manifest}, payloads={source_name: ("source", source)})
    package_path = tmp_path / (name + ".pops.json")
    package_path.write_text(json.dumps(package_data), encoding="utf-8")
    component = load(package_path).require(name, interface=interfaces.Writer)()
    return compile_component(component, include=str(ROOT / "include"))


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_external_writer_scalar", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _writer_case(example, artifacts, *, adaptive: bool):
    from pops.layouts import Uniform
    from pops.output import SelectedLevels

    core = example.build_authoring(output_root="unused")
    core.numerics.boundaries.add(example.build_transport_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    if adaptive:
        core.case.initials.add(example.build_initial_condition(core))
    core.case.program(core.program)
    communicators = {
        artifact.platform_manifest.communicator.require(
            "external Writer artifact communicator"
        )
        for artifact in artifacts
    }
    if communicators == {"serial"}:
        output_mode = ParallelMode.SERIAL
    elif communicators == {"MPI_COMM_WORLD"}:
        output_mode = ParallelMode.ROOT
    else:
        raise RuntimeError(
            "external Writers do not share one supported communicator: %r"
            % sorted(communicators)
        )
    outputs = []
    if not adaptive:
        outputs.append(ScientificOutput(
            format=ExternalWriter(
                artifacts[0], extension=".popsbin", mode=output_mode),
            schedule=on_start(clock=core.program.clock),
            fields=(core.tracer_state,), levels=CoarseOnly(), target="reject-stage",
        ))
    outputs.append(ScientificOutput(
        format=ExternalWriter(
            artifacts[-1], extension=".popsbin", mode=output_mode),
        schedule=every(1, clock=core.program.clock),
        fields=(core.tracer_state,),
        levels=SelectedLevels(0, 1) if adaptive else CoarseOnly(),
        target="amr-writer" if adaptive else "uniform-writer",
    ))
    core.case.consumers(ConsumerGraph.from_consumers(tuple(outputs)))
    layout = example.build_amr_layout(core) if adaptive else Uniform(core.grid)
    if adaptive:
        initial_state = None
    else:
        nx, ny = core.grid.cells
        initial_state = np.full((1, ny, nx), 0.05, dtype=np.float64)
    return core, layout, initial_state


def _bind_writer_case(example, core, layout, artifacts, initial_state=None):
    validated = pops.validate(core.case)
    resolved = pops.resolve(validated, layout=layout, components=tuple(artifacts))
    compiled = pops.compile(resolved)
    communicator = compiled.platform_manifest.communicator.require(
        "external Writer simulation communicator"
    )
    resources = (
        {}
        if communicator == "serial"
        else {"execution_context": pops.ExecutionContext.mpi_world(compiled)}
    )
    bind_inputs = (
        {}
        if initial_state is None
        else {"initial_state": {"tracer": initial_state}}
    )
    simulation = pops.bind(
        compiled,
        params=example.build_bind_params(core),
        resources=resources,
        **bind_inputs,
    )
    return simulation


def test_qualified_writer_runs_through_uniform_and_amr_runtime_transactions(tmp_path):
    example = _load_example()
    first = _compile_writer(tmp_path / "source-one", "writer_one")
    second = _compile_writer(tmp_path / "source-two", "writer_two")

    uniform_core, uniform_layout, uniform_initial = _writer_case(
        example, (first, second), adaptive=False)
    uniform = _bind_writer_case(
        example, uniform_core, uniform_layout, (first, second), uniform_initial)
    runtime = uniform

    # Rejection owns and discards the verified native temporary without publishing it.
    runtime._output_root = tmp_path / "uniform-output"
    transactions = runtime._stage_consumers(at_start=True)
    assert len(transactions) == 1
    stage_dir = runtime._output_root / "reject-stage"
    staged = tuple(stage_dir.glob(".*.writer-stage"))
    assert len(staged) == 1 and staged[0].is_file()
    rejected = transactions[0].reject()
    runtime._output_root = None
    assert rejected.status == "rejected"
    assert not staged[0].exists()
    assert not tuple(stage_dir.glob("*.popsbin"))

    # Every output carries one qualified component authority; one remaining installed Writer
    # can never stand in for a missing named Writer.
    ids = [row.output_format_data["component_id"]
           for row in runtime.consumer_graph.nodes]
    assert len(ids) == len(set(ids)) == 2
    with pytest.raises(ValueError, match="exact component is not installed"):
        RuntimeConsumerPublisher(SimpleNamespace(
            _consumer_graph=runtime.consumer_graph,
            _installed_components={ids[-1]: runtime._installed_components[ids[-1]]},
            _component_manifests=runtime._component_manifests,
            _execution_context=runtime._execution_context,
            _layout_plan=runtime._layout_plan,
            _retain_output_recoveries=runtime._retain_output_recoveries,
        ))

    # Two separately qualified Writers cannot claim the same logical target.
    collision_graph = ConsumerGraph(tuple(
        replace(row, target_uri="same-target")
        for row in runtime.consumer_graph.nodes
    ))
    with pytest.raises(ValueError, match="same logical target"):
        RuntimeConsumerPublisher(SimpleNamespace(
            _consumer_graph=collision_graph,
            _installed_components=runtime._installed_components,
            _component_manifests=runtime._component_manifests,
            _execution_context=runtime._execution_context,
            _layout_plan=runtime._layout_plan,
            _retain_output_recoveries=runtime._retain_output_recoveries,
        ))

    run_report = pops.run(
        uniform, t_end=1.0e-4, max_steps=1,
        output_dir=tmp_path / "uniform-run")
    assert run_report.accepted_steps == 1
    uniform_files = tuple((tmp_path / "uniform-run" / "uniform-writer").glob("*.popsbin"))
    assert len(uniform_files) == 1
    assert not tuple((tmp_path / "uniform-run").rglob(".*.writer-stage*"))
    assert any("fields=1" in path.read_text(encoding="utf-8")
               for path in uniform_files)
    report = uniform.inspect()
    assert report.runtime == "uniform"
    assert len(report.instance["installed_components"]) == 2

    amr_core, amr_layout, amr_initial = _writer_case(
        example, (first,), adaptive=True)
    assert amr_initial is None
    amr = _bind_writer_case(
        example, amr_core, amr_layout, (first,))
    run_report = pops.run(
        amr, t_end=1.0e-4, max_steps=1,
        output_dir=tmp_path / "amr-run")
    assert run_report.accepted_steps == 1
    amr_files = tuple((tmp_path / "amr-run" / "amr-writer").glob("*.popsbin"))
    assert len(amr_files) == 1
    assert not tuple((tmp_path / "amr-run").rglob(".*.writer-stage*"))
    body = amr_files[0].read_text(encoding="utf-8")
    assert "geometries=2 fields=2" in body
    assert " level=0 " in body and " level=1 " in body
    assert amr.inspect().runtime == "adaptive"
