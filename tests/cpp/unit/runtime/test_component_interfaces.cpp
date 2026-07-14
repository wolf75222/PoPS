#include <pops/runtime/config/component_interfaces.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include "component_abi_test_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace abi = pops::component::test_support;

struct Context {};

struct FluxComponent {
  std::vector<std::string> requirements() const { return {"state", "normal"}; }
  double stability() const { return 1.0; }
  pops::component::EvaluationOutcome<double> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<double>::ok(2.0);
  }
};

struct BoundaryComponent {
  std::vector<std::string> providers() const { return {"state", "logical_time"}; }
  int stencil() const { return 1; }
};

struct TaggerComponent {
  std::vector<std::string> requirements() const { return {"indicator"}; }
  std::string lower(Context&) const { return "tagger-plan"; }
};

struct ClusteringComponent {
  std::string lower(Context&) const { return "cluster-plan"; }
  std::vector<std::string> effects() const { return {"topology"}; }
};

struct TransferComponent {
  int stencil() const { return 2; }
  std::string restart() const { return "stateless"; }
};

struct RefluxComponent {
  std::vector<std::string> effects() const { return {"conservative-correction"}; }
  std::string report() const { return "reflux-report"; }
};

struct SolverComponent {
  pops::component::EvaluationOutcome<int> evaluate(Context&) const {
    return pops::component::EvaluationOutcome<int>::reject("non-converged");
  }
  std::string restart() const { return "warm-start"; }
  std::string report() const { return "solve-report"; }
};

struct WriterComponent {
  std::vector<std::string> effects() const { return {"io"}; }
  std::string format(const double& value) const { return std::to_string(value); }
  std::string report() const { return "writer-report"; }
};

static_assert(pops::component::Requirement<FluxComponent>);
static_assert(pops::component::Stability<FluxComponent>);
static_assert(pops::component::FallibleEvaluation<FluxComponent, Context&>);
static_assert(pops::component::Provider<BoundaryComponent>);
static_assert(pops::component::Stencil<BoundaryComponent>);
static_assert(pops::component::Requirement<TaggerComponent>);
static_assert(pops::component::Lowering<TaggerComponent, Context>);
static_assert(pops::component::Lowering<ClusteringComponent, Context>);
static_assert(pops::component::Effects<ClusteringComponent>);
static_assert(pops::component::Stencil<TransferComponent>);
static_assert(pops::component::Restart<TransferComponent>);
static_assert(pops::component::Effects<RefluxComponent>);
static_assert(pops::component::Report<RefluxComponent>);
static_assert(pops::component::FallibleEvaluation<SolverComponent, Context&>);
static_assert(pops::component::Restart<SolverComponent>);
static_assert(pops::component::Format<WriterComponent, double>);
static_assert(pops::component::Report<WriterComponent>);

pops::component::RegistrationRecord record(std::string id, std::string semantic) {
  return {
      std::move(id),
      "test.external",
      {},
      {"external", "pops://external.test/package", std::move(semantic), "manifest-digest"},
  };
}

TEST(ComponentInterfaces, FallibleOutcomeKeepsTransactionActionExplicit) {
  Context context;
  const auto flux = FluxComponent{}.evaluate(context);
  EXPECT_EQ(flux.status, pops::component::EvaluationStatus::kOk);
  ASSERT_TRUE(flux.value.has_value());
  EXPECT_EQ(*flux.value, 2.0);

  const auto solve = SolverComponent{}.evaluate(context);
  EXPECT_EQ(solve.status, pops::component::EvaluationStatus::kReject);
  EXPECT_EQ(solve.reason, "non-converged");
  EXPECT_THROW(pops::component::EvaluationOutcome<int>::retry(""), std::invalid_argument);
}

TEST(ComponentInterfaces, RegistryIsCollisionSafeIdempotentAndExplicitlyFrozen) {
  pops::component::Registry registry;
  const auto& first = registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(first.component_type, "test.external");
  EXPECT_EQ(registry.revision(), 1u);

  const auto& repeated =
      registry.register_component(record("pops://external.test/flux@1.0.0", "s1"));
  EXPECT_EQ(&first, &repeated);
  EXPECT_EQ(registry.revision(), 1u);

  EXPECT_THROW(
      registry.register_component(record("pops://external.test/flux@1.0.0", "different")),
      std::invalid_argument);
  EXPECT_EQ(registry.revision(), 1u);

  registry.freeze();
  EXPECT_TRUE(registry.frozen());
  EXPECT_THROW(
      registry.register_component(record("pops://external.test/writer@1.0.0", "s2")),
      std::logic_error);
}

PopsComponentTableHeaderV1 abi_header(std::size_t size, PopsNativeInterfaceIdV1 id) {
  return {static_cast<std::uint32_t>(size), POPS_COMPONENT_PROTOCOL_ABI_V1, id, 1,
          nullptr, nullptr};
}

PopsComponentStatusV1 ok_status() {
  return {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
}

const double* values(const PopsConstFieldViewV1& view) {
  return static_cast<const double*>(view.data);
}

double* values(PopsFieldViewV1& view) {
  return static_cast<double*>(view.data);
}

TEST(ComponentInterfaces, ExactAbiConsumersExecuteEveryClosedScientificFamily) {
  std::array<double, 2> left{2.0, 4.0}, right{6.0, 8.0}, normal{1.0, 0.0};
  const auto execution = abi::host_execution_context();
  EXPECT_NO_THROW(pops::component::validate_execution_context(execution));
  auto anonymous_execution = execution;
  anonymous_execution.execution_identity = "";
  EXPECT_THROW(pops::component::validate_execution_context(anonymous_execution),
               std::invalid_argument);
  auto packed_distributed = execution;
  packed_distributed.communicator_f_handle = 1;
  packed_distributed.communicator_datatype_f_handle = 2;
  packed_distributed.communicator_identity = "mpi:world";
  packed_distributed.communicator_datatype_identity = "mpi:byte";
  EXPECT_NO_THROW(pops::component::validate_execution_context(packed_distributed));
  auto ambiguous_distributed = packed_distributed;
  ambiguous_distributed.communicator_datatype_f_handle = 0;
  EXPECT_THROW(pops::component::validate_execution_context(ambiguous_distributed),
               std::invalid_argument);
  auto anonymous_device = execution;
  anonymous_device.device_identity = "";
  EXPECT_THROW(pops::component::validate_execution_context(anonymous_device),
               std::invalid_argument);
  std::array<double, 2> flux{};
  double speed = 0.0;
  PopsComponentActionV1 action = POPS_COMPONENT_ABORT_RUN_V1;
  PopsNumericalFluxApiV1 flux_api{
      abi_header(sizeof(PopsNumericalFluxApiV1), POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1),
      +[](void*, const PopsNumericalFluxRequestV1* request,
          PopsNumericalFluxResultV1* result) {
        const auto* left_values = values(request->left);
        const auto* right_values = values(request->right);
        auto* output_values = values(result->normal_flux);
        for (std::size_t c = 0; c < request->left.component_count; ++c)
          output_values[c] = 0.5 * (left_values[c] + right_values[c]);
        result->stability_bounds[0] = 8.0;
        result->actions[0] = POPS_COMPONENT_CONTINUE_V1;
        result->status = ok_status();
        return 0;
      }};
  PopsNumericalFluxRequestV1 flux_request{
      sizeof(PopsNumericalFluxRequestV1),
      abi::const_field_view(left.data(), 1, 1, 2),
      abi::const_field_view(right.data(), 1, 1, 2),
      abi::const_field_view(normal.data(), 1, 1, 2),
      nullptr, abi::logical_time(), execution};
  PopsNumericalFluxResultV1 flux_result{
      sizeof(PopsNumericalFluxResultV1),
      abi::field_view(flux.data(), 1, 1, 2), &speed, &action, {}};
  EXPECT_EQ(pops::component::evaluate_faces(
                flux_api, nullptr, flux_request, flux_result), 0);
  EXPECT_EQ(flux, (std::array<double, 2>{4.0, 6.0}));
  EXPECT_DOUBLE_EQ(speed, 8.0);
  auto mismatched_patch = flux_request;
  mismatched_patch.right.patch_identity = "test::other-patch";
  EXPECT_THROW(pops::component::evaluate_faces(
                   flux_api, nullptr, mismatched_patch, flux_result),
               std::invalid_argument);
  auto unsupported_dimension = flux_request;
  unsupported_dimension.left.dimension = 3;
  unsupported_dimension.left.extents[2] = 1;
  unsupported_dimension.left.axis_strides[2] = 1;
  EXPECT_THROW(pops::component::evaluate_faces(
                   flux_api, nullptr, unsupported_dimension, flux_result),
               std::invalid_argument);
  auto invalid_time = flux_request;
  invalid_time.logical_time.fraction_denominator = 0;
  EXPECT_THROW(pops::component::evaluate_faces(
                   flux_api, nullptr, invalid_time, flux_result),
               std::invalid_argument);

  std::array<double, 2> ghosts{};
  PopsGhostBoundaryApiV1 ghost_api{
      abi_header(sizeof(PopsGhostBoundaryApiV1), POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1),
      +[](void*, const PopsGhostBoundaryRequestV1* request, PopsComponentStatusV1* status) {
        if ((request->region.kind != POPS_BOUNDARY_FACE_V1 &&
             request->region.kind != POPS_BOUNDARY_CORNER_V1) ||
            request->dependency_count != 1 || request->parameter_count != 1 ||
            std::strcmp(request->dependencies[0].qualified_id, "case::velocity") != 0 ||
            std::strcmp(request->parameters[0].qualified_id, "case::inlet") != 0)
          return 9;
        auto* ghosts = static_cast<double*>(request->ghosts.data);
        const auto* interior = static_cast<const double*>(request->interior.data);
        for (std::size_t c = 0; c < request->ghosts.component_count; ++c)
          ghosts[c] = -interior[c];
        *status = ok_status();
        return 0;
      }};
  const std::array<std::int32_t, 1> face_axes{0}, face_sides{-1};
  const PopsBoundaryRegionV1 face_region{
      sizeof(PopsBoundaryRegionV1), POPS_BOUNDARY_FACE_V1, 2, 1,
      face_axes.size(), face_axes.data(), face_sides.data(), "x-low"};
  const PopsQualifiedConstFieldV1 ghost_dependencies[] = {{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::velocity",
      abi::const_field_view(normal.data(), 1, 1, 2)}};
  const PopsQualifiedScalarV1 ghost_parameters[] = {{
      sizeof(PopsQualifiedScalarV1), "case::inlet", 2.5}};
  PopsGhostBoundaryRequestV1 ghost_request{
      sizeof(PopsGhostBoundaryRequestV1), "case::ghost-producer", "case::state",
      "case::ghost-output",
      abi::const_field_view(left.data(), 1, 1, 2),
      abi::field_view(ghosts.data(), 1, 1, 2),
      abi::const_field_view(normal.data(), 1, 1, 2), face_region,
      1, ghost_dependencies, 1, ghost_parameters, abi::logical_time(), execution};
  auto status = ok_status();
  EXPECT_EQ(pops::component::apply_ghost_boundary(
                ghost_api, nullptr, ghost_request, status), 0);
  EXPECT_EQ(ghosts, (std::array<double, 2>{-2.0, -4.0}));
  const std::array<std::int32_t, 2> corner_axes{0, 1}, corner_sides{-1, 1};
  ghost_request.region = {
      sizeof(PopsBoundaryRegionV1), POPS_BOUNDARY_CORNER_V1, 2, 2,
      corner_axes.size(), corner_axes.data(), corner_sides.data(), "x-low-y-high"};
  EXPECT_EQ(pops::component::apply_ghost_boundary(
                ghost_api, nullptr, ghost_request, status), 0);
  auto invalid_region = ghost_request;
  invalid_region.region.kind = POPS_BOUNDARY_FACE_V1;
  EXPECT_THROW(pops::component::apply_ghost_boundary(
                   ghost_api, nullptr, invalid_region, status), std::invalid_argument);

  std::array<double, 2> direction{1.0, 2.0}, boundary_output{};
  const auto field_eval = +[](void*, const PopsFieldBoundaryRequestV1* request,
                              PopsComponentStatusV1* result) {
    if (request->state_count != 1 || request->direction_count > 1 ||
        request->field_count != 1 || request->parameter_count != 1 ||
        request->output_count != 1 || request->level != 2 ||
        request->logical_time.tick != 7 || request->rate.present != 1 ||
        request->nonlinear_iterate.present != 1 ||
        std::strcmp(request->fields[0].qualified_id, "case::coefficient") != 0)
      return 11;
    auto& output = request->outputs[0].values;
    auto* output_values = values(output);
    const auto* state_values = values(request->states[0].values);
    const auto* direction_values = request->direction_count == 0
                                       ? nullptr : values(request->directions[0].values);
    for (std::size_t c = 0; c < output.component_count; ++c)
      output_values[c] = state_values[c] +
                         (direction_values == nullptr ? 0.0 : direction_values[c]);
    *result = ok_status();
    return 0;
  };
  PopsFieldBoundaryClosureApiV1 field_boundary_api{
      abi_header(sizeof(PopsFieldBoundaryClosureApiV1),
                 POPS_NATIVE_INTERFACE_FIELD_BOUNDARY_CLOSURE_V1),
      field_eval, field_eval};
  const PopsQualifiedConstFieldV1 boundary_states[] = {{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::state",
      abi::const_field_view(left.data(), 1, 1, 2)}};
  const PopsQualifiedConstFieldV1 boundary_directions[] = {{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::normal-direction",
      abi::const_field_view(direction.data(), 1, 1, 2)}};
  const PopsQualifiedConstFieldV1 boundary_fields[] = {{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::coefficient",
      abi::const_field_view(right.data(), 1, 1, 2)}};
  const PopsQualifiedScalarV1 boundary_parameters[] = {{
      sizeof(PopsQualifiedScalarV1), "case::robin-alpha", 0.5}};
  PopsQualifiedFieldV1 boundary_outputs[] = {{
      sizeof(PopsQualifiedFieldV1), "case::residual",
      abi::field_view(boundary_output.data(), 1, 1, 2)}};
  PopsFieldBoundaryRequestV1 field_boundary_request{
      sizeof(PopsFieldBoundaryRequestV1), "case::field-boundary", ghost_request.region,
      abi::const_field_view(normal.data(), 1, 1, 2),
      1, boundary_states, 1, boundary_directions, 1, boundary_fields,
      1, boundary_parameters, 1, boundary_outputs,
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::rate",
       abi::const_field_view(right.data(), 1, 1, 2)},
      {sizeof(PopsQualifiedConstFieldV1), 1, "case::nonlinear-iterate",
       abi::const_field_view(left.data(), 1, 1, 2)},
      2, abi::logical_time(), execution};
  EXPECT_EQ(pops::component::evaluate_field_boundary(
                field_boundary_api, nullptr, field_boundary_request, status, true), 0);
  EXPECT_EQ(boundary_output, (std::array<double, 2>{3.0, 6.0}));
  field_boundary_request.direction_count = 0;
  field_boundary_request.directions = nullptr;
  EXPECT_EQ(pops::component::evaluate_field_boundary(
                field_boundary_api, nullptr, field_boundary_request, status, false), 0);
  EXPECT_EQ(boundary_output, left);

  std::array<double, 4> tag_values{-1.0, 2.0, 0.0, 3.0};
  std::array<std::uint8_t, 4> tags{};
  PopsTaggerApiV1 tagger_api{
      abi_header(sizeof(PopsTaggerApiV1), POPS_NATIVE_INTERFACE_TAGGER_V1),
      +[](void*, const PopsTaggerRequestV1* request, PopsComponentStatusV1* result) {
        const auto* state = static_cast<const double*>(request->state.data);
        for (std::size_t i = 0; i < request->tags.size; ++i)
          request->tags.data[i] = state[i] > 0.0 ? 1u : 0u;
        *result = ok_status();
        return 0;
      }};
  PopsTaggerRequestV1 tag_request{
      sizeof(PopsTaggerRequestV1),
      abi::const_field_view(tag_values.data(), 2, 2),
      {sizeof(PopsByteViewV1), tags.data(), tags.size()}, abi::logical_time(), execution};
  EXPECT_EQ(pops::component::tag_batch(tagger_api, nullptr, tag_request, status), 0);
  EXPECT_EQ(tags, (std::array<std::uint8_t, 4>{0, 1, 0, 1}));

  std::array<std::int64_t, 2> extents{4, 1};
  std::array<std::int64_t, 4> boxes{};
  std::size_t box_count = 0;
  PopsClusteringApiV1 cluster_api{
      abi_header(sizeof(PopsClusteringApiV1), POPS_NATIVE_INTERFACE_CLUSTERING_V1),
      +[](void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* result) {
        request->boxes[0] = 1;
        request->boxes[1] = 0;
        request->boxes[2] = 3;
        request->boxes[3] = 0;
        *request->box_count = 1;
        *result = ok_status();
        return 0;
      }};
  PopsClusteringRequestV1 cluster_request{
      sizeof(PopsClusteringRequestV1),
      {sizeof(PopsConstByteViewV1), tags.data(), tags.size()}, extents.data(), 2,
      boxes.data(), 1, &box_count, execution};
  EXPECT_EQ(pops::component::cluster_tags(
                cluster_api, nullptr, cluster_request, status), 0);
  EXPECT_EQ(box_count, 1u);
  EXPECT_EQ(boxes, (std::array<std::int64_t, 4>{1, 0, 3, 0}));

  std::array<double, 1> transferred{};
  std::array<std::int32_t, 2> ratio{2, 2};
  PopsTransferApiV1 transfer_api{
      abi_header(sizeof(PopsTransferApiV1), POPS_NATIVE_INTERFACE_TRANSFER_V1),
      +[](void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* result) {
        const auto* source = static_cast<const double*>(request->source.data);
        auto* destination = static_cast<double*>(request->destination.data);
        destination[0] = 0.25 * (source[0] + source[1] + source[2] + source[3]);
        *result = ok_status();
        return 0;
      }};
  PopsTransferRequestV1 transfer_request{
      sizeof(PopsTransferRequestV1),
      abi::const_field_view(tag_values.data(), 2, 2),
      abi::field_view(transferred.data(), 1, 1),
      ratio.data(), 2, POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1, execution};
  EXPECT_EQ(pops::component::apply_transfer(
                transfer_api, nullptr, transfer_request, status), 0);
  EXPECT_DOUBLE_EQ(transferred[0], 1.0);
  auto wrong_transfer_shape = transfer_request;
  wrong_transfer_shape.destination.extents[1] = 2;
  EXPECT_THROW(pops::component::apply_transfer(
                   transfer_api, nullptr, wrong_transfer_shape, status),
               std::invalid_argument);

  std::array<double, 2> coarse{1.0, 3.0}, fine{2.0, 7.0}, register_values{};
  PopsRefluxApiV1 reflux_api{
      abi_header(sizeof(PopsRefluxApiV1), POPS_NATIVE_INTERFACE_REFLUX_V1),
      +[](void*, const PopsRefluxRequestV1* request, PopsComponentStatusV1* result) {
        auto* output = static_cast<double*>(request->flux_register.data);
        const auto* fine_values = static_cast<const double*>(request->fine_integrated.data);
        const auto* coarse_values = static_cast<const double*>(request->coarse_integrated.data);
        for (std::size_t i = 0; i < 2; ++i)
          output[i] += fine_values[i] - coarse_values[i];
        *result = ok_status();
        return 0;
      }};
  PopsRefluxRequestV1 reflux_request{
      sizeof(PopsRefluxRequestV1),
      abi::const_field_view(coarse.data(), 1, 2),
      abi::const_field_view(fine.data(), 1, 2),
      abi::field_view(register_values.data(), 1, 2), execution};
  EXPECT_EQ(pops::component::deposit_reflux(
                reflux_api, nullptr, reflux_request, status), 0);
  EXPECT_EQ(register_values, (std::array<double, 2>{1.0, 4.0}));

  static constexpr PopsTopologyLabelV1 label_vocabulary[] = {
      {1, "island-a", "test-topology"}, {2, "island-b", "test-topology"}};
  PopsFieldTopologyApiV1 topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV1), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V1),
      +[](void*, const PopsFieldTopologyRequestV1* request,
          PopsFieldTopologyResultV1* result) {
        const std::uint8_t mask[] = {1, 1, 0, 1};
        const std::int32_t labels[] = {1, 1, 0, 2};
        std::copy(mask, mask + 4, request->material_mask.data);
        std::copy(labels, labels + 4, request->component_labels.data);
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "test-topology";
        result->topology_digest = "topology-v1";
        result->status = ok_status();
        return 0;
      }};
  PopsFieldTopologyApiV1 rejecting_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV1), POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V1),
      +[](void*, const PopsFieldTopologyRequestV1*,
          PopsFieldTopologyResultV1* result) {
        result->status = {sizeof(PopsComponentStatusV1), 17,
                          POPS_COMPONENT_REJECT_STEP_V1,
                          "topology rejected by test component"};
        return 0;
      }};
  const std::string topology_layout =
      "test::owned-topology-layout-identity-longer-than-small-string-storage";
  const std::string topology_patch =
      "test::owned-topology-patch-identity-longer-than-small-string-storage";
  const auto topology = [&] {
    std::array<double, 4> geometry{};
    std::string borrowed_layout = topology_layout;
    std::string borrowed_patch = topology_patch;
    const auto geometry_view = abi::const_field_view(
        geometry.data(), 2, 2, 1, borrowed_layout.c_str(), borrowed_patch.c_str());
    EXPECT_THROW(pops::component::prepare_field_topology(
                     rejecting_topology_api, nullptr, geometry_view, execution),
                 std::runtime_error);
    auto prepared = pops::component::prepare_field_topology(
        topology_api, nullptr, geometry_view, execution);
    std::fill(borrowed_layout.begin(), borrowed_layout.end(), 'x');
    std::fill(borrowed_patch.begin(), borrowed_patch.end(), 'y');
    return prepared;
  }();
  EXPECT_EQ(topology.topology_digest(), "topology-v1");
  EXPECT_EQ(topology.component_labels(),
            (std::vector<std::int32_t>{1, 1, 0, 2}));

  std::array<double, 4> rhs{1.0, 2.0, 0.0, 4.0}, solution{};
  PopsFieldSolverApiV1 solver_api{
      abi_header(sizeof(PopsFieldSolverApiV1), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V1),
      +[](void*, const PopsFieldSolverRequestV1* request, PopsSolveReportV1* report) {
        if (std::strcmp(request->topology_digest, "topology-v1") != 0) return 7;
        const auto* rhs_values = static_cast<const double*>(request->rhs.data);
        auto* solution_values = static_cast<double*>(request->solution.data);
        std::copy(rhs_values, rhs_values + 4, solution_values);
        report->converged = 1;
        report->iterations = 1;
        report->initial_residual = 1.0;
        report->final_residual = 0.0;
        report->status = ok_status();
        return 0;
      }};
  const auto solver_request = pops::component::bind_field_solver_request(
      topology,
      abi::const_field_view(rhs.data(), 2, 2, 1, topology_layout.c_str(),
                            topology_patch.c_str()),
      abi::field_view(solution.data(), 2, 2, 1, topology_layout.c_str(),
                      topology_patch.c_str()),
      execution, {},
      "{\"identity\":\"test::boundary\"}",
      1e-8, 0.0, 10);
  PopsSolveReportV1 solve_report{};
  solve_report.struct_size = sizeof(PopsSolveReportV1);
  EXPECT_EQ(pops::component::solve_field(
                solver_api, nullptr, solver_request, solve_report), 0);
  EXPECT_EQ(solution, rhs);
  std::array<double, 3> short_solution{};
  EXPECT_THROW(
      pops::component::bind_field_solver_request(
          topology,
          abi::const_field_view(rhs.data(), 2, 2, 1, topology_layout.c_str(),
                                topology_patch.c_str()),
          abi::field_view(short_solution.data(), 1, 3, 1,
                          topology_layout.c_str(), topology_patch.c_str()),
          execution, {},
          "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10),
      std::invalid_argument);

  struct WriterCallState {
    int publish_count = 0;
    bool reject_verification = false;
  } writer_state;
  PopsWriterApiV1 writer_api{
      abi_header(sizeof(PopsWriterApiV1), POPS_NATIVE_INTERFACE_WRITER_V1),
      +[](void* state, const PopsWriterRequestV1*, PopsWriterReceiptV1* receipt) {
        receipt->bytes_written = 999;
        receipt->content_digest = "verify-only";
        if (static_cast<WriterCallState*>(state)->reject_verification) {
          receipt->status = {sizeof(PopsComponentStatusV1), 23,
                             POPS_COMPONENT_REJECT_STEP_V1,
                             "writer verification rejected snapshot"};
          return 0;
        }
        receipt->status = ok_status();
        return 0;
      },
      +[](void* state, const PopsWriterRequestV1* request, PopsWriterReceiptV1* receipt) {
        if (request->snapshot_identity == nullptr || request->geometry_count != 1 ||
            request->field_count != 1 || request->fields[0].piece_count != 1 ||
            receipt->struct_size != sizeof(PopsWriterReceiptV1) ||
            receipt->bytes_written != 0 || receipt->content_digest != nullptr ||
            receipt->status.struct_size != sizeof(PopsComponentStatusV1) ||
            receipt->status.code != 0 ||
            receipt->status.action != POPS_COMPONENT_CONTINUE_V1)
          return 3;
        ++static_cast<WriterCallState*>(state)->publish_count;
        receipt->bytes_written =
            pops::component::field_point_count(
                request->fields[0].pieces[0].values) * sizeof(double);
        receipt->content_digest = "writer-v1";
        receipt->status = ok_status();
        return 0;
      },
      +[](void*, const PopsWriterRequestV1*) {},
      +[](void*, const PopsWriterRequestV1*) {}};
  const std::array<std::int64_t, 2> writer_lower{0, 0}, writer_upper{2, 2};
  const PopsWriterBoxV1 writer_box{
      sizeof(PopsWriterBoxV1), 2, writer_lower.data(), writer_upper.data()};
  const std::array<std::uint8_t, 4> valid_cells{1, 1, 1, 1};
  const std::array<std::uint8_t, 4> covered_cells{0, 0, 0, 0};
  const std::array<double, 4> cell_volumes{1.0, 1.0, 1.0, 1.0};
  const std::array<double, 2> writer_origin{0.0, 0.0}, writer_spacing{1.0, 1.0};
  const std::array<std::size_t, 2> writer_shape{2, 2};
  const PopsWriterGeometryV1 writer_geometry{
      sizeof(PopsWriterGeometryV1), "layout-v1", "uniform", 0, 2,
      writer_origin.data(), writer_spacing.data(), writer_shape.data(), 1, &writer_box,
      {sizeof(PopsConstByteViewV1), valid_cells.data(), valid_cells.size()},
      {sizeof(PopsConstByteViewV1), covered_cells.data(), covered_cells.size()},
      abi::const_field_view(cell_volumes.data(), 2, 2, 1, "layout-v1",
                            "geometry-patch")};
  const PopsWriterPieceV1 writer_piece{
      sizeof(PopsWriterPieceV1), 2, writer_lower.data(), writer_upper.data(),
      abi::const_field_view(solution.data(), 2, 2, 1, "layout-v1", "state-patch")};
  const char* component_names[] = {"u"};
  const PopsWriterFieldV1 writer_field{
      sizeof(PopsWriterFieldV1), "field-v1", "block::u", "manifest-v1",
      "layout-v1", 0, "accepted", "cell", "unspecified", 1,
      component_names, 2, writer_shape.data(), 1, &writer_piece};
  PopsWriterRequestV1 writer_request{
      sizeof(PopsWriterRequestV1), 1, &writer_geometry, 1, &writer_field,
      0, nullptr, "{}", "selection-v1", "temporary", "published",
      "snapshot-v1", abi::logical_time(), execution};
  PopsWriterReceiptV1 receipt{};
  receipt.struct_size = sizeof(PopsWriterReceiptV1);
  EXPECT_EQ(pops::component::publish_output(
                writer_api, &writer_state, writer_request, receipt), 0);
  EXPECT_EQ(writer_state.publish_count, 1);
  EXPECT_EQ(receipt.bytes_written, 4u * sizeof(double));
  writer_state.reject_verification = true;
  EXPECT_THROW(pops::component::publish_output(
                   writer_api, &writer_state, writer_request, receipt),
               std::runtime_error);
  EXPECT_EQ(writer_state.publish_count, 1);
}

}  // namespace
