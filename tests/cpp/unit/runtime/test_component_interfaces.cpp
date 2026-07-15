#include <pops/amr/tagging/tagging_truth.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/runtime/config/component_interfaces.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include "component_abi_test_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
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

TEST(ComponentInterfaces, SolveReportCarriesTypedIncompatibleRhsReason) {
  pops::SolveReport report;
  report.mark_failed(
      pops::SolveStatus::kIncompatibleRhs, pops::SolveAction::kRejectAttempt,
      "RHS violates the authenticated nullspace compatibility condition");
  EXPECT_TRUE(report.valid());
  EXPECT_FALSE(report.solved());
  EXPECT_STREQ(report.status_name(), "incompatible_rhs");
  EXPECT_STREQ(report.action_name(), "reject_attempt");
  EXPECT_EQ(report.reason,
            "RHS violates the authenticated nullspace compatibility condition");
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

TEST(ComponentInterfaces, TaggingTruthPreservesEqualityThroughLogicalRoots) {
  using pops::amr::TagTruth;
  const std::array any_unknown{TagTruth::False, TagTruth::Unknown};
  const std::array any_true{TagTruth::Unknown, TagTruth::True};
  const std::array all_unknown{TagTruth::True, TagTruth::Unknown};
  const std::array all_false{TagTruth::Unknown, TagTruth::False};

  EXPECT_EQ(pops::amr::tag_not(TagTruth::Unknown), TagTruth::Unknown);
  EXPECT_EQ(pops::amr::tag_not(TagTruth::True), TagTruth::False);
  EXPECT_EQ(pops::amr::tag_any(any_unknown.begin(), any_unknown.end()),
            TagTruth::Unknown);
  EXPECT_EQ(pops::amr::tag_any(any_true.begin(), any_true.end()), TagTruth::True);
  EXPECT_EQ(pops::amr::tag_all(all_unknown.begin(), all_unknown.end()),
            TagTruth::Unknown);
  EXPECT_EQ(pops::amr::tag_all(all_false.begin(), all_false.end()), TagTruth::False);
}

TEST(ComponentInterfaces, TaggingComparisonsRejectNonFiniteBeforeBooleanLogic) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)pops::amr::tag_comparison(nan, 0.0, true), std::domain_error);
  EXPECT_THROW((void)pops::amr::tag_comparison(nan, 0.0, false), std::domain_error);
  EXPECT_THROW(
      (void)pops::amr::tag_not(pops::amr::tag_comparison(nan, 0.0, true)),
      std::domain_error);
}

TEST(ComponentInterfaces, TaggingEqualityIsMappedBeforeConflictResolution) {
  using pops::amr::TagConflictPolicy;
  using pops::amr::TagEqualityPolicy;
  using pops::amr::TagTruth;

  const auto hold = pops::amr::resolve_tag_decision(
      TagTruth::Unknown, TagTruth::False, TagEqualityPolicy::Hold,
      TagConflictPolicy::Error);
  EXPECT_FALSE(hold.refine);
  EXPECT_FALSE(hold.coarsen);
  EXPECT_FALSE(hold.conflict_error);

  const auto refine = pops::amr::resolve_tag_decision(
      TagTruth::Unknown, TagTruth::False, TagEqualityPolicy::Refine,
      TagConflictPolicy::Error);
  EXPECT_TRUE(refine.refine);
  EXPECT_FALSE(refine.coarsen);

  const auto coarsen = pops::amr::resolve_tag_decision(
      TagTruth::False, TagTruth::Unknown, TagEqualityPolicy::Coarsen,
      TagConflictPolicy::Error);
  EXPECT_FALSE(coarsen.refine);
  EXPECT_TRUE(coarsen.coarsen);

  const auto equality_conflict = pops::amr::resolve_tag_decision(
      TagTruth::True, TagTruth::Unknown, TagEqualityPolicy::Coarsen,
      TagConflictPolicy::Error);
  EXPECT_TRUE(equality_conflict.conflict_error);
  const auto refine_wins = pops::amr::resolve_tag_decision(
      TagTruth::True, TagTruth::Unknown, TagEqualityPolicy::Coarsen,
      TagConflictPolicy::RefineWins);
  EXPECT_TRUE(refine_wins.refine);
  EXPECT_FALSE(refine_wins.coarsen);
}

PopsComponentTableHeaderV1 abi_header(
    std::size_t size, PopsNativeInterfaceIdV1 id, std::uint32_t version = 1) {
  return {static_cast<std::uint32_t>(size), POPS_COMPONENT_PROTOCOL_ABI_V1, id, version,
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
  // MPI_Comm_c2f/MPI_Type_c2f values are implementation-defined; zero is legal for a predefined
  // handle.  The exact identities, not the numeric values, select the distributed route.
  packed_distributed.communicator_f_handle = 0;
  packed_distributed.communicator_datatype_f_handle = 0;
  packed_distributed.communicator_identity = "MPI_COMM_WORLD";
  packed_distributed.communicator_datatype_identity = "MPI_DOUBLE";
  EXPECT_NO_THROW(pops::component::validate_execution_context(packed_distributed));
  auto ambiguous_distributed = packed_distributed;
  ambiguous_distributed.communicator_datatype_identity = "none";
  EXPECT_THROW(pops::component::validate_execution_context(ambiguous_distributed),
               std::invalid_argument);
  auto fabricated_distributed = packed_distributed;
  fabricated_distributed.communicator_identity = "mpi:custom";
  EXPECT_THROW(pops::component::validate_execution_context(fabricated_distributed),
               std::invalid_argument);
  auto hidden_serial_handle = execution;
  hidden_serial_handle.communicator_f_handle = 7;
  EXPECT_THROW(pops::component::validate_execution_context(hidden_serial_handle),
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
        const auto* state = static_cast<const double*>(request->states[0].values.data);
        for (std::size_t i = 0; i < request->refine_candidates.size; ++i)
          request->refine_candidates.data[i] = state[i] > request->program.leaves[0].threshold;
        *result = ok_status();
        return 0;
      }};
  const PopsQualifiedConstFieldV1 tag_states{
      sizeof(PopsQualifiedConstFieldV1), 1, "case::tag-state",
      abi::const_field_view(tag_values.data(), 2, 2)};
  const PopsTaggingLeafV1 tag_leaf{
      sizeof(PopsTaggingLeafV1), 0, 0, 1, 0.0, POPS_TAGGING_NO_STENCIL_V1};
  const std::int32_t tag_op = 1, tag_arg = 0;
  std::array<std::uint8_t, 4> coarsen{}, refine_equalities{}, coarsen_equalities{};
  PopsTaggerRequestV1 tag_request{
      sizeof(PopsTaggerRequestV1), 1, &tag_states,
      {sizeof(PopsTaggingProgramV1), "case::tag-program", 0, nullptr, 1, &tag_leaf,
       1, &tag_op, &tag_arg, 0, nullptr, nullptr, 0, 0, 0,
       POPS_TAGGING_NON_FINITE_REJECT_V1},
      {0, 0, 0}, {0, 0, 0}, {1, 1, 0}, {1.0, 1.0, 0.0}, 0,
      {sizeof(PopsByteViewV1), tags.data(), tags.size()},
      {sizeof(PopsByteViewV1), coarsen.data(), coarsen.size()},
      {sizeof(PopsByteViewV1), refine_equalities.data(), refine_equalities.size()},
      {sizeof(PopsByteViewV1), coarsen_equalities.data(), coarsen_equalities.size()},
      abi::logical_time(), execution};
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

  PopsClusteringApiV1 excessive_cluster_api{
      abi_header(sizeof(PopsClusteringApiV1), POPS_NATIVE_INTERFACE_CLUSTERING_V1),
      +[](void*, const PopsClusteringRequestV1* request, PopsComponentStatusV1* result) {
        *request->box_count = request->box_capacity + 1;
        *result = ok_status();
        return 0;
      }};
  box_count = 0;
  EXPECT_THROW(pops::component::cluster_tags(
                   excessive_cluster_api, nullptr, cluster_request, status),
               std::runtime_error);
  auto missing_cluster_api = cluster_api;
  missing_cluster_api.cluster = nullptr;
  EXPECT_THROW(pops::component::cluster_tags(
                   missing_cluster_api, nullptr, cluster_request, status),
               std::runtime_error);

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

  auto overflowing_ghosts = abi::const_field_view(tag_values.data(), 2, 2);
  overflowing_ghosts.ghost_lower[0] = std::numeric_limits<std::size_t>::max();
  overflowing_ghosts.ghost_upper[0] = 1;
  EXPECT_THROW(pops::component::validate_field_view(
                   overflowing_ghosts, "overflowing ghost test view"),
               std::invalid_argument);

  static constexpr PopsTopologyLabelV2 label_vocabulary[] = {
      {sizeof(PopsTopologyLabelV2), 1, "island-a", "test-topology"},
      {sizeof(PopsTopologyLabelV2), 2, "island-b", "test-topology"}};
  struct TopologyCallState {
    int calls = 0;
  } topology_calls;
  PopsFieldTopologyApiV2 topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2),
                 POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void* raw, const PopsFieldTopologyRequestV2* request,
          PopsFieldTopologyResultV2* result) {
        auto* state = static_cast<TopologyCallState*>(raw);
        if (++state->calls != 1 || request->topology.patch_count != 2 ||
            request->local_patch_count != 2 ||
            request->topology.periodic_axes != 1 ||
            std::strcmp(request->topology.topology_recipe_identity,
                        "test::topology-recipe") != 0)
          return 8;
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          if (patch.material_representation != POPS_FIELD_MATERIAL_FULL_V1 ||
              patch.material_coverage.data != nullptr ||
              patch.cut_cell_volume_fraction.data != nullptr ||
              patch.material_ids.data != nullptr || patch.material_mask.size != 2 ||
              patch.component_labels.size != 2)
            return 9;
          std::fill(patch.material_mask.data, patch.material_mask.data + 2, 1);
          std::fill(patch.component_labels.data, patch.component_labels.data + 2,
                    static_cast<std::int32_t>(local + 1));
        }
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "test-topology";
        result->topology_digest = "topology-v2";
        result->status = ok_status();
        return 0;
      }};
  PopsFieldTopologyApiV2 rejecting_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2),
                 POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2*,
          PopsFieldTopologyResultV2* result) {
        result->status = {sizeof(PopsComponentStatusV1), 17,
                          POPS_COMPONENT_REJECT_STEP_V1,
                          "topology rejected by test component"};
        return 0;
      }};
  PopsFieldTopologyApiV2 incomplete_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2),
                 POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2*,
          PopsFieldTopologyResultV2* result) {
        // A successful status is not permission to leave the caller-owned point outputs untouched.
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "incomplete-test-topology";
        result->topology_digest = "incomplete-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  static constexpr PopsTopologyLabelV2 undersized_label_vocabulary[] = {
      {0, 1, "island-a", "undersized-label-test-topology"}};
  PopsFieldTopologyApiV2 undersized_label_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2),
                 POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2* request,
          PopsFieldTopologyResultV2* result) {
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          std::fill(patch.material_mask.data,
                    patch.material_mask.data + patch.material_mask.size, 1);
          std::fill(patch.component_labels.data,
                    patch.component_labels.data + patch.component_labels.size, 1);
        }
        result->label_count = 1;
        result->labels = undersized_label_vocabulary;
        result->provenance = "undersized-label-test-topology";
        result->topology_digest = "undersized-label-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  PopsFieldTopologyApiV2 empty_full_topology_api{
      abi_header(sizeof(PopsFieldTopologyApiV2),
                 POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2),
      +[](void*, const PopsFieldTopologyRequestV2* request,
          PopsFieldTopologyResultV2* result) {
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          std::fill(patch.material_mask.data,
                    patch.material_mask.data + patch.material_mask.size, 0);
          std::fill(patch.component_labels.data,
                    patch.component_labels.data + patch.component_labels.size, 0);
        }
        result->label_count = 2;
        result->labels = label_vocabulary;
        result->provenance = "empty-full-test-topology";
        result->topology_digest = "empty-full-topology-v2";
        result->status = ok_status();
        return 0;
      }};
  std::string topology_layout =
      "test::owned-topology-layout-identity-longer-than-small-string-storage";
  std::array<std::string, 2> topology_patches{
      "test::owned-topology-patch-zero-longer-than-small-string-storage",
      "test::owned-topology-patch-one-longer-than-small-string-storage"};
  std::array<PopsFieldPatchMetadataV1, 2> metadata{};
  for (std::size_t index = 0; index < metadata.size(); ++index) {
    metadata[index] = {
        sizeof(PopsFieldPatchMetadataV1), index, 0, 0, 2, {}, {}, {}, {},
        POPS_FIELD_CENTERING_CELL_V1, 0, topology_layout.c_str(),
        topology_patches[index].c_str()};
    metadata[index].lower[0] = static_cast<std::int64_t>(2 * index);
    metadata[index].upper[0] = static_cast<std::int64_t>(2 * index + 1);
    metadata[index].lower[1] = metadata[index].upper[1] = 0;
    metadata[index].cell_spacing[0] = metadata[index].cell_spacing[1] = 0.25;
  }
  auto unrepresentable_patch = metadata[0];
  unrepresentable_patch.lower[0] = std::numeric_limits<std::int64_t>::min();
  unrepresentable_patch.upper[0] = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(pops::component::validate_field_patch_metadata(
                   unrepresentable_patch, 0),
               std::invalid_argument);
  const std::vector<pops::component::FieldTopologyPatchInputV2> topology_inputs{
      {0, POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}},
      {1, POPS_FIELD_MATERIAL_FULL_V1, {}, {}, {}},
  };
  std::array<std::uint8_t, 2> binary_coverage{1, 0};
  pops::component::FieldTopologyPatchInputV2 binary_input{
      0, POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1,
      {sizeof(PopsConstByteViewV1), binary_coverage.data(), binary_coverage.size()},
      {}, {}};
  EXPECT_EQ(pops::component::expected_topology_material_mask(binary_input, metadata[0]),
            (std::vector<std::uint8_t>{1, 0}));
  binary_coverage[1] = 2;
  EXPECT_THROW(
      (void)pops::component::expected_topology_material_mask(binary_input, metadata[0]),
      std::invalid_argument);
  binary_coverage[1] = 0;

  std::array<double, 2> cut_fractions{1.0, 0.0};
  pops::component::FieldTopologyPatchInputV2 cut_input{
      0, POPS_FIELD_MATERIAL_CUT_CELL_FRACTION_V1, {},
      abi::const_field_view(cut_fractions.data(), 2, 1, 1,
                            metadata[0].layout_identity,
                            metadata[0].patch_identity), {}};
  EXPECT_EQ(pops::component::expected_topology_material_mask(cut_input, metadata[0]),
            (std::vector<std::uint8_t>{1, 0}));
  cut_fractions[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      (void)pops::component::expected_topology_material_mask(cut_input, metadata[0]),
      std::invalid_argument);
  cut_fractions[1] = 0.0;
  PopsFieldGlobalTopologyV1 global_topology{
      sizeof(PopsFieldGlobalTopologyV1), "test::topology-recipe",
      topology_layout.c_str(), "test::materialized-layout", 2, {}, {}, 1,
      metadata.size(), metadata.data()};
  global_topology.domain_upper[0] = 3;
  const auto topology = [&] {
    EXPECT_THROW(pops::component::prepare_field_topology(
                     rejecting_topology_api, nullptr, global_topology, topology_inputs,
                     execution),
                 std::runtime_error);
    EXPECT_THROW(pops::component::prepare_field_topology(
                     incomplete_topology_api, nullptr, global_topology, topology_inputs,
                     execution),
                 std::runtime_error);
    EXPECT_THROW(pops::component::prepare_field_topology(
                     undersized_label_topology_api, nullptr, global_topology,
                     topology_inputs, execution),
                 std::runtime_error);
    EXPECT_THROW(pops::component::prepare_field_topology(
                     empty_full_topology_api, nullptr, global_topology, topology_inputs,
                     execution),
                 std::runtime_error);
    auto prepared = pops::component::prepare_field_topology(
        topology_api, &topology_calls, global_topology, topology_inputs, execution);
    std::fill(topology_layout.begin(), topology_layout.end(), 'x');
    for (auto& patch : topology_patches)
      std::fill(patch.begin(), patch.end(), 'y');
    return prepared;
  }();
  EXPECT_EQ(topology.topology_digest(), "topology-v2");
  ASSERT_EQ(topology.local_patches().size(), 2u);
  EXPECT_EQ(topology.local_patches()[0].component_labels,
            (std::vector<std::int32_t>{1, 1}));
  EXPECT_EQ(topology.local_patches()[1].component_labels,
            (std::vector<std::int32_t>{2, 2}));

  std::array<double, 2> rhs_a{1.0, 2.0}, rhs_b{3.0, 4.0};
  std::array<double, 2> solution_a{}, solution_b{};
  struct SolverCallState {
    int calls = 0;
  } solver_calls;
  PopsFieldSolverApiV2 solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void* raw, const PopsFieldSolverRequestV2* request, PopsSolveReportV2* report) {
        auto* state = static_cast<SolverCallState*>(raw);
        if (++state->calls != 1 || request->topology.patch_count != 2 ||
            request->local_patch_count != 2 ||
            request->topology_label_count != 2 || !request->topology_labels ||
            !request->topology_provenance ||
            std::strcmp(request->topology_provenance, "test-topology") != 0 ||
            std::strcmp(request->topology_digest, "topology-v2") != 0 ||
            std::strcmp(request->topology.topology_recipe_identity,
                        "test::topology-recipe") != 0)
          return 7;
        for (std::size_t index = 0; index < request->topology_label_count; ++index) {
          const auto& label = request->topology_labels[index];
          if (label.struct_size < sizeof(PopsFieldSolverTopologyLabelV2) ||
              label.id != static_cast<std::int32_t>(index + 1) || !label.label ||
              !label.provenance || std::strcmp(label.provenance, "test-topology") != 0)
            return 8;
        }
        for (std::size_t local = 0; local < request->local_patch_count; ++local) {
          const auto& patch = request->local_patches[local];
          if (patch.struct_size < sizeof(PopsFieldSolverPatchV2) ||
              patch.material_mask.size != 2 || patch.component_labels.size != 2)
            return 9;
          const auto* rhs_values = static_cast<const double*>(patch.rhs.data);
          auto* solution_values = static_cast<double*>(patch.solution.data);
          std::copy(rhs_values, rhs_values + 2, solution_values);
        }
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.0;
        report->reason = "tolerance reached";
        return 0;
      }};
  const auto& owned_metadata = topology.global_patches();
  const std::vector<pops::component::FieldSolverPatchBindingV2> solver_patches{
      {0,
       abi::const_field_view(rhs_a.data(), 2, 1, 1,
                             owned_metadata[0].layout_identity,
                             owned_metadata[0].patch_identity),
       abi::field_view(solution_a.data(), 2, 1, 1,
                       owned_metadata[0].layout_identity,
                       owned_metadata[0].patch_identity), {}},
      {1,
       abi::const_field_view(rhs_b.data(), 2, 1, 1,
                             owned_metadata[1].layout_identity,
                             owned_metadata[1].patch_identity),
       abi::field_view(solution_b.data(), 2, 1, 1,
                       owned_metadata[1].layout_identity,
                       owned_metadata[1].patch_identity), {}},
  };
  const auto solver_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution,
      "{\"identity\":\"test::boundary\"}",
      1e-8, 0.0, 10);
  auto omitted_vocabulary_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution,
      "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  const_cast<PopsFieldSolverRequestV2&>(omitted_vocabulary_request.request())
      .topology_labels = nullptr;
  PopsSolveReportV2 solve_report{};
  solve_report.struct_size = sizeof(PopsSolveReportV2);
  EXPECT_THROW(pops::component::solve_field(
                   solver_api, &solver_calls, omitted_vocabulary_request, solve_report),
               std::invalid_argument);
  auto substituted_digest_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution,
      "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  const_cast<PopsFieldSolverRequestV2&>(substituted_digest_request.request())
      .topology_digest = "substituted-topology";
  EXPECT_THROW(pops::component::solve_field(
                   solver_api, &solver_calls, substituted_digest_request, solve_report),
               std::invalid_argument);
  EXPECT_EQ(pops::component::solve_field(
                solver_api, &solver_calls, solver_request, solve_report), 0);
  EXPECT_EQ(solution_a, rhs_a);
  EXPECT_EQ(solution_b, rhs_b);
  EXPECT_EQ(solver_calls.calls, 1);
  auto topology_mutation_request = pops::component::bind_field_solver_request(
      topology, solver_patches, execution,
      "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10);
  PopsFieldSolverApiV2 topology_mutation_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2* request, PopsSolveReportV2* report) {
        auto* mask = const_cast<std::uint8_t*>(
            request->local_patches[0].material_mask.data);
        mask[0] = 0;
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.0;
        report->reason = "mutated topology";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(
                   topology_mutation_solver_api, nullptr,
                   topology_mutation_request, solve_report),
               std::runtime_error);
  EXPECT_EQ(topology.local_patches()[0].material_mask,
            (std::vector<std::uint8_t>{1, 1}));
  PopsFieldSolverApiV2 false_success_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 0.9;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 0.9;
        report->reason = "false success";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(
                   false_success_solver_api, nullptr, solver_request, solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 zero_forcing_false_success_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_SOLVED_V2;
        report->action = POPS_SOLVE_ACTION_NONE_V2;
        report->iterations = 1;
        report->relative_residual = 1.0e-12;
        report->reference_residual_norm = 0.0;
        report->residual_norm = 1.0e-12;
        report->reason = "false zero-reference success";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(
                   zero_forcing_false_success_api, nullptr, solver_request, solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 malformed_status_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = static_cast<PopsSolveStatusV2>(17);
        report->action = POPS_SOLVE_ACTION_REJECT_ATTEMPT_V2;
        report->iterations = 1;
        report->relative_residual = 1.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 1.0;
        report->reason = "malformed status";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(
                   malformed_status_solver_api, nullptr, solver_request, solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 incoherent_ratio_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_ITERATION_LIMIT_V2;
        report->action = POPS_SOLVE_ACTION_FAIL_RUN_V2;
        report->iterations = 10;
        report->relative_residual = 0.5;
        report->reference_residual_norm = 2.0;
        report->residual_norm = 0.2;
        report->reason = "ratio does not authenticate residual norms";
        return 0;
      }};
  EXPECT_THROW(pops::component::solve_field(
                   incoherent_ratio_solver_api, nullptr, solver_request, solve_report),
               std::runtime_error);
  PopsFieldSolverApiV2 incompatible_rhs_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2* report) {
        report->status = POPS_SOLVE_INCOMPATIBLE_RHS_V2;
        report->action = POPS_SOLVE_ACTION_FAIL_RUN_V2;
        report->iterations = 0;
        report->relative_residual = 1.0;
        report->reference_residual_norm = 1.0;
        report->residual_norm = 1.0;
        report->reason = "RHS is incompatible with the declared nullspace";
        return 0;
      }};
  EXPECT_EQ(pops::component::solve_field(
                incompatible_rhs_solver_api, nullptr, solver_request, solve_report), 0);
  EXPECT_EQ(solve_report.status, POPS_SOLVE_INCOMPATIBLE_RHS_V2);
  EXPECT_STREQ(solve_report.reason,
               "RHS is incompatible with the declared nullspace");
  PopsFieldSolverApiV2 transport_failure_solver_api{
      abi_header(sizeof(PopsFieldSolverApiV2), POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2),
      +[](void*, const PopsFieldSolverRequestV2*, PopsSolveReportV2*) { return 23; }};
  EXPECT_THROW(pops::component::solve_field(
                   transport_failure_solver_api, nullptr, solver_request, solve_report),
               std::runtime_error);
  std::array<double, 3> short_solution{};
  auto invalid_patches = solver_patches;
  invalid_patches[0].solution = abi::field_view(
      short_solution.data(), 1, 3, 1, owned_metadata[0].layout_identity,
      owned_metadata[0].patch_identity);
  EXPECT_THROW(
      pops::component::bind_field_solver_request(
          topology, invalid_patches, execution,
          "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10),
      std::invalid_argument);

  std::array<double, 4> ghosted_coefficients{};
  auto ghosted_patches = solver_patches;
  ghosted_patches[0].coefficients = abi::const_field_view(
      ghosted_coefficients.data(), 4, 1, 1,
      owned_metadata[0].layout_identity, owned_metadata[0].patch_identity);
  ghosted_patches[0].coefficients.ghost_lower[0] = 1;
  ghosted_patches[0].coefficients.ghost_upper[0] = 1;
  EXPECT_NO_THROW((void)pops::component::bind_field_solver_request(
      topology, ghosted_patches, execution,
      "{\"identity\":\"test::boundary\"}", 1e-8, 0.0, 10));

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
  const std::array<double, 4> writer_values{1.0, 2.0, 3.0, 4.0};
  const PopsWriterPieceV1 writer_piece{
      sizeof(PopsWriterPieceV1), 2, writer_lower.data(), writer_upper.data(),
      abi::const_field_view(writer_values.data(), 2, 2, 1, "layout-v1", "state-patch")};
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
