/// @file
/// @brief Exact compile-time-ranked AMR facade over runtime::amr::AmrRuntime<Dim>.

#include <pops/runtime/amr_system.hpp>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/berger_rigoutsos.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/amr/composite_reduction.hpp>
#include <pops/runtime/amr/persistent_tagging_state.hpp>
#include <pops/runtime/amr/prepared_component_providers.hpp>
#include <pops/runtime/analytic/collective_preflight.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/named_field_publication.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/external_riemann_brick.hpp>
#include <pops/runtime/program/module_metadata.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {
namespace {

std::string prepared_field_boundary_pair_key(const PreparedBoundaryComponentSpec& spec);
void require_prepared_field_boundary_pair(const PreparedBoundaryComponentSpec& residual,
                                          const PreparedBoundaryComponentSpec& jvp);

template <class Component>
void canonicalize_prepared_boundary_chain(std::vector<std::shared_ptr<Component>>& providers,
                                          const char* operation) {
  const auto less = [](const auto& left, const auto& right) {
    const PreparedBoundaryComponentSpec& lhs = left->spec();
    const PreparedBoundaryComponentSpec& rhs = right->spec();
    return std::tie(lhs.target_identity, lhs.component_id, lhs.manifest_identity,
                    lhs.interface_version, lhs.producer_identity, lhs.state_identity,
                    lhs.ghost_identity, lhs.layout_identity, lhs.region.kind, lhs.region.dimension,
                    lhs.region.codimension, lhs.region.axes, lhs.region.sides, lhs.region.identity,
                    lhs.states, lhs.directions, lhs.fields, lhs.parameter_ids, lhs.parameter_values,
                    lhs.outputs, lhs.rate, lhs.nonlinear_iterate, lhs.parameters_json,
                    lhs.target_json) <
           std::tie(rhs.target_identity, rhs.component_id, rhs.manifest_identity,
                    rhs.interface_version, rhs.producer_identity, rhs.state_identity,
                    rhs.ghost_identity, rhs.layout_identity, rhs.region.kind, rhs.region.dimension,
                    rhs.region.codimension, rhs.region.axes, rhs.region.sides, rhs.region.identity,
                    rhs.states, rhs.directions, rhs.fields, rhs.parameter_ids, rhs.parameter_values,
                    rhs.outputs, rhs.rate, rhs.nonlinear_iterate, rhs.parameters_json,
                    rhs.target_json);
  };
  std::sort(providers.begin(), providers.end(), less);
  if (std::adjacent_find(providers.begin(), providers.end(),
                         [&](const auto& left, const auto& right) {
                           return !less(left, right) && !less(right, left);
                         }) != providers.end())
    throw std::logic_error(std::string(operation) +
                           " provider chain contains one duplicate exact contract");
}

inline void require_amr_assembling(const runtime::system::SystemLifecycle& lifecycle,
                                   const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(std::string("AmrSystem::") + operation +
                             ": composition is frozen after bind");
}

struct NativeAmrPackageMetadata {
  std::string route_manifest;
  std::string parameter_names;
  int parameter_count = 0;
};

int native_amr_parameter_name_count(const char* raw) {
  if (raw == nullptr || *raw == '\0')
    return 0;
  int count = 1;
  for (const char* cursor = raw; *cursor != '\0'; ++cursor)
    if (*cursor == ',')
      ++count;
  return count;
}

NativeAmrPackageMetadata inspect_native_amr_package(pops::dynlib::handle handle,
                                                    const std::vector<double>& parameters) {
  constexpr const char* context = "AmrSystem::_install_native_block";
  const auto manifest = reinterpret_cast<const char* (*)()>(
      pops::dynlib::sym(handle, "pops_compiled_route_manifest"));
  const auto count =
      reinterpret_cast<int (*)()>(pops::dynlib::sym(handle, "pops_compiled_nparams"));
  const auto names =
      reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_compiled_param_names"));
  if (manifest == nullptr || count == nullptr || names == nullptr)
    throw std::runtime_error(std::string(context) +
                             ": strict package metadata is missing; rebuild the artifact");

  const char* route_manifest = manifest();
  const char* parameter_names = names();
  if (route_manifest == nullptr || parameter_names == nullptr)
    throw std::runtime_error(std::string(context) + ": package metadata returned null");
  pops::verify_route_manifest(route_manifest, context);

  const int parameter_count = count();
  if (parameter_count < 0 || parameter_count > kMaxRuntimeParams ||
      native_amr_parameter_name_count(parameter_names) != parameter_count ||
      parameters.size() != static_cast<std::size_t>(parameter_count))
    throw std::runtime_error(std::string(context) +
                             ": bound parameter vector disagrees with the exact package metadata");
  return {route_manifest, parameter_names, parameter_count};
}

std::string exact_amr_history_key(std::string_view name, int level) {
  if (name.empty() || level < 0)
    throw std::invalid_argument("AMR history key requires a name and non-negative level");
  return "pops.amr.level-history.v1/" + std::to_string(level) + "/" + std::to_string(name.size()) +
         ":" + std::string(name);
}

std::optional<std::pair<int, std::string>> decode_exact_amr_history_key(std::string_view key) {
  constexpr std::string_view prefix = "pops.amr.level-history.v1/";
  if (!key.starts_with(prefix))
    return std::nullopt;
  key.remove_prefix(prefix.size());
  const std::size_t level_end = key.find('/');
  const std::size_t length_end = key.find(':', level_end == std::string_view::npos ? 0 : level_end);
  if (level_end == std::string_view::npos || length_end == std::string_view::npos)
    throw std::invalid_argument("AMR history storage key is malformed");
  std::size_t consumed = 0;
  const int level = std::stoi(std::string(key.substr(0, level_end)), &consumed);
  if (consumed != level_end || level < 0)
    throw std::invalid_argument("AMR history storage key has an invalid level");
  const std::string length_text(key.substr(level_end + 1, length_end - level_end - 1));
  consumed = 0;
  const unsigned long long encoded_length = std::stoull(length_text, &consumed);
  if (consumed != length_text.size())
    throw std::invalid_argument("AMR history storage key has an invalid name length");
  const std::string name(key.substr(length_end + 1));
  if (encoded_length != name.size() || name.empty())
    throw std::invalid_argument("AMR history storage key has a truncated name");
  return std::pair<int, std::string>{level, std::move(name)};
}

template <int Dim>
void validate_amr_config(const AmrSystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "AmrSystem Cartesian core requires a dimension-qualified Cartesian provider");
  if (config.level_count < 1)
    throw std::invalid_argument("AmrSystem level_count must be positive");
  if (config.regrid_every < 0)
    throw std::invalid_argument("AmrSystem regrid_every must be non-negative");

  const std::size_t transitions = static_cast<std::size_t>(config.level_count - 1);
  if (config.transition_ratios.size() != transitions ||
      config.transition_buffers.size() != transitions ||
      config.transition_lookaheads.size() != transitions)
    throw std::invalid_argument(
        "AmrSystem transition tables must contain exactly level_count - 1 ranked rows");
  for (std::size_t transition = 0; transition < transitions; ++transition) {
    bool refines_any_axis = false;
    for (int axis = 0; axis < Dim; ++axis) {
      if (config.transition_ratios[transition][axis] < 1 ||
          config.transition_ratios[transition][axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "AmrSystem transition refinement ratios must be positive on every axis");
      refines_any_axis = refines_any_axis || config.transition_ratios[transition][axis] > 1;
      if (config.transition_buffers[transition][axis] < 0 ||
          config.transition_lookaheads[transition][axis] < 0)
        throw std::invalid_argument(
            "AmrSystem transition buffers and lookaheads must be non-negative");
      const auto reach = static_cast<std::uint64_t>(config.transition_buffers[transition][axis]) +
                         static_cast<std::uint64_t>(config.transition_lookaheads[transition][axis]);
      if (reach > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          reach > (static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) - 1u) / 2u)
        throw std::overflow_error(
            "AmrSystem transition buffer plus lookahead exceeds the ranked neighborhood budget");
    }
    if (!refines_any_axis)
      throw std::invalid_argument(
          "AmrSystem transition refinement ratio must refine at least one axis");
  }
  for (int axis = 0; axis < Dim; ++axis)
    if (config.coarse_max_grid[axis] < 0)
      throw std::invalid_argument("AmrSystem coarse_max_grid must be non-negative");
}

template <int Dim>
mesh::RankSpace<Dim> process_rank_space(const ExecutionLane& lane) {
  Extent<Dim> shape = runtime_config_detail::filled_extent<Dim>(1);
  shape[0] = lane.size();
  return mesh::RankSpace<Dim>(Index<Dim>{}, shape);
}

runtime::multiblock::BoundaryEvaluationPoint auxiliary_boundary_evaluation_point(
    const runtime::system::AuxiliaryEvaluationPoint& point) {
  if (point.accepted_step > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("AMR auxiliary accepted step exceeds boundary tick range");
  runtime::multiblock::BoundaryEvaluationPoint result;
  result.clock = point.clock;
  result.tick = static_cast<std::int64_t>(point.accepted_step);
  result.level = point.level;
  result.substep = point.substep;
  result.stage = point.stage;
  result.stage_fraction = {0, 1};
  return result;
}

EbThresholds resolved_amr_eb_thresholds(double kappa_min, double face_open_eps,
                                        double cut_theta_min) {
  if (!std::isfinite(kappa_min) || kappa_min < 0.0 || !std::isfinite(face_open_eps) ||
      face_open_eps < 0.0 || !std::isfinite(cut_theta_min) || cut_theta_min < 0.0)
    throw std::invalid_argument(
        "AMR embedded-boundary threshold overrides must be finite and non-negative");
  EbThresholds result;
  if (kappa_min > 0.0)
    result.kappa_min = static_cast<Real>(kappa_min);
  if (face_open_eps > 0.0)
    result.face_open_eps = static_cast<Real>(face_open_eps);
  if (cut_theta_min > 0.0)
    result.cut_theta_min = static_cast<Real>(cut_theta_min);
  return result;
}

std::uint64_t next_amr_eb_generation(std::uint64_t current) {
  if (current == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem embedded-boundary generation overflow");
  return current + 1;
}

std::string prefixed_sha256(std::string_view prefix, std::string_view contract) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(contract.size());
  for (const char value : contract)
    bytes.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(value)));
  return std::string(prefix) + identity::sha256_hex(bytes);
}

std::vector<int> resolved_composite_levels(std::size_t count, const std::vector<int>& requested) {
  std::vector<int> levels = requested;
  if (levels.empty()) {
    levels.reserve(count);
    for (std::size_t level = 0; level < count; ++level)
      levels.push_back(static_cast<int>(level));
  }
  int previous = -1;
  for (const int level : levels) {
    if (level < 0 || static_cast<std::size_t>(level) >= count || level <= previous)
      throw std::invalid_argument(
          "AMR composite levels must be strictly increasing live hierarchy indices");
    previous = level;
  }
  return levels;
}

runtime::amr::CompositeReductionKind composite_reduction_kind(std::string_view kind) {
  if (kind == "sum")
    return runtime::amr::CompositeReductionKind::Sum;
  if (kind == "abs_sum")
    return runtime::amr::CompositeReductionKind::AbsoluteSum;
  if (kind == "sum_sq")
    return runtime::amr::CompositeReductionKind::SumSquares;
  if (kind == "min")
    return runtime::amr::CompositeReductionKind::Minimum;
  if (kind == "max")
    return runtime::amr::CompositeReductionKind::Maximum;
  if (kind == "abs_max")
    return runtime::amr::CompositeReductionKind::AbsoluteMaximum;
  throw std::invalid_argument("AMR composite reduction kind is unknown: " + std::string(kind));
}

struct PreparedAmrEbAuthoring {
  std::string configuration_contract;
  std::string semantic_digest;
};

template <int Dim>
PreparedAmrEbAuthoring prepare_amr_eb_authoring(const AmrSystemConfig<Dim>& config,
                                                const PreparedAmrSystemBlock<Dim>& block,
                                                const std::vector<std::string>& opcodes,
                                                const std::vector<double>& literals,
                                                runtime::system::PreparedEmbeddedBoundaryMode mode,
                                                const EbThresholds& thresholds,
                                                std::uint64_t generation) {
  PreparedAmrEbAuthoring result;
  if (mode == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell &&
      block.cut_cell_provider_identity.empty())
    throw std::invalid_argument("AMR cut-cell transport requires an authenticated provider");
  if (mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive &&
      std::any_of(config.periodicity.begin(), config.periodicity.end(),
                  [](bool periodic) { return !periodic; }))
    throw std::invalid_argument(
        "active AMR embedded transport has no EB-qualified physical-boundary provider");
  if (!std::isfinite(thresholds.kappa_min) || !(thresholds.kappa_min > Real(0)) ||
      thresholds.kappa_min > Real(1) || !std::isfinite(thresholds.face_open_eps) ||
      thresholds.face_open_eps < Real(0) || thresholds.face_open_eps > Real(1) ||
      !std::isfinite(thresholds.cut_theta_min) || !(thresholds.cut_theta_min > Real(0)) ||
      thresholds.cut_theta_min > Real(1))
    throw std::invalid_argument("AMR embedded-boundary thresholds are outside their unit ranges");
  std::vector<analytic::AnalyticProgram> programs =
      analytic::compile_component_programs({opcodes}, {literals});
  if (programs.size() != 1)
    throw std::logic_error("AMR embedded-boundary authoring did not compile one scalar program");
  (void)analytic::make_analytic_level_set<Dim>(programs.front());

  ExactContractBuilder semantic_builder;
  semantic_builder.text("pops.amr.embedded-boundary-semantic")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(runtime::system::prepared_embedded_boundary_mode_name(mode))
      .scalar(thresholds.kappa_min)
      .scalar(thresholds.face_open_eps)
      .scalar(thresholds.cut_theta_min)
      .text(block.staircase_provider_identity)
      .text(block.cut_cell_provider_identity)
      .scalar(static_cast<std::uint64_t>(opcodes.size()));
  for (const std::string& opcode : opcodes)
    semantic_builder.text(opcode);
  semantic_builder.scalar(static_cast<std::uint64_t>(literals.size()));
  for (const double literal : literals)
    semantic_builder.scalar(literal);
  for (int axis = 0; axis < Dim; ++axis)
    semantic_builder.scalar(config.index_domain().lo[axis])
        .scalar(config.index_domain().hi[axis])
        .scalar(config.lower[axis])
        .scalar(config.upper[axis])
        .scalar(config.periodicity[axis]);
  std::string semantic = std::move(semantic_builder).release();
  result.semantic_digest = prefixed_sha256("pops.prepared-eb-semantic.v1:sha256:", semantic);
  ExactContractBuilder configuration;
  configuration.text("pops.amr.embedded-boundary-configuration")
      .scalar(std::uint32_t{1})
      .bytes(semantic)
      .scalar(generation);
  result.configuration_contract = std::move(configuration).release();
  return result;
}

template <int Dim>
std::int64_t checked_layout_cells(const mesh::BoxArray<Dim>& layout) {
  std::int64_t total = 0;
  for (const Box<Dim>& box : layout.boxes()) {
    const std::int64_t cells = box.numPts();
    if (cells < 1 || cells > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("AmrSystem layout cell budget exceeds int64_t");
    total += cells;
  }
  return total;
}

inline std::size_t checked_pair_count(std::size_t count) {
  if (count > 1 && count - 1 > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem layout pair budget exceeds size_t");
  return count < 2 ? 0 : count * (count - 1) / 2;
}

inline std::size_t checked_square_count(std::size_t count) {
  if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem hierarchy pair budget exceeds size_t");
  return count * count;
}

inline std::size_t checked_size_sum(std::size_t left, std::size_t right, const char* operation) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(operation);
  return left + right;
}

inline std::size_t checked_size_product(std::size_t left, std::size_t right,
                                        const char* operation) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("AmrSystem ranked field exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear);

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box);

template <int Dim>
amr::RefinementRatio<Dim> refinement_ratio(const Extent<Dim>& authored) {
  std::array<int, Dim> values{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (authored[axis] < 1 || authored[axis] > std::numeric_limits<int>::max())
      throw std::invalid_argument("AMR transition ratio exceeds the exact ranked index range");
    values[static_cast<std::size_t>(axis)] = static_cast<int>(authored[axis]);
  }
  return amr::RefinementRatio<Dim>(values);
}

template <int Dim>
amr::tagging::TagMaskBudget exact_tag_mask_budget(const amr::hierarchy::LevelLayout<Dim>& layout,
                                                  const Index<Dim>& rank) {
  const std::vector<std::size_t> local = layout.distribution().local_box_indices(rank);
  std::size_t maximum_patch_cells = 0;
  std::size_t owned_cells = 0;
  for (const std::size_t global : local) {
    const std::size_t cells = checked_cells(layout.patches()[global]);
    maximum_patch_cells = std::max(maximum_patch_cells, cells);
    owned_cells = checked_size_sum(owned_cells, cells, "AMR tag-mask owned cells exceed size_t");
  }
  std::size_t identity_bytes = checked_size_product(layout.patches().size(), sizeof(Box<Dim>),
                                                    "AMR tag-mask layout identity exceeds size_t");
  identity_bytes = checked_size_sum(
      identity_bytes,
      checked_size_product(layout.distribution().owners().size(), sizeof(Index<Dim>),
                           "AMR tag-mask ownership identity exceeds size_t"),
      "AMR tag-mask identity exceeds size_t");
  identity_bytes = checked_size_sum(
      identity_bytes,
      checked_size_product(local.size(), sizeof(amr::tagging::PatchTagIdentity<Dim>),
                           "AMR tag-mask patch identity exceeds size_t"),
      "AMR tag-mask identity exceeds size_t");
  identity_bytes =
      checked_size_sum(identity_bytes, owned_cells, "AMR tag-mask identity exceeds size_t");
  return {layout.patches().size(), local.size(), maximum_patch_cells, owned_cells, owned_cells,
          identity_bytes};
}

template <int Dim>
runtime::amr::PreparedTaggingExecutionBudget exact_tagging_budget(
    const amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& rank) {
  const auto mask = exact_tag_mask_budget(layout, rank);
  const std::size_t consensus =
      layout.distribution().replicated()
          ? checked_size_product(mask.owned_cells, 2, "AMR replicated tag consensus exceeds size_t")
          : 0;
  return {mask, mask.owned_cells, consensus};
}

template <int Dim>
std::vector<char> gather_tag_mask(const amr::tagging::TagMask<Dim>& mask, const Box<Dim>& domain,
                                  const CommunicatorView& communicator) {
  std::vector<char> global(checked_cells(domain), char{0});
  const bool contributes =
      mask.level_identity().distribution_mode != mesh::DistributionMode::replicated ||
      communicator.rank() == 0;
  if (contributes)
    for (const auto& patch : mask.patches())
      for (std::size_t ordinal = 0; ordinal < patch.tags.size(); ++ordinal)
        if (patch.tags[ordinal] != 0)
          global[offset(unflatten(patch.box, ordinal), domain)] = char{1};
  all_reduce_max_inplace(global.data(), global.size(), communicator);
  return global;
}

template <int Dim>
struct SparseFieldImage {
  Box<Dim> domain{};
  int components = 0;
  std::vector<double> values{};
  std::vector<char> populated{};
};

template <int Dim>
SparseFieldImage<Dim> gather_sparse_field(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                          const CommunicatorView& communicator) {
  const std::size_t cells = checked_cells(domain);
  SparseFieldImage<Dim> result{domain, field.ncomp(), {}, std::vector<char>(cells, char{0})};
  result.values.assign(checked_size_product(static_cast<std::size_t>(field.ncomp()), cells,
                                            "AMR transfer image exceeds size_t"),
                       0.0);
  const bool contributes = !field.distribution().replicated() || communicator.rank() == 0;
  if (contributes)
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const Fab<Dim>& fab = field.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const Box<Dim>& valid = fab.box();
      const Box<Dim>& grown = fab.grown_box();
      const std::size_t component_stride = checked_cells(grown);
      for (std::size_t ordinal = 0; ordinal < checked_cells(valid); ++ordinal) {
        const Index<Dim> index = unflatten(valid, ordinal);
        const std::size_t global = offset(index, domain);
        result.populated[global] = char{1};
        for (int component = 0; component < field.ncomp(); ++component)
          result.values[static_cast<std::size_t>(component) * cells + global] = static_cast<double>(
              host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)));
      }
    }
  all_reduce_sum_inplace(result.values.data(), result.values.size(), communicator);
  all_reduce_max_inplace(result.populated.data(), result.populated.size(), communicator);
  return result;
}

template <int Dim>
struct PreparedRegriddedStateTransfer {
  using host_mirror_type = typename Fab<Dim>::host_mirror_type;

  MultiFab<Dim> child;
  Fab<Dim> dense_parent;
  host_mirror_type dense_host;
  std::vector<double> values;
  std::vector<char> populated;
  std::vector<amr::transfer::PreparedTransfer<Dim>> kernels;
  std::vector<host_mirror_type> child_hosts;
  const SparseFieldImage<Dim>* previous_child = nullptr;
  std::size_t old_cells = 0;

  PreparedRegriddedStateTransfer(const MultiFab<Dim>& parent,
                                 const amr::hierarchy::LevelLayout<Dim>& parent_layout,
                                 const amr::hierarchy::LevelLayout<Dim>& child_layout,
                                 int source_radius,
                                 const std::optional<SparseFieldImage<Dim>>& previous)
      : child(child_layout.patches(), child_layout.distribution(), parent.local_rank(),
              parent.ncomp(), parent.ghosts()),
        dense_parent(parent_layout.domain(), parent.ncomp(),
                     [&] {
                       Extent<Dim> ghosts{};
                       for (int axis = 0; axis < Dim; ++axis)
                         ghosts[axis] = source_radius;
                       return ghosts;
                     }()),
        dense_host(dense_parent.create_host_mirror()),
        values(checked_size_product(static_cast<std::size_t>(parent.ncomp()),
                                    checked_cells(dense_parent.grown_box()),
                                    "AMR linear transfer source exceeds size_t"),
               0.0),
        populated(checked_cells(dense_parent.grown_box()), char{0}),
        previous_child(previous && previous->domain == child_layout.domain() ? &*previous
                                                                             : nullptr),
        old_cells(previous_child != nullptr ? checked_cells(previous_child->domain)
                                            : std::size_t{0}) {
    child.set_val(Real(0));
  }
};

template <int Dim>
std::unique_ptr<PreparedRegriddedStateTransfer<Dim>> prepare_regridded_state_transfer(
    const MultiFab<Dim>& field, const amr::hierarchy::LevelLayout<Dim>& parent_layout,
    const amr::hierarchy::LevelLayout<Dim>& child_layout,
    const std::optional<SparseFieldImage<Dim>>& previous_child,
    amr::transfer::TransferKind transfer_kind, int collective_rank) {
  const int source_radius = transfer_kind == amr::transfer::TransferKind::ConstantInjection ? 0 : 1;
  if (source_radius < 0 || source_radius > 1)
    throw std::invalid_argument("AMR prepared transfer requested an unsupported source radius");
  Extent<Dim> required_ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    required_ghosts[axis] = source_radius;
    if (field.ghosts()[axis] < required_ghosts[axis])
      throw std::invalid_argument("AMR prepared transfer lacks its exact parent ghost stencil");
  }
  auto prepared = std::make_unique<PreparedRegriddedStateTransfer<Dim>>(
      field, parent_layout, child_layout, source_radius, previous_child);
  const Box<Dim>& dense_box = prepared->dense_parent.grown_box();
  const std::size_t dense_cells = prepared->populated.size();

  const bool replicated = field.distribution().replicated();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const std::size_t global = field.global_index(local);
    if (replicated && collective_rank != 0)
      continue;
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t ordinal = 0; ordinal < dense_cells; ++ordinal) {
      const Index<Dim> index = unflatten(dense_box, ordinal);
      std::size_t selected = parent_layout.patches().size();
      for (std::size_t patch = 0; patch < parent_layout.patches().size(); ++patch)
        if (parent_layout.patches()[patch].contains(index)) {
          selected = patch;
          break;
        }
      if (selected == parent_layout.patches().size())
        for (std::size_t patch = 0; patch < parent_layout.patches().size(); ++patch) {
          Box<Dim> patch_grown = parent_layout.patches()[patch];
          for (int axis = 0; axis < Dim; ++axis)
            patch_grown = patch_grown.grow(axis, static_cast<int>(field.ghosts()[axis]));
          if (patch_grown.contains(index)) {
            selected = patch;
            break;
          }
        }
      if (selected != global || !grown.contains(index))
        continue;
      prepared->populated[ordinal] = char{1};
      for (int component = 0; component < field.ncomp(); ++component)
        prepared->values[static_cast<std::size_t>(component) * dense_cells + ordinal] =
            static_cast<double>(host(static_cast<std::size_t>(component) * component_stride +
                                     offset(index, grown)));
    }
  }
  const amr::RefinementRatio<Dim>& ratio = child_layout.ratio_from_parent();
  amr::transfer::IndexMapping<Dim> mapping;
  mapping.coarse_origin = parent_layout.domain().lo;
  mapping.fine_origin = child_layout.domain().lo;
  const amr::transfer::ComponentRange components{0, 0, field.ncomp()};
  const amr::transfer::TransferProvider<Dim, amr::transfer::Centering::Cell> provider(
      transfer_kind);
  prepared->kernels.reserve(prepared->child.local_size());
  prepared->child_hosts.reserve(prepared->previous_child != nullptr ? prepared->child.local_size()
                                                                    : 0);
  for (std::size_t local = 0; local < prepared->child.local_size(); ++local) {
    Fab<Dim>& fab = prepared->child.fab(local);
    prepared->kernels.push_back(provider.prepare(std::as_const(prepared->dense_parent).view(),
                                                 fab.view(), fab.box(), ratio, mapping,
                                                 components));
    if (prepared->previous_child != nullptr)
      prepared->child_hosts.push_back(fab.create_host_mirror());
  }
  return prepared;
}

template <int Dim>
void execute_regridded_state_transfer(PreparedRegriddedStateTransfer<Dim>& prepared,
                                      const CommunicatorView& communicator) {
  all_reduce_sum_inplace(prepared.values.data(), prepared.values.size(), communicator);
  all_reduce_max_inplace(prepared.populated.data(), prepared.populated.size(), communicator);
  if (std::any_of(prepared.populated.begin(), prepared.populated.end(),
                  [](char value) { return value == 0; }))
    throw std::runtime_error(
        "AMR prepared transfer source ghosts were not materialized collectively");
  for (std::size_t index = 0; index < prepared.values.size(); ++index)
    prepared.dense_host(index) = static_cast<Real>(prepared.values[index]);
  prepared.dense_parent.copy_from_host(prepared.dense_host);
  for (std::size_t local = 0; local < prepared.child.local_size(); ++local)
    for_each_cell(prepared.child.fab(local).box(), prepared.kernels[local]);
  device_fence();

  if (prepared.previous_child == nullptr)
    return;
  for (std::size_t local = 0; local < prepared.child.local_size(); ++local) {
    Fab<Dim>& fab = prepared.child.fab(local);
    auto& host = prepared.child_hosts[local];
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t ordinal = 0; ordinal < checked_cells(valid); ++ordinal) {
      const Index<Dim> fine = unflatten(valid, ordinal);
      const std::size_t fine_global = offset(fine, prepared.previous_child->domain);
      const bool reuse = prepared.previous_child->populated[fine_global] != 0;
      if (!reuse)
        continue;
      for (int component = 0; component < prepared.child.ncomp(); ++component) {
        const double value =
            prepared.previous_child
                ->values[static_cast<std::size_t>(component) * prepared.old_cells + fine_global];
        host(static_cast<std::size_t>(component) * component_stride + offset(fine, grown)) =
            static_cast<Real>(value);
      }
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
MultiFab<Dim> transfer_regridded_state(const MultiFab<Dim>& parent,
                                       const amr::hierarchy::LevelLayout<Dim>& parent_layout,
                                       const amr::hierarchy::LevelLayout<Dim>& child_layout,
                                       const std::optional<SparseFieldImage<Dim>>& previous_child,
                                       const CommunicatorView& communicator,
                                       amr::transfer::TransferKind transfer_kind) {
  std::unique_ptr<PreparedRegriddedStateTransfer<Dim>> prepared;
  std::exception_ptr local_error;
  try {
    prepared = prepare_regridded_state_transfer(parent, parent_layout, child_layout, previous_child,
                                                transfer_kind, communicator.rank());
  } catch (...) {
    local_error = std::current_exception();
  }
  const long preparation_failures = all_reduce_sum(local_error ? 1L : 0L, communicator);
  if (preparation_failures != 0) {
    if (communicator.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR transfer preparation failed collectively on " +
                             std::to_string(preparation_failures) + " rank(s)");
  }
  local_error = {};
  try {
    execute_regridded_state_transfer(*prepared, communicator);
  } catch (...) {
    local_error = std::current_exception();
  }
  const long execution_failures = all_reduce_sum(local_error ? 1L : 0L, communicator);
  if (execution_failures != 0) {
    if (communicator.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR transfer execution failed collectively on " +
                             std::to_string(execution_failures) + " rank(s)");
  }
  static_assert(std::is_nothrow_move_constructible_v<MultiFab<Dim>>);
  return std::move(prepared->child);
}

inline std::string direct_amr_state_identity(std::string_view block) {
  return "pops://runtime/amr-direct-state/" + std::to_string(block.size()) + ":" +
         std::string(block);
}

template <int Dim>
bool layout_contains(const amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& index) {
  for (const Box<Dim>& patch : layout.patches().boxes())
    if (patch.contains(index))
      return true;
  return false;
}

template <int Dim>
std::vector<char> child_coverage_on_parent(
    const amr::hierarchy::LevelLayout<Dim>& parent,
    const std::optional<amr::hierarchy::LevelLayout<Dim>>& child) {
  std::vector<char> covered(checked_cells(parent.domain()), char{0});
  if (!child)
    return covered;
  for (const Box<Dim>& fine : child->patches().boxes()) {
    const Box<Dim> footprint = amr::hierarchy::coarsen_box(fine, child->ratio_from_parent());
    for (std::size_t ordinal = 0; ordinal < checked_cells(footprint); ++ordinal)
      covered[offset(unflatten(footprint, ordinal), parent.domain())] = char{1};
  }
  return covered;
}

template <int Dim>
amr::tagging::ClusterOptions<Dim> exact_cluster_options(
    const AmrSystemConfig<Dim>& config, const amr::hierarchy::LevelLayout<Dim>& parent) {
  const std::size_t cells = checked_cells(parent.domain());
  const std::size_t patches = parent.patches().size();
  const std::size_t ranks = parent.distribution().rank_space().size();
  const std::size_t nodes = checked_size_sum(
      checked_size_product(std::max<std::size_t>(cells, 1), static_cast<std::size_t>(2 * Dim),
                           "AMR clustering recursion budget exceeds size_t"),
      1, "AMR clustering recursion budget exceeds size_t");
  const std::size_t visits = checked_size_product(
      std::max<std::size_t>(cells, 1), nodes, "AMR clustering cell-visit budget exceeds size_t");
  if (visits > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::length_error("AMR clustering cell-visit budget exceeds exact signed counters");
  const auto add_identity = [](std::size_t total, std::size_t count, std::size_t width) {
    return checked_size_sum(
        total, checked_size_product(count, width, "AMR clustering identity budget exceeds size_t"),
        "AMR clustering identity budget exceeds size_t");
  };
  std::size_t identity_bytes =
      std::string_view{amr::tagging::BergerRigoutsosProvider<Dim>::kIdentity}.size();
  identity_bytes = add_identity(identity_bytes, patches, sizeof(Box<Dim>));
  identity_bytes =
      add_identity(identity_bytes, parent.distribution().owners().size(), sizeof(Index<Dim>));
  identity_bytes = add_identity(identity_bytes, cells, sizeof(Box<Dim>));
  identity_bytes = add_identity(identity_bytes, ranks, sizeof(amr::tagging::TagShardIdentity<Dim>));
  identity_bytes =
      add_identity(identity_bytes, patches, sizeof(amr::tagging::PatchTagIdentity<Dim>));
  identity_bytes = checked_size_sum(
      identity_bytes, static_cast<std::size_t>(checked_layout_cells(parent.patches())),
      "AMR clustering identity budget exceeds size_t");
  amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency =
      config.cluster_min_efficiency > 0.0 ? config.cluster_min_efficiency : 0.7;
  const int minimum = config.cluster_min_box_size > 0 ? config.cluster_min_box_size : 1;
  const int maximum = config.cluster_max_box_size > 0 ? config.cluster_max_box_size : 32;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = minimum;
    options.max_box_size[static_cast<std::size_t>(axis)] = maximum;
  }
  options.budget = {ranks, nodes, visits, std::max<std::size_t>(cells, 1),
                    std::max<std::size_t>(identity_bytes, 1)};
  return options;
}

template <int Dim>
amr::regridding::RegridPreparationBudget exact_regrid_budget(
    const amr::hierarchy::LevelLayout<Dim>& parent, const amr::RefinementRatio<Dim>& ratio,
    const amr::tagging::ClusterResult<Dim>& clustered) {
  std::int64_t fine_cells = 0;
  for (const Box<Dim>& box : clustered.boxes.boxes()) {
    const std::int64_t cells = amr::hierarchy::refine_box(box, ratio).numPts();
    if (cells < 0 || cells > std::numeric_limits<std::int64_t>::max() - fine_cells)
      throw std::length_error("AMR regrid load-balance weight exceeds int64_t");
    fine_cells += cells;
  }
  const std::size_t boxes = clustered.boxes.size();
  const mesh::BoxArrayValidationBudget layout_budget{boxes, checked_pair_count(boxes)};
  return {
      layout_budget, layout_budget, {boxes, parent.distribution().rank_space().size(), fine_cells}};
}

template <int Dim>
std::size_t exact_hierarchy_pair_budget(const AmrSystemConfig<Dim>& config,
                                        std::size_t coarse_patches) {
  Box<Dim> parent_domain = config.index_domain();
  std::size_t parent_patch_bound = coarse_patches;
  std::size_t pairs = 0;
  for (std::size_t transition = 0; transition < static_cast<std::size_t>(config.level_count - 1);
       ++transition) {
    const std::size_t child_patch_bound = checked_cells(parent_domain);
    pairs = checked_size_sum(
        pairs,
        checked_size_product(parent_patch_bound, child_patch_bound,
                             "AMR hierarchy parent/child pair budget exceeds size_t"),
        "AMR hierarchy parent/child pair budget exceeds size_t");
    parent_patch_bound = child_patch_bound;
    parent_domain = amr::hierarchy::refine_box(
        parent_domain, refinement_ratio(config.transition_ratios[transition]));
  }
  return pairs;
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(linear % extent);
    linear /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
std::vector<double> gather_field(const MultiFab<Dim>& field, const Box<Dim>& domain, int components,
                                 const ExecutionLane* lane = nullptr) {
  if (components < 1 || components > field.ncomp())
    throw std::invalid_argument("AmrSystem gather component count is invalid");
  const std::size_t domain_cells = checked_cells(domain);
  if (static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / domain_cells)
    throw std::overflow_error("AmrSystem gather buffer exceeds size_t");
  std::vector<double> result(static_cast<std::size_t>(components) * domain_cells, 0.0);
  const int rank = lane == nullptr ? my_rank() : lane->rank();
  const bool contributes = !field.distribution().replicated() || rank == 0;
  for (std::size_t local = 0; contributes && local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        result[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)] =
            static_cast<double>(host(static_cast<std::size_t>(component) * component_stride +
                                     offset(index, grown)));
      }
  }
  if (lane == nullptr)
    all_reduce_sum_inplace(result.data(), result.size());
  else
    all_reduce_sum_inplace(result.data(), result.size(), *lane);
  return result;
}

template <int Dim>
bool finite_valid_field_local(const MultiFab<Dim>& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        if (!std::isfinite(static_cast<double>(host(
                static_cast<std::size_t>(component) * component_stride + offset(index, grown)))))
          return false;
      }
  }
  return true;
}

template <int Dim>
void write_field(MultiFab<Dim>& field, const Box<Dim>& domain, const std::vector<double>& values,
                 int components) {
  const std::size_t domain_cells = checked_cells(domain);
  if (components < 1 || components > field.ncomp() ||
      values.size() != static_cast<std::size_t>(components) * domain_cells)
    throw std::invalid_argument("AmrSystem field input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
            static_cast<Real>(
                values[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)]);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void write_component(MultiFab<Dim>& field, const Box<Dim>& domain,
                     const std::vector<double>& values, int component) {
  const std::size_t domain_cells = checked_cells(domain);
  if (component < 0 || component >= field.ncomp() || values.size() != domain_cells)
    throw std::invalid_argument("AmrSystem component input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t linear = 0; linear < local_cells; ++linear) {
      const Index<Dim> index = unflatten(valid, linear);
      host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
          static_cast<Real>(values[offset(index, domain)]);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
double cell_measure(const AmrSystemConfig<Dim>& config, const Box<Dim>& domain) {
  double measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
               static_cast<double>(domain.length(axis));
  return measure;
}

template <int Dim>
std::size_t periodic_image_bound(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                 const BoundaryTopology<Dim>& topology) {
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t axis_images = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && ghosts[axis] > 0) {
      const std::int64_t length = domain.length(axis);
      if (length <= 0)
        throw std::invalid_argument("AMR halo budget requires a non-empty periodic domain");
      const std::int64_t wraps = 1 + (ghosts[axis] - 1) / length;
      if (wraps > static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
        throw std::length_error("AMR halo periodic-image budget exceeds size_t");
      axis_images = 1 + 2 * static_cast<std::size_t>(wraps);
    }
    images =
        checked_size_product(images, axis_images, "AMR halo periodic-image product exceeds size_t");
  }
  return images;
}

template <int Dim>
std::size_t grown_layout_elements(const mesh::BoxArray<Dim>& layout, const Extent<Dim>& ghosts,
                                  int ncomp) {
  if (ncomp < 1)
    throw std::invalid_argument("AMR halo element budget requires positive components");
  std::size_t cells = 0;
  for (const Box<Dim>& valid : layout.boxes()) {
    Box<Dim> grown = valid;
    for (int axis = 0; axis < Dim; ++axis) {
      if (ghosts[axis] < 0 || ghosts[axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument("AMR halo ghost budget exceeds native coordinates");
      grown = grown.grow(axis, static_cast<int>(ghosts[axis]));
    }
    cells =
        checked_size_sum(cells, checked_cells(grown), "AMR halo grown-cell budget exceeds size_t");
  }
  return checked_size_product(cells, static_cast<std::size_t>(ncomp),
                              "AMR halo element budget exceeds size_t");
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                     const BoundaryTopology<Dim>& topology) {
  const std::size_t patches = field.layout().size();
  const std::size_t images = periodic_image_bound(domain, field.ghosts(), topology);
  const std::size_t patch_pairs =
      checked_size_product(patches, patches, "AMR halo patch-pair budget exceeds size_t");
  const std::size_t work =
      checked_size_product(patch_pairs, images, "AMR halo box-image budget exceeds size_t");
  const std::size_t jobs = checked_size_product(work, static_cast<std::size_t>(2 * Dim),
                                                "AMR halo job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(grown_layout_elements(field.layout(), field.ghosts(), field.ncomp()),
                           std::max<std::size_t>(patches, 1),
                           "AMR halo patch-element budget exceeds size_t"),
      images, "AMR halo image-element budget exceeds size_t");
  return HaloScheduleBudget{
      mesh::BoxArrayValidationBudget{patches, checked_pair_count(patches)},
      work,
      jobs,
      images,
      field.rank_space().size(),
      elements,
      elements,
      elements,
  };
}

template <int Dim>
runtime::amr::AmrGhostFillBudget exact_amr_ghost_budget(const MultiFab<Dim>& coarse,
                                                        const MultiFab<Dim>& fine,
                                                        const Box<Dim>& coarse_domain,
                                                        const Box<Dim>& fine_domain,
                                                        const BoundaryTopology<Dim>& topology) {
  const std::size_t coarse_patches = coarse.layout().size();
  const std::size_t fine_patches = fine.layout().size();
  const std::size_t ranks = fine.rank_space().size();
  const std::size_t images = periodic_image_bound(fine_domain, fine.ghosts(), topology);
  const std::size_t regions =
      checked_size_product(checked_size_product(fine_patches, static_cast<std::size_t>(2 * Dim),
                                                "AMR coarse/fine region budget exceeds size_t"),
                           images, "AMR coarse/fine periodic-region budget exceeds size_t");
  const std::size_t cross_pairs = checked_size_product(
      coarse_patches, fine_patches, "AMR coarse/fine patch-pair budget exceeds size_t");
  const std::size_t pair_budget =
      std::max({cross_pairs, checked_pair_count(coarse_patches), checked_pair_count(fine_patches)});
  const std::size_t jobs = checked_size_product(
      checked_size_product(std::max<std::size_t>(regions, fine_patches),
                           std::max<std::size_t>(coarse_patches, 1),
                           "AMR coarse/fine job budget exceeds size_t"),
      std::max<std::size_t>(ranks, 1), "AMR coarse/fine rank-job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(
          checked_size_product(checked_cells(coarse_domain), static_cast<std::size_t>(fine.ncomp()),
                               "AMR coarse/fine component budget exceeds size_t"),
          std::max<std::size_t>(fine_patches, 1),
          "AMR coarse/fine patch-element budget exceeds size_t"),
      checked_size_product(std::max<std::size_t>(images, 1), std::max<std::size_t>(ranks, 1),
                           "AMR coarse/fine image-rank budget exceeds size_t"),
      "AMR coarse/fine element budget exceeds size_t");

  runtime::amr::AmrGhostFillBudget budget;
  budget.coarse_fine = runtime::amr::CoarseFineGhostScheduleBudget{
      fine_patches, regions, pair_budget, jobs, ranks, elements, elements, elements};
  budget.same_level = exact_halo_budget(fine, fine_domain, topology);
  return budget;
}

template <int Dim>
std::vector<const Real*> field_storage_identity(const MultiFab<Dim>& field) {
  std::vector<const Real*> identity;
  identity.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    identity.push_back(field.fab(local).view().data);
  return identity;
}

template <int Dim>
bool field_storage_matches(const MultiFab<Dim>& field,
                           const std::vector<const Real*>& identity) noexcept {
  try {
    if (field.local_size() != identity.size())
      return false;
    for (std::size_t local = 0; local < field.local_size(); ++local)
      if (field.fab(local).view().data != identity[local])
        return false;
    return true;
  } catch (...) {
    return false;
  }
}

template <int Dim>
bool same_field_shape(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return left.layout() == right.layout() && left.distribution() == right.distribution() &&
         left.local_rank() == right.local_rank() && left.local_size() == right.local_size() &&
         left.ncomp() == right.ncomp();
}

template <int Dim>
bool same_field_contract(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return same_field_shape(left, right) && left.ghosts() == right.ghosts();
}

template <int Dim>
PreparedAmrLevelEvaluation<Dim> make_prepared_level_evaluation_workspace(
    const MultiFab<Dim>& prototype, std::string_view spatial_contract, std::uint64_t topology_epoch,
    std::uint64_t materialization_generation) {
  PreparedAmrLevelEvaluation<Dim> evaluation{
      .spatial_contract = std::string(spatial_contract),
      .topology_epoch = topology_epoch,
      .materialization_generation = materialization_generation,
      .residual = MultiFab<Dim>(prototype.layout(), prototype.distribution(),
                                prototype.local_rank(), prototype.ncomp(), prototype.ghosts()),
      .integrated_face_fluxes = nd::make_face_flux_workspace(prototype)};
  evaluation.point.clock.reserve(kPreparedAmrClockIdentityCapacity);
  evaluation.residual.set_val(Real(0));
  for (auto& faces : evaluation.integrated_face_fluxes)
    faces.set_val(Real(0));
  return evaluation;
}

template <int Dim>
void copy_full_field_in_place(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (!same_field_contract(source, destination))
    throw std::invalid_argument("AMR full-field copy requires one exact ranked field contract");

  // Complete validation precedes the first device write so a malformed candidate cannot partially
  // mutate storage pinned by a prepared halo provider.
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const Fab<Dim>& source_fab = source.fab(local);
    const Fab<Dim>& destination_fab = destination.fab(local);
    if (source.global_index(local) != destination.global_index(local) ||
        source_fab.box() != destination_fab.box() ||
        source_fab.grown_box() != destination_fab.grown_box() ||
        source_fab.size() != destination_fab.size())
      throw std::invalid_argument(
          "AMR full-field copy encountered different exact local patch storage");
  }
  for (std::size_t local = 0; local < source.local_size(); ++local)
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  Kokkos::fence();
}

template <int Dim>
void copy_auxiliary_groups_in_place(const runtime::system::AuxiliaryStorageGroups<Dim>& source,
                                    runtime::system::AuxiliaryStorageGroups<Dim>& destination) {
  if (source.groups.size() != destination.groups.size())
    throw std::invalid_argument("AMR provider candidate has another resolved storage-group set");
  for (const auto& [identity, source_group] : source.groups) {
    const MultiFab<Dim>* destination_group = destination.find(identity);
    if (destination_group == nullptr || !same_field_contract(source_group, *destination_group))
      throw std::invalid_argument("AMR provider candidate lost one resolved storage group");
  }
  for (const auto& [identity, source_group] : source.groups) {
    MultiFab<Dim>* destination_group = destination.find(identity);
    copy_full_field_in_place(source_group, *destination_group);
  }
}

template <int Dim>
void restore_exact_field_collectively(bool& staged, MultiFab<Dim>& backup, MultiFab<Dim>& live,
                                      const CommunicatorView& communicator) {
  if (!staged)
    return;
  std::exception_ptr restore_error;
  long restore_failure = 0;
  try {
    copy_full_field_in_place(backup, live);
  } catch (...) {
    restore_failure = 1;
    restore_error = std::current_exception();
  }
  if (all_reduce_max(restore_failure, communicator) != 0) {
    if (restore_error)
      std::rethrow_exception(restore_error);
    throw std::runtime_error("prepared AMR live-state restoration failed collectively");
  }
  staged = false;
}

template <int Dim>
bool stage_exact_field_collectively(const MultiFab<Dim>& candidate, MultiFab<Dim>& live,
                                    MultiFab<Dim>& backup, const CommunicatorView& communicator) {
  const long staged = &candidate == &live ? 0L : 1L;
  if (all_reduce_min(staged, communicator) != all_reduce_max(staged, communicator))
    throw std::invalid_argument("prepared AMR candidate/live selection differs between MPI ranks");
  if (staged == 0)
    return false;

  std::exception_ptr preflight_error;
  long preflight_failure = 0;
  try {
    if (!same_field_contract(candidate, live) || !same_field_contract(backup, live) ||
        &backup == &candidate || &backup == &live)
      throw std::invalid_argument(
          "prepared AMR candidate/backup differs from its exact live level contract");
  } catch (...) {
    preflight_failure = 1;
    preflight_error = std::current_exception();
  }
  if (all_reduce_max(preflight_failure, communicator) != 0) {
    if (preflight_error)
      std::rethrow_exception(preflight_error);
    throw std::runtime_error("prepared AMR candidate staging preflight failed collectively");
  }

  std::exception_ptr backup_error;
  try {
    copy_full_field_in_place(live, backup);
  } catch (...) {
    backup_error = std::current_exception();
  }
  if (all_reduce_max(backup_error ? 1L : 0L, communicator) != 0) {
    if (backup_error)
      std::rethrow_exception(backup_error);
    throw std::runtime_error("prepared AMR live backup failed collectively");
  }

  bool published = true;
  std::exception_ptr staging_error;
  try {
    copy_full_field_in_place(candidate, live);
  } catch (...) {
    staging_error = std::current_exception();
  }
  if (all_reduce_max(staging_error ? 1L : 0L, communicator) != 0) {
    restore_exact_field_collectively(published, backup, live, communicator);
    if (staging_error)
      std::rethrow_exception(staging_error);
    throw std::runtime_error("prepared AMR candidate staging failed collectively");
  }
  return true;
}

template <int Dim, class Callback>
decltype(auto) invoke_with_staged_parent(int runtime_block, std::string_view block_identity,
                                         int child_level, int parent_level,
                                         const MultiFab<Dim>* staged_parent,
                                         MultiFab<Dim>& live_parent,
                                         std::string_view hierarchy_contract,
                                         const CommunicatorView& communicator,
                                         const std::vector<MultiFab<Dim>>*& hierarchy_candidates,
                                         MultiFab<Dim>& parent_backup, Callback&& callback) {
  std::exception_ptr binding_error;
  std::string binding_contract;
  try {
    if (runtime_block < 0 || block_identity.empty() || child_level < 1 ||
        parent_level != child_level - 1 || staged_parent == nullptr)
      throw std::invalid_argument(
          "subcycled AMR provider requires one exact block-qualified staged parent");
    if (!same_field_contract(*staged_parent, live_parent))
      throw std::invalid_argument(
          "subcycled AMR staged parent differs from its exact live block/level contract");
    ExactContractBuilder exact;
    exact.text("pops.amr-system.block-staged-parent")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(hierarchy_contract)
        .scalar(std::int32_t{runtime_block})
        .text(block_identity)
        .scalar(std::int32_t{child_level})
        .scalar(std::int32_t{parent_level})
        .scalar(std::int32_t{live_parent.ncomp()});
    for (int axis = 0; axis < Dim; ++axis)
      exact.scalar(live_parent.ghosts()[axis]);
    binding_contract = std::move(exact).release();
  } catch (...) {
    binding_error = std::current_exception();
  }
  if (all_reduce_max(binding_error ? 1L : 0L, communicator) != 0) {
    if (binding_error)
      std::rethrow_exception(binding_error);
    throw std::runtime_error("subcycled AMR staged-parent binding failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-system-block-staged-parent"), binding_contract}}, communicator))
    throw std::invalid_argument(
        "subcycled AMR staged-parent block/level identities differ between ranks");

  bool parent_staged =
      stage_exact_field_collectively(*staged_parent, live_parent, parent_backup, communicator);
  const std::vector<MultiFab<Dim>>* const saved_hierarchy_candidates = hierarchy_candidates;
  hierarchy_candidates = nullptr;

  using result_type = std::invoke_result_t<Callback>;
  if constexpr (std::is_void_v<result_type>) {
    std::exception_ptr callback_error;
    try {
      std::invoke(std::forward<Callback>(callback));
    } catch (...) {
      callback_error = std::current_exception();
    }
    hierarchy_candidates = saved_hierarchy_candidates;
    restore_exact_field_collectively(parent_staged, parent_backup, live_parent, communicator);
    if (callback_error)
      std::rethrow_exception(callback_error);
  } else if constexpr (std::is_lvalue_reference_v<result_type>) {
    using value_type = std::remove_reference_t<result_type>;
    value_type* result = nullptr;
    std::exception_ptr callback_error;
    try {
      result = &std::invoke(std::forward<Callback>(callback));
    } catch (...) {
      callback_error = std::current_exception();
    }
    hierarchy_candidates = saved_hierarchy_candidates;
    restore_exact_field_collectively(parent_staged, parent_backup, live_parent, communicator);
    if (callback_error)
      std::rethrow_exception(callback_error);
    return static_cast<result_type>(*result);
  } else {
    std::optional<result_type> result;
    std::exception_ptr callback_error;
    try {
      result.emplace(std::invoke(std::forward<Callback>(callback)));
    } catch (...) {
      callback_error = std::current_exception();
    }
    hierarchy_candidates = saved_hierarchy_candidates;
    restore_exact_field_collectively(parent_staged, parent_backup, live_parent, communicator);
    if (callback_error)
      std::rethrow_exception(callback_error);
    return result_type(std::move(*result));
  }
}

template <int Dim>
void authenticate_generated_block_point(std::string_view route, int runtime_block,
                                        std::string_view block_identity,
                                        const runtime::multiblock::BoundaryEvaluationPoint& point,
                                        std::string_view hierarchy_contract,
                                        const CommunicatorView& communicator) {
  std::exception_ptr local_error;
  std::string exact_contract;
  try {
    if (route.empty() || runtime_block < 0 || block_identity.empty() || hierarchy_contract.empty())
      throw std::invalid_argument("generated AMR provider target identity is incomplete");
    ExactContractBuilder exact;
    exact.text("pops.generated-amr-block-point")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(route)
        .bytes(hierarchy_contract)
        .scalar(std::int32_t{runtime_block})
        .text(block_identity)
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    exact_contract = std::move(exact).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("generated AMR block/provider point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-block-point"), exact_contract}}, communicator))
    throw std::invalid_argument(
        "generated AMR block/provider point identities differ between MPI ranks");
}

template <int Dim>
struct CopyValidFieldKernel {
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < source.ncomp; ++component)
      destination(index, component) = source(index, component);
  }
};

template <int Dim>
void copy_valid_field(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (!same_field_shape(source, destination))
    throw std::invalid_argument("AMR auxiliary copy requires one exact ranked field shape");
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const std::size_t global = destination.global_index(local);
    if (!source.contains_local(global))
      throw std::invalid_argument("AMR auxiliary copy source lacks the destination global patch");
    const Fab<Dim>& source_fab = source.fab(source.local_index_of(global));
    Fab<Dim>& destination_fab = destination.fab(local);
    if (source_fab.box() != destination_fab.box() ||
        source_fab.grown_box() != destination_fab.grown_box())
      throw std::invalid_argument("AMR auxiliary copy patch storage differs");
    const Box<Dim>& valid = source_fab.box();
    for_each_cell(valid, CopyValidFieldKernel<Dim>{source_fab.view(), destination_fab.view()});
  }
  Kokkos::fence();
}

template <int Dim>
std::string exact_root_ghost_contract(const HaloSchedule<Dim>& schedule,
                                      std::string_view field_identity,
                                      std::uint64_t topology_generation,
                                      std::uint64_t materialization_generation,
                                      std::string_view lane_identity) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-root-ghost-fill")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(field_identity)
      .text(lane_identity)
      .scalar(topology_generation)
      .scalar(materialization_generation)
      .scalar(schedule.coverage())
      .scalar(std::int32_t{schedule.ncomp()});
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int64_t{schedule.domain().lo[axis]})
        .scalar(std::int64_t{schedule.domain().hi[axis]})
        .scalar(std::int64_t{schedule.ghosts()[axis]})
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::lower}))
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::upper}));
  contract.sequence(schedule.layout().boxes(), [](ExactContractBuilder& item, const Box<Dim>& box) {
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{box.lo[axis]}).scalar(std::int64_t{box.hi[axis]});
  });
  contract.sequence(schedule.distribution().owners(),
                    [](ExactContractBuilder& item, const Index<Dim>& owner) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(std::int64_t{owner[axis]});
                    });
  contract.sequence(schedule.canonical_jobs(), [](ExactContractBuilder& item, const auto& job) {
    item.scalar(static_cast<std::uint64_t>(job.source_box))
        .scalar(static_cast<std::uint64_t>(job.destination_box))
        .scalar(static_cast<std::uint64_t>(job.elements));
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{job.destination_region.lo[axis]})
          .scalar(std::int64_t{job.destination_region.hi[axis]})
          .scalar(std::int64_t{job.source_from_destination[axis]});
  });
  return std::move(contract).release();
}

template <int Dim>
struct GeneratedRootGhostSource {
  using field_type = MultiFab<Dim>;

  struct State {
    std::optional<HaloSchedule<Dim>> schedule;
    const ExecutionLane* lane = nullptr;
    std::optional<ExecutionLane::ImmutableBorrow> lane_borrow;
    std::optional<HaloExchange<Dim>> exchange;
    std::vector<const Real*> storage;
    std::string contract;

    void execute(field_type& field, const runtime::multiblock::BoundaryEvaluationPoint& point) {
      const long binding_invalid = !schedule || lane == nullptr ? 1L : 0L;
      if (lane == nullptr)
        throw std::logic_error("prepared AMR root ghost provider lost its immutable binding");
      if (all_reduce_max(binding_invalid, lane->communicator()) != 0)
        throw std::logic_error("prepared AMR root ghost provider lost its immutable binding");
      const long invalid = point.level != 0 || !field_storage_matches(field, storage) ? 1L : 0L;
      if (all_reduce_max(invalid, lane->communicator()) != 0)
        throw std::invalid_argument(
            "prepared AMR root ghost provider received stale storage or a non-root level");
      if (exchange)
        fill_boundary(field, *exchange, *lane);
      else
        fill_boundary(field, *schedule);
    }
  };

  std::shared_ptr<State> state;

  [[nodiscard]] static PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.generated.amr.root-ghost-fill", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    contract.bytes(state->contract);
  }

  void operator()(field_type& field,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    state->execute(field, point);
  }
};

inline std::uint64_t exchange_generation(std::uint64_t generation, const char* label) {
  if (generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string(label) + " cannot form a halo exchange generation");
  return generation + 1;
}

template <int Dim>
PreparedRootAmrGhostFill<Dim> prepare_root_ghost_fill(MultiFab<Dim>& field, const Box<Dim>& domain,
                                                      const BoundaryTopology<Dim>& topology,
                                                      std::string field_identity,
                                                      std::uint64_t topology_generation,
                                                      std::uint64_t materialization_generation,
                                                      const ExecutionLane& lane) {
  using source_type = GeneratedRootGhostSource<Dim>;
  std::shared_ptr<typename source_type::State> state;
  std::exception_ptr metadata_error;
  long metadata_failure = 0;
  try {
    state = std::make_shared<typename source_type::State>();
    state->schedule.emplace(prepare_halo_schedule(field, domain, topology,
                                                  HaloLayoutCoverage::full_domain,
                                                  exact_halo_budget(field, domain, topology)));
    state->lane = &lane;
    state->storage = field_storage_identity(field);
    state->contract =
        exact_root_ghost_contract(*state->schedule, field_identity, topology_generation,
                                  materialization_generation, lane.identity());
  } catch (...) {
    metadata_failure = 1;
    metadata_error = std::current_exception();
  }
  if (all_reduce_max(metadata_failure, lane.communicator()) != 0) {
    if (metadata_error)
      std::rethrow_exception(metadata_error);
    throw std::runtime_error("generated AMR root ghost metadata failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-root-ghost"), std::string_view(state->contract)}},
          lane.communicator()))
    throw std::invalid_argument("generated AMR root ghost contracts differ between ranks");

  const long remote_any =
      all_reduce_max(state->schedule->has_remote_jobs() ? 1L : 0L, lane.communicator());
  std::exception_ptr transport_error;
  long transport_failure = 0;
  try {
    if (remote_any != 0) {
      state->exchange.emplace(
          *state->schedule, lane,
          HaloExchangeContext{
              .context_generation = exchange_generation(topology_generation, "topology"),
              .schedule_generation =
                  exchange_generation(materialization_generation, "materialization"),
          });
    } else {
      state->lane_borrow.emplace(lane.borrow_immutably());
    }
  } catch (...) {
    transport_failure = 1;
    transport_error = std::current_exception();
  }
  if (all_reduce_max(transport_failure, lane.communicator()) != 0) {
    if (transport_error)
      std::rethrow_exception(transport_error);
    throw std::runtime_error("generated AMR root ghost transport failed collectively");
  }
  std::optional<PreparedRootAmrGhostFill<Dim>> prepared;
  std::exception_ptr provider_error;
  long provider_failure = 0;
  try {
    prepared.emplace(source_type{std::move(state)});
  } catch (...) {
    provider_failure = 1;
    provider_error = std::current_exception();
  }
  if (all_reduce_max(provider_failure, lane.communicator()) != 0) {
    if (provider_error)
      std::rethrow_exception(provider_error);
    throw std::runtime_error("generated AMR root ghost provider failed collectively");
  }
  return std::move(*prepared);
}

/// One prepared ghost route per resolved storage group.  Group identity is part of the retained
/// contract, so a regrid cannot accidentally fill a similarly-shaped but differently-owned value.
template <int Dim>
void append_provider_groups_structure(ExactContractBuilder& exact,
                                      const runtime::system::AuxiliaryStorageGroups<Dim>& groups,
                                      std::string_view role) {
  exact.text(role).scalar(static_cast<std::uint64_t>(groups.groups.size()));
  for (const auto& [identity, group] : groups.groups) {
    if (identity.empty() || group.ncomp() < 1)
      throw std::invalid_argument(
          "AMR provider-group ghost preparation requires named, non-empty storage groups");
    exact.text(identity).scalar(std::int32_t{group.ncomp()});
    for (int axis = 0; axis < Dim; ++axis)
      exact.scalar(std::int64_t{group.ghosts()[axis]});
    exact.sequence(group.layout().boxes(), [](ExactContractBuilder& item, const Box<Dim>& box) {
      for (int axis = 0; axis < Dim; ++axis)
        item.scalar(std::int64_t{box.lo[axis]}).scalar(std::int64_t{box.hi[axis]});
    });
    const auto& distribution = group.distribution();
    exact.scalar(distribution.mode());
    for (int axis = 0; axis < Dim; ++axis)
      exact.scalar(std::int64_t{distribution.rank_space().origin()[axis]})
          .scalar(std::int64_t{distribution.rank_space().extent()[axis]});
    exact.sequence(distribution.owners(), [](ExactContractBuilder& item, const Index<Dim>& owner) {
      for (int axis = 0; axis < Dim; ++axis)
        item.scalar(std::int64_t{owner[axis]});
    });
  }
}

template <int Dim>
std::string require_collective_provider_groups_structure(
    const runtime::system::AuxiliaryStorageGroups<Dim>* coarse,
    const runtime::system::AuxiliaryStorageGroups<Dim>& target, const ExecutionLane& lane,
    std::string_view label) {
  std::string contract;
  std::exception_ptr local_error;
  try {
    if (coarse != nullptr) {
      if (coarse->groups.size() != target.groups.size())
        throw std::invalid_argument(
            "AMR provider-group transfer requires one identical resolved group set");
      for (const auto& [identity, child] : target.groups) {
        const auto parent = coarse->groups.find(identity);
        if (parent == coarse->groups.end())
          throw std::invalid_argument(
              "AMR provider-group transfer lost one resolved group identity");
        (void)child;
      }
    }
    ExactContractBuilder exact;
    exact.text("pops.generated-amr-provider-groups-structure")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(label);
    if (coarse != nullptr)
      append_provider_groups_structure(exact, *coarse, "coarse");
    append_provider_groups_structure(exact, target, "target");
    contract = std::move(exact).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      local_error, &lane, "AMR provider-group structure preparation failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-provider-groups-structure"),
            std::string_view(contract)}},
          lane.communicator()))
    throw std::invalid_argument(
        "AMR provider-group ordered structure differs across communicator ranks");
  return contract;
}

template <int Dim>
struct GeneratedProviderGroupsGhostSource {
  using groups_type = runtime::system::AuxiliaryStorageGroups<Dim>;
  using root_fill_type = PreparedRootAmrGhostFill<Dim>;
  using fine_fill_type = runtime::amr::PreparedAmrGhostFill<Dim>;

  struct State {
    bool root = false;
    const ExecutionLane* lane = nullptr;
    std::vector<std::string> identities;
    std::vector<const MultiFab<Dim>*> carriers;
    std::vector<int> component_counts;
    std::vector<Extent<Dim>> ghosts;
    std::vector<std::vector<const Real*>> storage;
    std::vector<root_fill_type> root_fills;
    std::vector<fine_fill_type> fine_fills;
    std::string contract;

    void execute(groups_type& groups,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
      if (lane == nullptr)
        throw std::logic_error("prepared AMR provider-group ghost state lost its collective lane");
      std::exception_ptr binding_error;
      try {
        const std::size_t count = identities.size();
        if (groups.groups.size() != count || carriers.size() != count ||
            component_counts.size() != count || ghosts.size() != count || storage.size() != count ||
            count != (root ? root_fills.size() : fine_fills.size()))
          throw std::logic_error("prepared AMR provider-group ghost state is malformed");
        for (std::size_t index = 0; index < count; ++index) {
          const auto* group = groups.find(identities[index]);
          if (group == nullptr || group != carriers[index] ||
              group->ncomp() != component_counts[index] || group->ghosts() != ghosts[index] ||
              !field_storage_matches(*group, storage[index]))
            throw std::logic_error(
                "prepared AMR provider-group carrier lost one exact storage binding");
        }
      } catch (...) {
        binding_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          binding_error, lane, "prepared AMR provider-group ghost binding failed collectively");
      for (std::size_t index = 0; index < identities.size(); ++index) {
        std::exception_ptr fill_error;
        try {
          auto* group = groups.find(identities[index]);
          if (root)
            root_fills[index](*group, point);
          else
            fine_fills[index](*group, point);
        } catch (...) {
          fill_error = std::current_exception();
        }
        runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
            fill_error, lane, "prepared AMR provider-group ghost phase failed collectively");
      }
    }
  };

  std::shared_ptr<State> state;

  [[nodiscard]] static PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.generated.amr.provider-groups-ghost-fill", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    if (!state)
      throw std::logic_error("generated AMR provider-group ghost source is empty");
    contract.bytes(state->contract);
  }
  void operator()(groups_type& groups,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!state)
      throw std::logic_error("generated AMR provider-group ghost source is empty");
    state->execute(groups, point);
  }
};

template <int Dim>
PreparedProviderGroupsGhostFill<Dim> prepare_provider_groups_root_ghost_fill(
    runtime::system::AuxiliaryStorageGroups<Dim>& groups, const Box<Dim>& domain,
    const BoundaryTopology<Dim>& topology, std::string_view identity,
    std::uint64_t topology_generation, std::uint64_t materialization_generation,
    const ExecutionLane& lane) {
  using source_type = GeneratedProviderGroupsGhostSource<Dim>;
  const std::string group_structure = require_collective_provider_groups_structure(
      static_cast<const runtime::system::AuxiliaryStorageGroups<Dim>*>(nullptr), groups, lane,
      identity);
  std::shared_ptr<typename source_type::State> state;
  std::vector<std::string> route_identities;
  std::exception_ptr metadata_error;
  try {
    state = std::make_shared<typename source_type::State>();
    state->root = true;
    state->lane = &lane;
    const std::size_t count = groups.groups.size();
    state->identities.reserve(count);
    state->carriers.reserve(count);
    state->component_counts.reserve(count);
    state->ghosts.reserve(count);
    state->storage.reserve(count);
    state->root_fills.reserve(count);
    route_identities.reserve(count);
    for (auto& [group_identity, group] : groups.groups) {
      state->identities.push_back(group_identity);
      state->carriers.push_back(&group);
      state->component_counts.push_back(group.ncomp());
      state->ghosts.push_back(group.ghosts());
      state->storage.push_back(field_storage_identity(group));
      route_identities.push_back(std::string(identity) + "/" + group_identity);
    }
  } catch (...) {
    metadata_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      metadata_error, &lane, "AMR root provider-group metadata preparation failed collectively");
  for (std::size_t index = 0; index < state->identities.size(); ++index) {
    auto group = groups.groups.find(state->identities[index]);
    state->root_fills.emplace_back(
        prepare_root_ghost_fill(group->second, domain, topology, std::move(route_identities[index]),
                                topology_generation, materialization_generation, lane));
  }
  std::optional<PreparedProviderGroupsGhostFill<Dim>> prepared;
  std::exception_ptr provider_error;
  try {
    ExactContractBuilder exact;
    exact.text("pops.generated-amr-provider-groups-root-ghost")
        .scalar(std::uint32_t{1})
        .text(identity)
        .scalar(topology_generation)
        .scalar(materialization_generation)
        .bytes(group_structure);
    for (std::size_t index = 0; index < state->identities.size(); ++index)
      exact.text(state->identities[index]).bytes(state->root_fills[index].collective_contract());
    state->contract = std::move(exact).release();
    prepared.emplace(source_type{state});
  } catch (...) {
    provider_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      provider_error, &lane, "AMR root provider-group authority failed collectively");
  return std::move(*prepared);
}

template <int Dim>
PreparedProviderGroupsGhostFill<Dim> prepare_provider_groups_fine_ghost_fill(
    runtime::system::AuxiliaryStorageGroups<Dim>& coarse,
    runtime::system::AuxiliaryStorageGroups<Dim>& fine, const Box<Dim>& coarse_domain,
    const Box<Dim>& fine_domain, const amr::RefinementRatio<Dim>& ratio,
    const BoundaryTopology<Dim>& topology, std::string_view identity, int fine_level,
    std::uint64_t topology_generation, std::uint64_t materialization_generation,
    const ExecutionLane& lane) {
  using source_type = GeneratedProviderGroupsGhostSource<Dim>;
  const std::string group_structure =
      require_collective_provider_groups_structure(&coarse, fine, lane, identity);
  std::shared_ptr<typename source_type::State> state;
  std::vector<std::string> route_identities;
  std::vector<runtime::amr::AmrGhostFillBudget> budgets;
  std::exception_ptr metadata_error;
  try {
    state = std::make_shared<typename source_type::State>();
    state->lane = &lane;
    const std::size_t count = fine.groups.size();
    state->identities.reserve(count);
    state->carriers.reserve(count);
    state->component_counts.reserve(count);
    state->ghosts.reserve(count);
    state->storage.reserve(count);
    state->fine_fills.reserve(count);
    route_identities.reserve(count);
    budgets.reserve(count);
    for (auto& [group_identity, child] : fine.groups) {
      auto parent = coarse.groups.find(group_identity);
      state->identities.push_back(group_identity);
      state->carriers.push_back(&child);
      state->component_counts.push_back(child.ncomp());
      state->ghosts.push_back(child.ghosts());
      state->storage.push_back(field_storage_identity(child));
      route_identities.push_back(std::string(identity) + "/" + group_identity);
      budgets.push_back(
          exact_amr_ghost_budget(parent->second, child, coarse_domain, fine_domain, topology));
    }
  } catch (...) {
    metadata_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      metadata_error, &lane, "AMR fine provider-group metadata preparation failed collectively");
  for (std::size_t index = 0; index < state->identities.size(); ++index) {
    auto parent = coarse.groups.find(state->identities[index]);
    auto child = fine.groups.find(state->identities[index]);
    state->fine_fills.emplace_back(runtime::amr::prepare_amr_ghost_fill(
        parent->second, child->second,
        runtime::amr::AmrGhostFillPreparation<Dim>{
            .fine_level = fine_level,
            .coarse_domain = coarse_domain,
            .fine_domain = fine_domain,
            .ratio = ratio,
            .topology = topology,
            .topology_generation = topology_generation,
            .materialization_generation = materialization_generation,
            .field_identity = std::move(route_identities[index]),
            .budget = budgets[index],
        },
        lane));
  }
  std::optional<PreparedProviderGroupsGhostFill<Dim>> prepared;
  std::exception_ptr provider_error;
  try {
    ExactContractBuilder exact;
    exact.text("pops.generated-amr-provider-groups-fine-ghost")
        .scalar(std::uint32_t{1})
        .text(identity)
        .scalar(topology_generation)
        .scalar(materialization_generation)
        .bytes(group_structure);
    for (std::size_t index = 0; index < state->identities.size(); ++index)
      exact.text(state->identities[index]).bytes(state->fine_fills[index].collective_contract());
    state->contract = std::move(exact).release();
    prepared.emplace(source_type{state});
  } catch (...) {
    provider_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      provider_error, &lane, "AMR fine provider-group authority failed collectively");
  return std::move(*prepared);
}

void validate_variable_set(const VariableSet& variables, VariableKind expected, int ncomp,
                           const char* label) {
  if (variables.kind != expected || variables.size != ncomp ||
      variables.names.size() != static_cast<std::size_t>(ncomp) ||
      variables.roles.size() != static_cast<std::size_t>(ncomp) ||
      (!variables.user_roles.empty() &&
       variables.user_roles.size() != static_cast<std::size_t>(ncomp)))
    throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                " variable metadata differs from its component count");
  std::set<std::string> names;
  for (const std::string& name : variables.names)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                  " variable names must be unique and non-empty");
}

template <int Dim>
void validate_prepared_amr_block(const PreparedAmrSystemBlock<Dim>& block) {
  if (block.name.empty() || block.provider_identity.empty() ||
      block.staircase_provider_identity.empty() || block.collective_contract.empty())
    throw std::invalid_argument(
        "prepared AMR block requires non-empty block, Cartesian, staircase, and collective "
        "identities");
  if (block.cut_cell_provider_identity.empty())
    throw std::invalid_argument("prepared AMR block requires a cut-cell provider identity");
  if (block.ncomp < 1 || block.provider_components < 0 || block.reconstruction_order < 1)
    throw std::invalid_argument(
        "prepared AMR block requires positive state/order and a non-negative provider-value "
        "count");
  if (!std::isfinite(block.gamma) || !(block.gamma > 0.0) || block.substeps < 1 ||
      block.stride < 1 || block.time_route.empty())
    throw std::invalid_argument("prepared AMR block gamma, cadence, or time route is invalid");
  for (int axis = 0; axis < Dim; ++axis)
    if (block.ghosts[axis] < 1)
      throw std::invalid_argument(
          "prepared AMR block must declare a positive ghost extent on every native axis");
  validate_variable_set(block.conservative_variables, VariableKind::Conservative, block.ncomp,
                        "conservative");
  validate_variable_set(block.primitive_variables, VariableKind::Primitive, block.ncomp,
                        "primitive");
  if (!block.materialize_level || !block.primitive_to_conservative ||
      !block.conservative_to_primitive || !block.batch_conservative_to_primitive)
    throw std::invalid_argument(
        "prepared AMR block does not implement its complete exact-ranked execution contract");
}

template <int Dim>
std::string exact_hyperbolic_boundary_contract(const PreparedHyperbolicBoundary<Dim>& boundary) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-hyperbolic-boundary")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{boundary.ncomp()})
      .scalar(boundary.corner_policy());
  for (int component = 0; component < boundary.ncomp(); ++component) {
    const auto& transform = boundary.component_transform(component);
    contract.scalar(transform.parity).scalar(std::int32_t{transform.axis});
  }
  for (int axis = 0; axis < Dim; ++axis) {
    for (const int side : {-1, 1}) {
      const PreparedHyperbolicFace& face = boundary.face(axis, side);
      contract.scalar(std::int32_t{axis})
          .scalar(std::int32_t{side})
          .scalar(face.law)
          .text(face.identity)
          .scalar(face.identity_token)
          .scalar(face.authored_representation)
          .text(face.converter_identity)
          .scalar(face.fixed_state_converted)
          .text(face.analytic_clock)
          .scalar(static_cast<std::uint64_t>(face.fixed_state.size()));
      for (const Real value : face.fixed_state)
        contract.scalar(value);
      contract.scalar(static_cast<std::uint64_t>(face.analytic_state.size()));
      for (const analytic::AnalyticProgram& program : face.analytic_state) {
        const analytic::AnalyticProgramView view = program.view();
        contract.scalar(static_cast<std::uint64_t>(program.instruction_count()))
            .scalar(static_cast<std::uint64_t>(program.literal_count()))
            .scalar(static_cast<std::uint64_t>(program.required_stack()))
            .scalar(std::uint8_t{static_cast<std::uint8_t>(program.required_dimension())})
            .scalar(program.result_type());
        for (std::size_t index = 0; index < program.instruction_count(); ++index)
          contract.scalar(view.instructions[index].op).scalar(view.instructions[index].operand);
        for (std::size_t index = 0; index < program.literal_count(); ++index)
          contract.scalar(view.literals[index]);
      }
    }
  }
  return std::move(contract).release();
}

}  // namespace

template <int Dim>
struct AmrSystem<Dim>::Impl {
  using engine_type = runtime::amr::AmrRuntime<Dim>;
  using field_type = MultiFab<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;
  using prepared_block_type = PreparedAmrSystemBlock<Dim>;
  using multiblock_type = runtime::amr::PreparedMultiBlockAmrHierarchy<Dim>;
  using flux_expression_block_budget_type =
      typename AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  using flux_expression_budget_type =
      typename AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBudget;
  using level_block_type = PreparedGeneratedAmrLevelBlock<Dim>;
  using evaluation_type = PreparedAmrLevelEvaluation<Dim>;
  using exact_field_solver_type = runtime::amr::ExactAmrFieldSolver<Dim>;
  using exact_field_provider_type = runtime::amr::ExactAmrFieldSolverProvider<Dim>;
  using exact_field_registry_type = runtime::amr::ExactAmrFieldSolverRegistry<Dim>;
  using hierarchy_tensor_provider_type = runtime::program::HierarchyTensorSolverProvider<Dim>;
  using hierarchy_tensor_registry_type =
      runtime::program::HierarchyTensorSolverProviderRegistry<Dim>;
  using auxiliary_registry_type = runtime::system::ExactAuxiliaryRegistry<Dim>;
  using auxiliary_groups_type = runtime::system::AuxiliaryStorageGroups<Dim>;
  using auxiliary_publication_type = typename auxiliary_registry_type::PublicationTransaction;

  struct BlockSpec {
    std::string name;
    int ncomp = 0;
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    int substeps = 1;
    int stride = 1;
    int required_ghost_depth = 1;
    int reconstruction_order = 1;
    Extent<Dim> ghosts{};
    std::string time = "euler";
    bool has_density = false;
    std::vector<double> density;
    bool has_state = false;
    std::vector<double> state;
    bool has_analytic_state = false;
    std::vector<analytic::AnalyticProgram> analytic_state;
  };

  enum class BootstrapSourceKind : std::uint8_t { unstaged, analytic, array };
  struct BootstrapSourceAuthority {
    std::string runtime_block;
    std::string source_route;
    BootstrapSourceKind kind = BootstrapSourceKind::unstaged;
  };
  /// Resolved initial Handle -> exact runtime block/source association.  This is deliberately
  /// distinct from the block label: only the pre-installed owner-qualified state route and the
  /// resolved InitialCondition source can establish it.
  std::map<std::string, BootstrapSourceAuthority> bootstrap_sources;

  struct GlobalDtBound {
    std::string label;
    std::function<double()> evaluate;
  };

  struct BootstrapTransferRoute {
    std::string identity;
    std::vector<std::string> subjects;
    std::string provider_identity;
    std::string space;
    std::string centering;
    std::string representation;
    std::string storage;
    std::string operation;
    std::string kernel;
    int order = 0;
    Extent<Dim> ghost_depth{};
    Extent<Dim> refinement_ratio{};
  };

  struct FieldProviderBinding {
    std::string identity;
    std::string block;
    std::string key;
    double coefficient = 0.0;
  };

  struct PreparedFieldRhs {
    std::function<void(const field_type&, field_type&)> evaluate;
    Real coefficient = Real(1);
  };

  struct TaggingSpec {
    std::vector<std::string> leaf_subject_kinds;
    std::vector<std::string> leaf_subject_identities;
    std::vector<std::string> leaf_blocks;
    std::vector<std::string> leaf_variables;
    std::vector<int> leaf_field_component_indices;
    std::vector<int> leaf_ops;
    std::vector<double> leaf_thresholds;
    std::vector<int> leaf_stencil_indices;
    std::vector<typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil> stencils;
    std::vector<std::int32_t> refine_ops;
    std::vector<std::int32_t> refine_args;
    std::vector<std::int32_t> coarsen_ops;
    std::vector<std::int32_t> coarsen_args;
    int min_cycles = 0;
    int equality_policy = 0;
    int conflict_policy = 0;
    std::string clock_identity;
    std::string provider_identity;
  };

  enum class TaggingFieldKind : std::uint8_t { state, auxiliary };

  struct ResolvedTaggingField {
    TaggingFieldKind kind = TaggingFieldKind::state;
    std::string qualified_identity;
    std::string provider_group;
  };

  struct ResolvedTaggingProgram {
    runtime::amr::PreparedTaggingProgram<Dim> program;
    std::vector<ResolvedTaggingField> fields;
  };

  struct FieldPlan;

  enum class ExternalBoundaryDependencyOperation {
    ghost_region,
    flux_transform,
    field_closure,
  };

  struct PreparedBoundaryContext {
    struct GhostRoute {
      using point_type = runtime::multiblock::BoundaryEvaluationPoint;

      const ExecutionLane* lane = nullptr;
      std::size_t target_level = 0;
      bool preserve_source_physical_ghosts = false;
      std::vector<std::unique_ptr<field_type>> images;
      PreparedRootAmrGhostFill<Dim> root_fill;
      std::vector<std::optional<runtime::amr::PreparedAmrGhostFill<Dim>>> fine_fills;
      std::vector<amr::transfer::TransferKind> fine_interpolation_kinds;
      std::vector<point_type> level_points;
      std::function<const field_type*(std::size_t)> source_at;
      std::function<void(const point_type&, std::size_t, field_type&)> prepare_physical;
      std::string collective_contract;
      std::string route_identity;
      /// Every string-bearing root/fine request is built in the all-level local prepass.  The
      /// following collective constructors only consume these retained requests.
      std::string root_identity;
      std::optional<Box<Dim>> root_domain;
      std::optional<BoundaryTopology<Dim>> request_topology;
      std::uint64_t request_topology_epoch = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t request_materialization_generation = std::numeric_limits<std::uint64_t>::max();
      std::vector<runtime::amr::AmrGhostFillPreparation<Dim>> fine_preparations;
      std::string request_contract;
      bool requests_prepared = false;
      std::optional<std::size_t> source_block;

      void prepare_requests_locally(const engine_type& engine,
                                    const BoundaryTopology<Dim>& topology,
                                    std::uint64_t topology_epoch,
                                    std::uint64_t materialization_generation,
                                    const ExecutionLane& requested_lane) {
        if (lane == nullptr || lane != &requested_lane || route_identity.empty() ||
            images.empty() || images.size() != target_level + 1 ||
            fine_interpolation_kinds.size() != target_level || !fine_fills.empty() || root_fill ||
            requests_prepared)
          throw std::logic_error("AMR GhostBoundary route request preparation is invalid");
        root_identity = route_identity + "/level/0";
        root_domain = engine.hierarchy().layout(0).domain();
        request_topology = topology;
        request_topology_epoch = topology_epoch;
        request_materialization_generation = materialization_generation;
        fine_fills.resize(target_level);
        fine_preparations.reserve(target_level);
        for (std::size_t fine_level = 1; fine_level <= target_level; ++fine_level) {
          const Box<Dim>& coarse_domain = engine.hierarchy().layout(fine_level - 1).domain();
          const Box<Dim>& fine_domain = engine.hierarchy().layout(fine_level).domain();
          fine_preparations.push_back(runtime::amr::AmrGhostFillPreparation<Dim>{
              .fine_level = static_cast<int>(fine_level),
              .coarse_domain = coarse_domain,
              .fine_domain = fine_domain,
              .ratio = engine.hierarchy().layout(fine_level).ratio_from_parent(),
              .interpolation_kind = fine_interpolation_kinds[fine_level - 1],
              .topology = topology,
              .topology_generation = topology_epoch,
              .materialization_generation = materialization_generation,
              .field_identity = route_identity + "/level/" + std::to_string(fine_level),
              .budget = exact_amr_ghost_budget(*images[fine_level - 1], *images[fine_level],
                                               coarse_domain, fine_domain, topology),
          });
        }
        ExactContractBuilder exact;
        exact.text("pops.amr.external-boundary-dependency-request")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .text(root_identity)
            .scalar(request_topology_epoch)
            .scalar(request_materialization_generation)
            .scalar(static_cast<std::uint64_t>(fine_preparations.size()));
        for (const auto& request : fine_preparations)
          exact.scalar(request.fine_level)
              .text(request.field_identity)
              .scalar(request.topology_generation)
              .scalar(request.materialization_generation);
        request_contract = std::move(exact).release();
        requests_prepared = true;
      }

      void require_collective_prepared_requests(const ExecutionLane& requested_lane) const {
        if (lane != &requested_lane || !requests_prepared || request_contract.empty())
          throw std::logic_error("AMR GhostBoundary route request was not prepared");
        if (!all_ranks_agree_exact_ordered_byte_pairs(
                {{"amr-external-boundary-dependency-request", request_contract}}, requested_lane))
          throw std::invalid_argument(
              "AMR GhostBoundary root/fine transport requests differ between MPI ranks");
      }

      void prepare_collectively(std::vector<std::string>& hierarchy_contracts,
                                const ExecutionLane& requested_lane) {
        if (lane != &requested_lane || !requests_prepared || root_identity.empty() ||
            fine_preparations.size() != target_level || fine_fills.size() != target_level ||
            root_fill || !root_domain || !request_topology)
          throw std::logic_error("AMR GhostBoundary route collective requests are incomplete");
        // No request assembly occurs below: root/fine constructors consume only retained images,
        // identities, budgets and request values prepared before the exact-lane consensus.
        root_fill = prepare_root_ghost_fill(*images[0], *root_domain, *request_topology,
                                            std::move(root_identity), request_topology_epoch,
                                            request_materialization_generation, requested_lane);
        for (std::size_t fine_level = 1; fine_level <= target_level; ++fine_level) {
          fine_fills[fine_level - 1].emplace(runtime::amr::prepare_amr_ghost_fill(
              *images[fine_level - 1], *images[fine_level],
              std::move(fine_preparations[fine_level - 1]), requested_lane));
        }
        runtime::program::collective_boundary_provider_phase(
            requested_lane, "AMR GhostBoundary route publication failed collectively", [&] {
              ExactContractBuilder exact;
              exact.text("pops.amr.external-boundary-dependency")
                  .scalar(std::uint32_t{1})
                  .scalar(std::int32_t{Dim})
                  .text(route_identity)
                  .scalar(static_cast<std::uint64_t>(target_level))
                  .scalar(preserve_source_physical_ghosts)
                  .presence(source_block.has_value())
                  .bytes(root_fill.collective_contract());
              if (source_block)
                exact.scalar(static_cast<std::uint64_t>(*source_block));
              for (const auto& fine_fill : fine_fills)
                exact.bytes(fine_fill->collective_contract());
              collective_contract = std::move(exact).release();
              hierarchy_contracts.push_back(collective_contract);
            });
      }

      [[nodiscard]] field_type& target_image() const {
        if (target_level >= images.size() || !images[target_level])
          throw std::logic_error("prepared AMR GhostBoundary route lost its target image");
        return *images[target_level];
      }

      void preflight(const point_type& point, const ExecutionLane& requested_lane) const {
        if (lane == nullptr || lane != &requested_lane || images.size() != target_level + 1 ||
            fine_fills.size() != target_level || level_points.size() != images.size() || !source_at)
          throw std::logic_error("prepared AMR GhostBoundary route is incomplete");
        for (std::size_t level = 0; level < images.size(); ++level) {
          const field_type* source = source_at(level);
          const field_type* image = images[level].get();
          if (source == nullptr || image == nullptr || !same_field_contract(*source, *image))
            throw std::logic_error(
                "prepared AMR GhostBoundary source changed its exact level contract");
          for (std::size_t local = 0; local < image->local_size(); ++local) {
            const std::size_t global = image->global_index(local);
            if (!source->contains_local(global))
              throw std::logic_error(
                  "prepared AMR GhostBoundary source lost a routed global patch");
            const Fab<Dim>& source_fab = source->fab(source->local_index_of(global));
            const Fab<Dim>& image_fab = image->fab(local);
            if (source_fab.box() != image_fab.box() ||
                source_fab.grown_box() != image_fab.grown_box())
              throw std::logic_error(
                  "prepared AMR GhostBoundary source changed routed patch storage");
          }
          if (level_points[level].clock.capacity() < point.clock.size())
            throw std::length_error(
                "prepared AMR GhostBoundary clock exceeds its prepared capacity");
        }
      }

      void materialize(const point_type& point) {
        std::exception_ptr copy_error;
        try {
          for (std::size_t level = 0; level < images.size(); ++level) {
            const field_type& source = *source_at(level);
            field_type& image = *images[level];
            if (preserve_source_physical_ghosts)
              copy_full_field_in_place(source, image);
            else {
              image.set_val(Real(0));
              copy_valid_field(source, image);
            }

            point_type& routed = level_points[level];
            routed.clock.assign(point.clock.data(), point.clock.size());
            routed.tick = point.tick;
            routed.level = static_cast<int>(level);
            routed.substep = point.substep;
            routed.stage = point.stage;
            routed.stage_fraction = point.stage_fraction;
            routed.dt = point.dt;
            routed.physical_time = point.physical_time;
          }
        } catch (...) {
          copy_error = std::current_exception();
        }
        if (all_reduce_max(copy_error ? 1L : 0L, *lane) != 0) {
          if (lane->size() == 1 && copy_error)
            std::rethrow_exception(copy_error);
          throw std::runtime_error(
              "prepared AMR GhostBoundary dependency copy failed collectively");
        }

        for (std::size_t level = 0; level < images.size(); ++level)
          runtime::program::collective_boundary_provider_phase(
              *lane, "prepared AMR GhostBoundary level transport failed collectively", [&] {
                field_type& image = *images[level];
                const point_type& routed = level_points[level];
                if (level == 0)
                  root_fill(image, routed);
                else
                  (*fine_fills[level - 1])(image, routed);
                if (prepare_physical)
                  prepare_physical(routed, level, image);
              });
      }
    };

    struct DetachedDependency {
      const field_type* source = nullptr;
      const FieldPlan* field_plan = nullptr;
      std::unique_ptr<field_type> image;
      std::shared_ptr<GhostRoute> ghost_route;
    };

    std::vector<const field_type*> states;
    std::vector<FieldDistribution> state_distributions;
    std::vector<std::string> state_identities;
    std::vector<const field_type*> fields;
    std::vector<const FieldPlan*> field_plans;
    std::size_t level = 0;
    const field_type* owning = nullptr;
    std::vector<FieldDistribution> field_distributions;
    std::vector<std::string> field_identities;
    const std::vector<Real>* parameters = nullptr;
    FieldLogicalTimePoint point{};
    mutable FieldBoundaryFailure<Dim> failure{};
    std::string clock_identity;
    runtime::multiblock::BoundaryEvaluationPoint invocation_point;
    std::vector<DetachedDependency> detached;
    const ExecutionLane* lane = nullptr;
    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();

    void prepare_dependency_transport_requests_locally(const engine_type& engine,
                                                       const BoundaryTopology<Dim>& topology,
                                                       const ExecutionLane& requested_lane) {
      if (lane != &requested_lane || topology_epoch != engine.topology_epoch() ||
          materialization_generation != engine.materialization_generation())
        throw std::logic_error("AMR external boundary dependency transport authority changed");
      for (auto& dependency : detached)
        if (dependency.ghost_route)
          dependency.ghost_route->prepare_requests_locally(
              engine, topology, topology_epoch, materialization_generation, requested_lane);
    }

    void require_collective_dependency_transport_requests(
        const ExecutionLane& requested_lane) const {
      for (const auto& dependency : detached)
        if (dependency.ghost_route)
          dependency.ghost_route->require_collective_prepared_requests(requested_lane);
    }

    [[nodiscard]] std::size_t dependency_transport_count() const noexcept {
      return static_cast<std::size_t>(
          std::count_if(detached.begin(), detached.end(), [](const DetachedDependency& dependency) {
            return static_cast<bool>(dependency.ghost_route);
          }));
    }

    void prepare_dependency_transports_collectively(std::vector<std::string>& hierarchy_contracts,
                                                    const ExecutionLane& requested_lane) {
      for (auto& dependency : detached)
        if (dependency.ghost_route)
          dependency.ghost_route->prepare_collectively(hierarchy_contracts, requested_lane);
    }

    void refresh(const FieldLogicalTimePoint& invocation_point,
                 const std::string* invocation_clock_identity,
                 const ExecutionLane& requested_lane) {
      std::exception_ptr point_error;
      try {
        if (invocation_clock_identity == nullptr || invocation_clock_identity->empty())
          throw std::invalid_argument(
              "prepared AMR boundary dependency refresh requires an exact clock identity");
        if (lane == nullptr || lane != &requested_lane)
          throw std::logic_error("prepared AMR boundary dependency lost its exact execution lane");
        if (this->invocation_point.clock.capacity() < invocation_clock_identity->size())
          throw std::length_error(
              "prepared AMR boundary clock exceeds its prepared identity capacity");
        this->invocation_point.clock.assign(invocation_clock_identity->data(),
                                            invocation_clock_identity->size());
        this->invocation_point.tick = invocation_point.step;
        this->invocation_point.level = invocation_point.level;
        this->invocation_point.substep = invocation_point.substep;
        this->invocation_point.stage = invocation_point.stage_slot;
        this->invocation_point.stage_fraction = amr::Rational{
            invocation_point.stage_fraction_numerator, invocation_point.stage_fraction_denominator};
        this->invocation_point.dt = static_cast<double>(invocation_point.dt);
        this->invocation_point.physical_time = static_cast<double>(invocation_point.time);
      } catch (...) {
        point_error = std::current_exception();
      }
      if (all_reduce_max(point_error ? 1L : 0L, requested_lane) != 0) {
        if (requested_lane.size() == 1 && point_error)
          std::rethrow_exception(point_error);
        throw std::runtime_error("prepared AMR boundary evaluation point failed collectively");
      }
      const auto& routed_point = this->invocation_point;
      const auto source_for = [&](DetachedDependency& dependency) -> const field_type* {
        const field_type* source = dependency.source;
        if (dependency.field_plan != nullptr) {
          const FieldPlan& plan = *dependency.field_plan;
          if (plan.topology_epoch != topology_epoch ||
              plan.materialization_generation != materialization_generation ||
              level >= plan.accepted_potential.size() || !plan.accepted_potential[level])
            throw std::logic_error("prepared AMR boundary field route lost its exact level");
          if (plan.candidate_ready) {
            if (!plan.prepared_solver)
              throw std::logic_error("prepared AMR boundary field candidate has no solver");
            source = &plan.prepared_solver->candidate_level(static_cast<int>(level));
          } else {
            source = plan.accepted_potential[level].get();
          }
        }
        return source;
      };

      std::exception_ptr preflight_error;
      try {
        for (auto& dependency : detached) {
          if (dependency.ghost_route) {
            dependency.ghost_route->preflight(routed_point, *lane);
            continue;
          }
          const field_type* source = source_for(dependency);
          if (source == nullptr || !dependency.image ||
              !same_field_contract(*source, *dependency.image))
            throw std::logic_error("prepared AMR boundary dependency source is absent");
          for (std::size_t local = 0; local < dependency.image->local_size(); ++local) {
            const std::size_t global = dependency.image->global_index(local);
            if (!source->contains_local(global) ||
                source->fab(source->local_index_of(global)).box() !=
                    dependency.image->fab(local).box() ||
                source->fab(source->local_index_of(global)).grown_box() !=
                    dependency.image->fab(local).grown_box())
              throw std::logic_error("prepared AMR boundary dependency patch route changed");
          }
        }
      } catch (...) {
        preflight_error = std::current_exception();
      }
      if (all_reduce_max(preflight_error ? 1L : 0L, requested_lane) != 0) {
        if (requested_lane.size() == 1 && preflight_error)
          std::rethrow_exception(preflight_error);
        throw std::runtime_error("prepared AMR boundary dependency preflight failed collectively");
      }

      std::exception_ptr copy_error;
      try {
        for (auto& dependency : detached)
          if (!dependency.ghost_route) {
            const field_type* source = source_for(dependency);
            // Flux and field closures only pack valid face/cell slices. Copying stale source ghosts
            // would incorrectly claim a transport that these operations do not consume.
            copy_valid_field(*source, *dependency.image);
          }
      } catch (...) {
        copy_error = std::current_exception();
      }
      if (all_reduce_max(copy_error ? 1L : 0L, requested_lane) != 0) {
        if (requested_lane.size() == 1 && copy_error)
          std::rethrow_exception(copy_error);
        throw std::runtime_error("prepared AMR boundary dependency copy failed collectively");
      }

      for (auto& dependency : detached)
        if (dependency.ghost_route)
          runtime::program::collective_boundary_provider_phase(
              requested_lane, "prepared AMR GhostBoundary routed dependency failed collectively",
              [&] { dependency.ghost_route->materialize(routed_point); });
    }

    FieldBoundaryExecutionContext<Dim> view(const FieldLogicalTimePoint& invocation_point,
                                            const std::string* invocation_clock_identity) const {
      for (std::size_t index = 0; index < fields.size(); ++index) {
        const field_type& dependency = *fields[index];
        if (owning == nullptr || dependency.layout() != owning->layout() ||
            dependency.local_rank() != owning->local_rank())
          throw std::logic_error("prepared AMR boundary field route changed exact-level layout");
      }
      if (states.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          fields.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          (parameters != nullptr &&
           parameters->size() > static_cast<std::size_t>(std::numeric_limits<int>::max())))
        throw std::overflow_error("AMR field boundary dependency pack exceeds native int");
      return {invocation_point,
              states.empty() ? nullptr : states.data(),
              state_distributions.empty() ? nullptr : state_distributions.data(),
              state_identities.empty() ? nullptr : state_identities.data(),
              static_cast<int>(states.size()),
              fields.empty() ? nullptr : fields.data(),
              field_distributions.empty() ? nullptr : field_distributions.data(),
              field_identities.empty() ? nullptr : field_identities.data(),
              static_cast<int>(fields.size()),
              parameters,
              parameters == nullptr ? 0 : static_cast<int>(parameters->size()),
              &failure,
              invocation_clock_identity};
    }

    void refresh_dynamic_fields() {
      if (field_plans.size() != fields.size())
        throw std::logic_error("prepared AMR dynamic boundary field table is incomplete");
      for (std::size_t index = 0; index < field_plans.size(); ++index) {
        const FieldPlan& plan = *field_plans[index];
        if (level >= plan.accepted_potential.size() || !plan.accepted_potential[level])
          throw std::logic_error("prepared AMR boundary field route lost its exact level");
        fields[index] = plan.candidate_ready
                            ? &plan.prepared_solver->candidate_level(static_cast<int>(level))
                            : plan.accepted_potential[level].get();
      }
    }

    FieldBoundaryExecutionContext<Dim> view() const {
      return view(point, clock_identity.empty() ? nullptr : &clock_identity);
    }
  };

  struct FieldPlan {
    std::string plan_identity;
    std::string provider_identity;
    std::string output_owner_identity;
    std::string output_block;
    std::string output_key;
    std::vector<FieldProviderBinding> providers;
    std::string solver_route;
    AmrFieldHierarchyPolicyAuthority hierarchy_policy;
    AmrFieldSolverOptions solver_options;
    std::optional<runtime::field::NamedFieldOutput<Dim>> output;
    std::vector<runtime::system::AuxiliaryComponentKey> output_keys;
    std::vector<std::vector<PreparedFieldRhs>> rhs_by_block;
    bool use_prepared_level_rhs = false;
    double reaction = 0.0;
    bool has_reaction = false;
    std::string nullspace_provider_identity;
    PreparedProviderOptions nullspace_options;
    std::string topology_provider_kind;
    std::string topology_provenance;
    std::string topology_digest;
    std::vector<std::string> boundary_kind;
    std::vector<double> boundary_alpha;
    std::vector<double> boundary_beta;
    std::vector<double> boundary_value;
    std::vector<std::string> boundary_state_blocks;
    std::vector<int> boundary_state_components;
    std::vector<std::string> boundary_field_blocks;
    std::vector<std::string> boundary_field_keys;
    std::vector<int> boundary_field_components;
    std::vector<Real> boundary_parameters;
    std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel;
    std::optional<FieldLogicalTimePoint> boundary_point;
    std::optional<FieldNewtonOptions> newton;

    std::unique_ptr<exact_field_solver_type> prepared_solver;
    std::vector<std::unique_ptr<field_type>> accepted_potential;
    std::vector<std::unique_ptr<field_type>> candidate_outputs;
    std::vector<auxiliary_groups_type*> candidate_provider_storage;
    std::vector<std::optional<auxiliary_publication_type>> candidate_provider_publications;
    std::vector<std::string> stale_auxiliary_providers;
    std::vector<std::unique_ptr<field_type>> contribution_scratch;
    std::vector<std::shared_ptr<const field_type>> active_coverage;
    std::shared_ptr<std::vector<PreparedBoundaryContext>> boundary_context_storage;
    std::shared_ptr<PreparedFieldBoundaryContextSet<Dim>> boundary_contexts;
    std::string prepared_contract;
    std::string prepared_nullspace_contract;
    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
    bool candidate_ready = false;

    bool materialized_for(const engine_type& engine) const noexcept {
      return prepared_solver && topology_epoch == engine.topology_epoch() &&
             materialization_generation == engine.materialization_generation();
    }

    void discard_materialization() noexcept {
      prepared_solver.reset();
      accepted_potential.clear();
      for (auto& publication : candidate_provider_publications)
        if (publication)
          publication->reject();
      candidate_outputs.clear();
      candidate_provider_storage.clear();
      candidate_provider_publications.clear();
      stale_auxiliary_providers.clear();
      contribution_scratch.clear();
      active_coverage.clear();
      boundary_context_storage.reset();
      boundary_contexts.reset();
      prepared_contract.clear();
      prepared_nullspace_contract.clear();
      topology_epoch = std::numeric_limits<std::uint64_t>::max();
      materialization_generation = std::numeric_limits<std::uint64_t>::max();
      candidate_ready = false;
    }
  };

  struct PreparedHierarchy {
    struct StageScratch {
      explicit StageScratch(const field_type& prototype)
          : backup(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                   prototype.ncomp(), prototype.ghosts()),
            candidate(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                      prototype.ncomp(), prototype.ghosts()) {
        backup.set_val(Real(0));
        candidate.set_val(Real(0));
      }

      field_type backup;
      field_type candidate;
      bool staged = false;
    };

    // The lane is declared first so every provider that pins it is destroyed before MPI_Comm_free.
    std::optional<ExecutionLane> lane;
    mutable std::recursive_mutex execution_mutex;
    /// One independent carrier per hierarchy level.  Each carrier owns storage groups keyed by
    /// the sealed owner-qualified registry; there is no shared slab or canonical component order.
    std::vector<std::unique_ptr<auxiliary_groups_type>> provider_storage;
    /// Stable transaction candidates own distinct allocations.  AMR coarse/fine kernels bind
    /// their device views at hierarchy materialization, so candidates cannot be ephemeral copies.
    std::vector<std::unique_ptr<auxiliary_groups_type>> provider_candidate_storage;
    std::vector<auxiliary_registry_type> auxiliary_registries;
    std::vector<PreparedProviderGroupsGhostFill<Dim>> provider_candidate_ghost_fills;
    std::vector<std::optional<runtime::system::PreparedAuxiliaryPhysicalBoundaries<Dim>>>
        provider_candidate_physical_boundaries;
    std::vector<std::shared_ptr<const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>>>
        embedded_boundary;
    std::vector<std::shared_ptr<const field_type>> active_coverage;
    std::vector<std::vector<level_block_type>> block_levels;
    std::vector<std::vector<std::optional<evaluation_type>>> block_evaluations;
    std::vector<std::vector<std::optional<evaluation_type>>> block_evaluation_candidates;
    std::vector<std::vector<bool>> block_evaluation_published;
    std::vector<std::vector<std::unique_ptr<StageScratch>>> block_stage_scratch;
    std::vector<std::string> external_boundary_dependency_contracts;
    std::vector<std::vector<std::vector<const Real*>>> block_state_storage;
    std::vector<std::map<std::string, std::vector<const Real*>>> provider_storage_identity;
    std::vector<std::map<std::string, std::vector<const Real*>>>
        provider_candidate_storage_identity;
    std::vector<std::string> state_field_identities;
    std::vector<std::vector<std::string>> block_state_field_identities;
    std::vector<std::string> provider_storage_field_identities;
    std::string spatial_contract;
    std::string package_contract;
    std::string collective_contract;
    std::string embedded_boundary_configuration_contract;
    std::string embedded_boundary_materialization_digest;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;

    bool matches(const engine_type& live, const multiblock_type& live_blocks,
                 std::string_view expected_package,
                 std::string_view expected_embedded) const noexcept {
      try {
        if (!lane || spatial_contract != live.spatial_contract() ||
            package_contract != expected_package ||
            embedded_boundary_configuration_contract != expected_embedded ||
            topology_epoch != live.topology_epoch() ||
            materialization_generation != live.materialization_generation() ||
            block_levels.size() != live_blocks.block_count() ||
            block_evaluations.size() != block_levels.size() ||
            block_evaluation_candidates.size() != block_levels.size() ||
            block_evaluation_published.size() != block_levels.size() ||
            block_stage_scratch.size() != block_levels.size() ||
            block_state_storage.size() != block_levels.size() ||
            block_state_field_identities.size() != block_levels.size() ||
            embedded_boundary.size() != live.hierarchy().num_levels() ||
            active_coverage.size() != live.hierarchy().num_levels() ||
            provider_storage_identity.size() != provider_storage.size() ||
            provider_candidate_storage.size() != provider_storage.size() ||
            provider_candidate_storage_identity.size() != provider_storage.size() ||
            provider_candidate_ghost_fills.size() != provider_storage.size() ||
            provider_candidate_physical_boundaries.size() != provider_storage.size() ||
            provider_storage_field_identities.size() != provider_storage.size() ||
            auxiliary_registries.size() != provider_storage.size())
          return false;
        for (std::size_t block = 0; block < block_levels.size(); ++block) {
          if (block_levels[block].size() != live.hierarchy().num_levels() ||
              block_evaluations[block].size() != live.hierarchy().num_levels() ||
              block_evaluation_candidates[block].size() != live.hierarchy().num_levels() ||
              block_evaluation_published[block].size() != live.hierarchy().num_levels() ||
              block_stage_scratch[block].size() != live.hierarchy().num_levels() ||
              block_state_storage[block].size() != live.hierarchy().num_levels() ||
              block_state_field_identities[block].size() != live.hierarchy().num_levels())
            return false;
          for (std::size_t level = 0; level < live.hierarchy().num_levels(); ++level)
            if (!block_evaluations[block][level] || !block_stage_scratch[block][level] ||
                !block_evaluation_candidates[block][level] ||
                !same_field_contract(block_evaluations[block][level]->residual,
                                     live_blocks.state(block, level)) ||
                !same_field_contract(block_evaluation_candidates[block][level]->residual,
                                     live_blocks.state(block, level)) ||
                !field_storage_matches(live_blocks.state(block, level),
                                       block_state_storage[block][level]))
              return false;
        }
        for (std::size_t level = 0; level < live.hierarchy().num_levels(); ++level) {
          if (!provider_storage[level])
            return false;
          if (provider_storage_identity[level].size() != provider_storage[level]->groups.size())
            return false;
          for (const auto& [identity, storage] : provider_storage_identity[level]) {
            const auto* group = provider_storage[level]->find(identity);
            if (group == nullptr || !field_storage_matches(*group, storage))
              return false;
          }
          if (!provider_candidate_storage[level])
            return false;
          if (provider_candidate_storage_identity[level].size() !=
              provider_candidate_storage[level]->groups.size())
            return false;
          for (const auto& [identity, storage] : provider_candidate_storage_identity[level]) {
            const auto* group = provider_candidate_storage[level]->find(identity);
            if (group == nullptr || !field_storage_matches(*group, storage))
              return false;
          }
          const bool has_groups = !provider_candidate_storage[level]->groups.empty();
          if (static_cast<bool>(provider_candidate_ghost_fills[level]) != has_groups ||
              provider_candidate_physical_boundaries[level].has_value() != has_groups)
            return false;
        }
        return true;
      } catch (...) {
        return false;
      }
    }
  };

  using provider_snapshot_type = std::vector<auxiliary_groups_type>;
  using provider_registry_snapshot_type = std::vector<auxiliary_registry_type>;
  struct AcceptedSnapshot;

  // Declared before every package-owned closure carrier so it is destroyed last.  The generated
  // DSO remains loaded until all prepared block, auxiliary and elliptic closures are gone.
  std::vector<std::shared_ptr<void>> external_package_lifetimes;
  std::vector<std::shared_ptr<pops::dynlib::handle>> native_package_lifetimes;
  AmrSystemConfig<Dim> cfg;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance;
  std::vector<BlockSpec> blocks;
  struct PreparedCouplingInstall {
    std::string provider_contract;
    CouplingOperatorView view;
    typename multiblock_type::coupling_operation_type operation;
  };
  std::vector<PreparedCouplingInstall> prepared_couplings;
  boundary_registry_type boundary_registry;
  struct PreparedBoundaryComponents {
    struct FieldPair {
      std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual;
      std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp;
    };
    std::vector<std::shared_ptr<PreparedGhostBoundaryComponent>> ghosts;
    std::vector<std::shared_ptr<PreparedBoundaryFluxComponent>> fluxes;
    std::map<std::string, FieldPair> fields;
  };
  std::map<std::string, PreparedBoundaryComponents> prepared_boundary_components;
  std::shared_ptr<const component::PreparedExecutionContextV1> boundary_execution_context;
  std::shared_ptr<exact_field_registry_type> field_solver_providers;
  std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers;
  std::shared_ptr<hierarchy_tensor_registry_type> hierarchy_tensor_solver_providers;
  mutable std::map<std::string, FieldPlan> field_plans;
  std::string default_field_slot;
  std::string default_nullspace_provider_identity;
  PreparedProviderOptions default_nullspace_options;
  mutable std::string active_field_slot;
  runtime::program::ProgramRuntimeState<Dim> program;
  runtime::system::SystemLifecycle lifecycle;
  mutable std::shared_ptr<engine_type> engine;
  std::map<std::string, BootstrapTransferRoute> bootstrap_transfer_routes;
  std::map<std::pair<std::string, std::string>, std::string> bootstrap_subject_routes;
  std::map<std::string, std::array<std::string, Dim>> bootstrap_oriented_face_groups;
  std::vector<::pops::amr::ParentChildClockRelation> temporal_relations;
  std::vector<prepared_block_type> prepared_blocks;
  mutable std::unique_ptr<multiblock_type> multiblock_hierarchy;
  mutable std::optional<typename multiblock_type::ProgramBlockMap> prepared_program_block_map;
  mutable std::optional<flux_expression_budget_type> program_flux_expression_budget;
  std::vector<std::string> embedded_boundary_opcodes;
  std::vector<double> embedded_boundary_literals;
  runtime::system::PreparedEmbeddedBoundaryMode embedded_boundary_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  EbThresholds embedded_boundary_thresholds{};
  std::uint64_t embedded_boundary_generation = 0;
  std::string embedded_boundary_configuration_contract;
  std::string embedded_boundary_semantic_digest;
  mutable std::unique_ptr<PreparedHierarchy> prepared_hierarchy;
  mutable std::vector<const std::vector<field_type>*> program_hierarchy_candidates;
  mutable std::shared_ptr<const provider_snapshot_type> pending_provider_restore;
  mutable std::shared_ptr<const provider_registry_snapshot_type> pending_provider_registry_restore;
  auxiliary_registry_type auxiliary_registry;
  std::map<std::string, std::vector<double>> staged_auxiliary_inputs;
  std::vector<std::string> dirty_auxiliary_providers;
  bool auxiliary_registry_consensus_verified = false;
  std::vector<GlobalDtBound> dt_bounds;
  double accepted_time = 0.0;
  int macro_step = 0;
  mutable int checkpoint_regrid_count_value = 0;
  mutable std::vector<int> last_replay_regrid_steps;
  std::string last_dt_reason;
  mutable std::vector<std::uint8_t> program_accepted_bytes;
  mutable std::uint64_t program_accepted_revision = 0;
  mutable bool program_accepted_bytes_runtime_owned = false;
  std::optional<TaggingSpec> tagging_spec;
  struct TaggerComponentAuthority {
    std::shared_ptr<component::LoadedComponent> component{};
    runtime::amr::PreparedTaggerComponentSpec spec{};
  };
  std::optional<TaggerComponentAuthority> tagger_component;
  mutable std::optional<ResolvedTaggingProgram> resolved_tagging;
  mutable std::unique_ptr<runtime::amr::PreparedTaggingExecutionPlan<Dim>> tagging_plan;
  mutable std::unique_ptr<runtime::amr::PreparedTaggerComponent<Dim>> component_tagging_plan;
  mutable runtime::amr::PersistentTaggingState<Dim> tagging_state;
  mutable std::unique_ptr<AcceptedSnapshot> bootstrap_transaction;
  mutable std::set<std::pair<std::string, int>> bootstrap_materialized_actions;
  mutable bool automatic_bootstrap_complete = false;

  struct AcceptedSnapshot {
    std::optional<typename engine_type::Snapshot> engine;
    std::optional<typename multiblock_type::Snapshot> multiblock;
    std::shared_ptr<const provider_snapshot_type> provider_storage;
    std::shared_ptr<const provider_registry_snapshot_type> provider_registries;
    runtime::program::ProgramRuntimeState<Dim> program;
    std::optional<flux_expression_budget_type> program_flux_expression_budget;
    std::vector<::pops::amr::ParentChildClockRelation> temporal_relations;
    double accepted_time = 0.0;
    int macro_step = 0;
    int checkpoint_regrid_count_value = 0;
    std::vector<int> last_replay_regrid_steps;
    std::vector<std::uint8_t> program_accepted_bytes;
    std::uint64_t program_accepted_revision = 0;
    bool program_accepted_bytes_runtime_owned = false;
    std::map<std::string, std::vector<field_type>> field_potentials;
    std::set<std::string> field_plan_slots;
    runtime::amr::PersistentTaggingState<Dim> tagging_state;
    std::set<std::pair<std::string, int>> bootstrap_materialized_actions;
    bool automatic_bootstrap_complete = false;

    explicit AcceptedSnapshot(const Impl& owner)
        : engine(owner.engine
                     ? std::optional<typename engine_type::Snapshot>(owner.engine->snapshot())
                     : std::nullopt),
          multiblock(owner.multiblock_hierarchy ? std::optional<typename multiblock_type::Snapshot>(
                                                      owner.multiblock_hierarchy->snapshot())
                                                : std::nullopt),
          provider_storage(owner.snapshot_provider_storage()),
          provider_registries(owner.snapshot_provider_registries()),
          program(owner.program),
          program_flux_expression_budget(owner.program_flux_expression_budget),
          temporal_relations(owner.temporal_relations),
          accepted_time(owner.accepted_time),
          macro_step(owner.macro_step),
          checkpoint_regrid_count_value(owner.checkpoint_regrid_count_value),
          last_replay_regrid_steps(owner.last_replay_regrid_steps),
          program_accepted_bytes(owner.program_accepted_bytes),
          program_accepted_revision(owner.program_accepted_revision),
          program_accepted_bytes_runtime_owned(owner.program_accepted_bytes_runtime_owned),
          tagging_state(owner.tagging_state),
          bootstrap_materialized_actions(owner.bootstrap_materialized_actions),
          automatic_bootstrap_complete(owner.automatic_bootstrap_complete) {
      if (!owner.active_field_slot.empty())
        throw std::logic_error(
            "AmrSystem cannot snapshot an unconsumed exact field solve candidate");
      for (const auto& [slot, plan] : owner.field_plans) {
        field_plan_slots.insert(slot);
        if (plan.accepted_potential.empty())
          continue;
        auto& levels = field_potentials[slot];
        levels.reserve(plan.accepted_potential.size());
        for (const auto& level : plan.accepted_potential) {
          if (!level)
            throw std::logic_error(
                "AmrSystem materialized field plan contains an empty accepted level");
          levels.push_back(*level);
        }
      }
    }

    struct PreparedFieldRestore {
      FieldPlan* plan = nullptr;
      bool retain_materialization = false;
      std::vector<std::unique_ptr<field_type>> accepted_potential;
    };

    struct PreparedAcceptedRestore {
      std::optional<typename multiblock_type::ProgramBlockMap> program_block_map;
      std::optional<flux_expression_budget_type> program_flux_expression_budget;
      std::vector<::pops::amr::ParentChildClockRelation> temporal_relations;
      double accepted_time = 0.0;
      int macro_step = 0;
      int checkpoint_regrid_count_value = 0;
      std::vector<int> last_replay_regrid_steps;
      std::vector<std::uint8_t> program_accepted_bytes;
      std::uint64_t program_accepted_revision = 0;
      bool program_accepted_bytes_runtime_owned = false;
      std::shared_ptr<const provider_snapshot_type> pending_provider_restore;
      std::shared_ptr<const provider_registry_snapshot_type> pending_provider_registry_restore;
      std::vector<PreparedFieldRestore> fields;
      runtime::amr::PersistentTaggingState<Dim> tagging_state;
      std::set<std::pair<std::string, int>> bootstrap_materialized_actions;
      bool automatic_bootstrap_complete = false;
      std::string active_field_slot;
      std::optional<typename multiblock_type::PreparedRestore> carrier_restore;
      std::optional<
          typename runtime::program::ProgramRuntimeState<Dim>::PreparedProgramAcceptedRestore>
          program_restore;

      PreparedAcceptedRestore(const AcceptedSnapshot& snapshot, Impl& owner)
          : program_flux_expression_budget(snapshot.program_flux_expression_budget),
            temporal_relations(snapshot.temporal_relations),
            accepted_time(snapshot.accepted_time),
            macro_step(snapshot.macro_step),
            checkpoint_regrid_count_value(snapshot.checkpoint_regrid_count_value),
            last_replay_regrid_steps(snapshot.last_replay_regrid_steps),
            program_accepted_bytes(snapshot.program_accepted_bytes),
            program_accepted_revision(snapshot.program_accepted_revision),
            program_accepted_bytes_runtime_owned(snapshot.program_accepted_bytes_runtime_owned),
            pending_provider_restore(snapshot.provider_storage),
            pending_provider_registry_restore(snapshot.provider_registries),
            tagging_state(snapshot.tagging_state),
            bootstrap_materialized_actions(snapshot.bootstrap_materialized_actions),
            automatic_bootstrap_complete(snapshot.automatic_bootstrap_complete) {
        const bool snapshot_materialized = snapshot.engine && snapshot.multiblock;
        const bool owner_materialized = owner.engine && owner.multiblock_hierarchy;
        if (snapshot_materialized != owner_materialized ||
            snapshot.engine.has_value() != snapshot.multiblock.has_value() ||
            static_cast<bool>(owner.engine) != static_cast<bool>(owner.multiblock_hierarchy))
          throw std::logic_error(
              "AmrSystem accepted restore changed engine/carrier materialization");
        if (owner.field_plans.size() != snapshot.field_plan_slots.size())
          throw std::logic_error("AmrSystem transaction changed its field-plan registry");
        for (const std::string& slot : snapshot.field_plan_slots)
          if (!owner.field_plans.contains(slot))
            throw std::logic_error("AmrSystem transaction changed its field-plan registry");

        if (snapshot_materialized && !snapshot.program.block_map_.empty()) {
          if (snapshot.program.block_map_.size() != owner.blocks.size() ||
              snapshot.multiblock->additional.size() + 1 != owner.blocks.size())
            throw std::invalid_argument(
                "AMR Program block map must cover every restored carrier exactly once");
          typename multiblock_type::ProgramBlockMap candidate;
          candidate.canonical_indices.reserve(snapshot.program.block_map_.size());
          std::vector<bool> seen(owner.blocks.size(), false);
          for (const int runtime_block : snapshot.program.block_map_) {
            if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= owner.blocks.size())
              throw std::out_of_range(
                  "AMR restored Program block map contains an invalid runtime block");
            const std::size_t canonical = static_cast<std::size_t>(runtime_block);
            if (seen[canonical])
              throw std::invalid_argument(
                  "AMR restored Program block map contains a duplicate block");
            seen[canonical] = true;
            if (canonical != 0 && snapshot.multiblock->additional[canonical - 1].identity !=
                                      owner.blocks[canonical].name)
              throw std::invalid_argument(
                  "AMR restored Program block map differs from its carrier identity");
            candidate.canonical_indices.push_back(canonical);
          }
          candidate.hierarchy_contract = snapshot.multiblock->exact_collective_contract;
          ExactContractBuilder exact;
          exact.text("pops.prepared-multiblock-amr.program-map")
              .scalar(std::uint32_t{1})
              .scalar(std::int32_t{Dim})
              .bytes(candidate.hierarchy_contract)
              .scalar(static_cast<std::uint64_t>(candidate.canonical_indices.size()));
          for (const std::size_t canonical : candidate.canonical_indices)
            exact.text(owner.blocks[canonical].name).scalar(static_cast<std::uint64_t>(canonical));
          candidate.exact_contract = std::move(exact).release();
          program_block_map.emplace(std::move(candidate));
        }
        if (program_flux_expression_budget) {
          if (!snapshot_materialized || !program_block_map ||
              program_flux_expression_budget->program_block_map.exact_contract !=
                  program_block_map->exact_contract)
            throw std::invalid_argument(
                "AMR restored flux-expression budget lost its exact Program block map");
        }

        fields.reserve(snapshot.field_plan_slots.size());
        for (const std::string& slot : snapshot.field_plan_slots) {
          FieldPlan& plan = owner.field_plans.at(slot);
          PreparedFieldRestore candidate{.plan = &plan};
          const auto saved = snapshot.field_potentials.find(slot);
          if (snapshot_materialized && saved != snapshot.field_potentials.end() &&
              plan.prepared_solver && plan.topology_epoch == snapshot.engine->topology_epoch &&
              plan.materialization_generation == snapshot.engine->materialization_generation) {
            if (plan.accepted_potential.size() != saved->second.size())
              throw std::logic_error("AmrSystem transaction changed a materialized field plan");
            candidate.accepted_potential.reserve(saved->second.size());
            for (const field_type& level : saved->second)
              candidate.accepted_potential.push_back(std::make_unique<field_type>(level));
            candidate.retain_materialization = true;
          }
          fields.push_back(std::move(candidate));
        }
        if (!snapshot_materialized && !snapshot.field_potentials.empty())
          throw std::logic_error(
              "AmrSystem unmaterialized restore contains accepted field storage");
        if (snapshot_materialized)
          carrier_restore.emplace(
              owner.multiblock_hierarchy->prepare_restore(*snapshot.multiblock));
        program_restore.emplace(owner.program.prepare_accepted_restore(snapshot.program));
      }
    };

    static void publish_prepared_restore(Impl& owner, PreparedAcceptedRestore&& prepared) noexcept {
      static_assert(std::is_nothrow_move_assignable_v<decltype(owner.prepared_program_block_map)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.program_flux_expression_budget)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.temporal_relations)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.last_replay_regrid_steps)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.program_accepted_bytes)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.tagging_state)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.bootstrap_materialized_actions)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.pending_provider_restore)>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.pending_provider_registry_restore)>);
      static_assert(std::is_nothrow_swappable_v<std::vector<std::unique_ptr<field_type>>>);
      static_assert(std::is_nothrow_swappable_v<decltype(owner.active_field_slot)>);
      if (!prepared.program_restore)
        std::terminate();
      if (prepared.carrier_restore) {
        if (!owner.multiblock_hierarchy)
          std::terminate();
        owner.multiblock_hierarchy->publish_prepared_restore(std::move(*prepared.carrier_restore));
      }
      owner.prepared_hierarchy.reset();
      owner.program.publish_prepared_accepted_restore(std::move(*prepared.program_restore));
      owner.prepared_program_block_map = std::move(prepared.program_block_map);
      owner.program_flux_expression_budget.swap(prepared.program_flux_expression_budget);
      owner.temporal_relations.swap(prepared.temporal_relations);
      owner.accepted_time = prepared.accepted_time;
      owner.macro_step = prepared.macro_step;
      owner.checkpoint_regrid_count_value = prepared.checkpoint_regrid_count_value;
      owner.last_replay_regrid_steps.swap(prepared.last_replay_regrid_steps);
      owner.program_accepted_bytes.swap(prepared.program_accepted_bytes);
      owner.program_accepted_revision = prepared.program_accepted_revision;
      owner.program_accepted_bytes_runtime_owned = prepared.program_accepted_bytes_runtime_owned;
      owner.pending_provider_restore.swap(prepared.pending_provider_restore);
      owner.pending_provider_registry_restore.swap(prepared.pending_provider_registry_restore);
      std::swap(owner.tagging_state, prepared.tagging_state);
      owner.bootstrap_materialized_actions.swap(prepared.bootstrap_materialized_actions);
      owner.automatic_bootstrap_complete = prepared.automatic_bootstrap_complete;
      owner.tagging_plan.reset();
      owner.component_tagging_plan.reset();
      for (PreparedFieldRestore& field : prepared.fields) {
        if (!field.retain_materialization)
          field.plan->discard_materialization();
        else {
          field.plan->accepted_potential.swap(field.accepted_potential);
          field.plan->candidate_ready = false;
        }
      }
      owner.active_field_slot.swap(prepared.active_field_slot);
    }

    void restore(Impl& owner) {
      if (!owner.multiblock_hierarchy) {
        // Assembly-time artifact rollback has no materialized carrier or execution lane. Its outer
        // installer already owns rank consensus; this branch restores only locally prepared
        // metadata and Program accepted state through the same noexcept publication boundary.
        PreparedAcceptedRestore prepared(*this, owner);
        publish_prepared_restore(owner, std::move(prepared));
        return;
      }
      const CommunicatorView communicator = owner.multiblock_hierarchy->lane().communicator();
      const auto collectively_rethrow = [&](const std::exception_ptr& local_error,
                                            std::string_view phase) {
        const long failures = all_reduce_sum(local_error ? 1L : 0L, communicator);
        if (failures == 0)
          return;
        if (communicator.size() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("AmrSystem accepted restore " + std::string(phase) +
                                 " failed collectively on " + std::to_string(failures) +
                                 " rank(s)");
      };

      std::unique_ptr<PreparedAcceptedRestore> prepared;
      std::exception_ptr local_error;
      try {
        prepared = std::make_unique<PreparedAcceptedRestore>(*this, owner);
      } catch (...) {
        local_error = std::current_exception();
      }
      collectively_rethrow(local_error, "preparation");

      local_error = {};
      try {
        owner.multiblock_hierarchy->execute_prepared_restore(*prepared->carrier_restore);
      } catch (...) {
        local_error = std::current_exception();
      }
      collectively_rethrow(local_error, "carrier execution");

      publish_prepared_restore(owner, std::move(*prepared));
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction;
  bool external_step_committed = false;
  std::unique_ptr<AcceptedSnapshot> restart_transaction;

  explicit Impl(const AmrSystemConfig<Dim>& config)
      : cfg(config),
        load_balance(std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
            prepare_load_balance_authority<Dim>(cfg.load_balance_route, cfg.load_balance_identity,
                                                cfg.load_balance_options))),
        field_solver_providers(std::make_shared<exact_field_registry_type>()),
        field_nullspace_providers(make_default_field_nullspace_provider_registry<Dim>()),
        hierarchy_tensor_solver_providers(
            std::make_shared<runtime::program::HierarchyTensorSolverProviderRegistry<Dim>>()) {
    field_solver_providers->add(runtime::amr::make_builtin_exact_amr_field_solver_provider<Dim>());
    const FieldNullspaceProviderSelection selection = operator_topology_zero_mean_nullspace();
    default_nullspace_provider_identity = selection.provider_identity;
    default_nullspace_options = selection.options;
  }

  BlockSpec& block(const std::string& name) {
    for (BlockSpec& candidate : blocks)
      if (candidate.name == name || (blocks.size() == 1 && name.empty()))
        return candidate;
    throw std::runtime_error("AmrSystem has no block named '" + name + "'");
  }

  const BlockSpec& block(const std::string& name) const {
    return const_cast<Impl*>(this)->block(name);
  }

  std::size_t block_index(const std::string& name) const {
    const BlockSpec& selected = block(name);
    return static_cast<std::size_t>(&selected - blocks.data());
  }

  field_type& block_state(std::size_t block, std::size_t level) const {
    if (!multiblock_hierarchy)
      throw std::logic_error("AmrSystem multi-block carrier is not materialized");
    return multiblock_hierarchy->state(block, level);
  }

  ExecutionCommunicator boundary_parent_communicator() const {
    if (!boundary_execution_context)
      throw std::logic_error("AMR boundary operation has no RuntimeInstance communicator");
    const PopsExecutionContextV1 execution = boundary_execution_context->view();
#ifdef POPS_HAS_MPI
    return ExecutionCommunicator::borrowed(
        execution.communicator_identity,
        MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle)));
#else
    if (execution.communicator_f_handle != 0 || execution.communicator_datatype_f_handle != 0 ||
        std::string_view(execution.communicator_identity) != "serial")
      throw std::invalid_argument("serial AMR boundary operation requires serial authority");
    return ExecutionCommunicator::world();
#endif
  }

  std::string prepared_package_contract() const {
    ExactContractBuilder packages;
    packages.text("pops.amr-system.prepared-package-registry")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(prepared_blocks.size()));
    for (const auto& block : prepared_blocks)
      packages.text(block.name).bytes(block.collective_contract);
    return std::move(packages).release();
  }

  std::optional<typename multiblock_type::ProgramBlockMap> prepare_program_block_map_candidate(
      const multiblock_type& carrier) const {
    if (program.block_map_.empty())
      return std::nullopt;
    if (program.block_map_.size() != blocks.size())
      throw std::invalid_argument(
          "AMR Program block map must cover every prepared carrier exactly once");
    std::vector<std::string> ordered;
    ordered.reserve(program.block_map_.size());
    for (const int runtime_block : program.block_map_) {
      if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= blocks.size())
        throw std::out_of_range("AMR Program block map contains an invalid runtime block");
      ordered.push_back(blocks[static_cast<std::size_t>(runtime_block)].name);
    }
    return carrier.prepare_program_block_map(ordered);
  }

  void prepare_program_block_map() const {
    std::optional<typename multiblock_type::ProgramBlockMap> candidate;
    if (multiblock_hierarchy)
      candidate = prepare_program_block_map_candidate(*multiblock_hierarchy);
    prepared_program_block_map = std::move(candidate);
  }

  static bool flux_expression_budget_is_active(
      const std::vector<flux_expression_block_budget_type>& blocks) noexcept {
    return std::any_of(blocks.begin(), blocks.end(), [](const auto& block) {
      return block.rhs_basis_bound != 0 || block.coefficient_term_bound != 0;
    });
  }

  flux_expression_budget_type prepare_program_flux_expression_budget(
      std::string program_hash, std::vector<flux_expression_block_budget_type> blocks,
      const typename multiblock_type::ProgramBlockMap& block_map, bool has_flux_expression,
      const engine_type& prepared_engine, const multiblock_type& prepared_carrier) const {
    const ExecutionLane& lane = prepared_carrier.lane();
    flux_expression_budget_type candidate;
    std::exception_ptr local_error;
    try {
      if (program_hash.empty())
        throw std::invalid_argument(
            "AMR Program flux-expression budget requires a non-empty Program hash");
      if (block_map.canonical_indices.size() != prepared_carrier.block_count() ||
          block_map.hierarchy_contract != prepared_carrier.collective_contract() ||
          block_map.exact_contract.empty())
        throw std::invalid_argument(
            "AMR Program flux-expression budget differs from the exact Program block map");
      if (blocks.size() != block_map.canonical_indices.size())
        throw std::invalid_argument(
            "AMR Program flux-expression budget count differs from the Program block count");

      std::vector<bool> seen(prepared_carrier.block_count(), false);
      for (const std::size_t canonical : block_map.canonical_indices) {
        if (canonical >= seen.size() || seen[canonical])
          throw std::invalid_argument(
              "AMR Program flux-expression budget has a malformed exact block map");
        seen[canonical] = true;
      }
      bool any_active_block = false;
      for (const auto& block : blocks) {
        const bool active = block.rhs_basis_bound != 0 || block.coefficient_term_bound != 0;
        if (active && (block.rhs_basis_bound == 0 || block.coefficient_term_bound == 0))
          throw std::invalid_argument(
              "AMR Program flux-expression budget must provide both finite bounds per block");
        any_active_block = any_active_block || active;
        if (block.rhs_basis_bound >
            std::numeric_limits<std::size_t>::max() - block.coefficient_term_bound)
          throw std::overflow_error("AMR Program flux-expression budget sum overflows size_t");
      }
      if (has_flux_expression != any_active_block)
        throw std::invalid_argument(
            "AMR Program flux-expression flag differs from its per-block budgets");

      ExactContractBuilder exact;
      exact.text("pops.amr-system.prepared-program-flux-expression-budget")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .text(program_hash)
          .scalar(program.step_install_generation_)
          .scalar(prepared_engine.topology_epoch())
          .scalar(prepared_engine.materialization_generation())
          .bytes(prepared_carrier.collective_contract())
          .bytes(block_map.exact_contract)
          .scalar(has_flux_expression)
          .sequence(blocks, [](ExactContractBuilder& entry, const auto& block) {
            entry.scalar(static_cast<std::uint64_t>(block.rhs_basis_bound))
                .scalar(static_cast<std::uint64_t>(block.coefficient_term_bound));
          });
      candidate.program_hash = std::move(program_hash);
      candidate.generation = prepared_engine.materialization_generation();
      candidate.program_block_map = block_map;
      candidate.blocks = std::move(blocks);
      candidate.exact_contract = std::move(exact).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane.communicator()) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "AMR Program flux-expression budget preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("amr-program-flux-expression-budget"),
              std::string_view(candidate.exact_contract)}},
            lane.communicator()))
      throw std::invalid_argument(
          "AMR Program flux-expression budgets differ between prepared-lane ranks");
    return candidate;
  }

  std::shared_ptr<const provider_snapshot_type> snapshot_provider_storage() const {
    if (!prepared_hierarchy)
      return {};
    auto snapshot = std::make_shared<provider_snapshot_type>();
    snapshot->reserve(prepared_hierarchy->provider_storage.size());
    for (const auto& level : prepared_hierarchy->provider_storage) {
      if (!level)
        throw std::logic_error("prepared AMR hierarchy contains an empty provider carrier");
      snapshot->push_back(*level);
    }
    return snapshot;
  }

  std::shared_ptr<const provider_registry_snapshot_type> snapshot_provider_registries() const {
    if (!prepared_hierarchy)
      return {};
    return std::make_shared<provider_registry_snapshot_type>(
        prepared_hierarchy->auxiliary_registries);
  }

  BoundaryTopology<Dim> topology() const {
    return BoundaryTopology<Dim>::axis_periodic(cfg.periodicity);
  }

  static std::string exact_field_plan_contract(std::string_view slot, const FieldPlan& plan) {
    ExactContractBuilder contract;
    contract.text("pops.amr.exact-ranked-field-plan")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .text(slot)
        .text(plan.plan_identity)
        .text(plan.provider_identity)
        .text(plan.output_owner_identity)
        .text(plan.output_block)
        .text(plan.output_key)
        .sequence(plan.providers,
                  [](ExactContractBuilder& item, const FieldProviderBinding& binding) {
                    item.text(binding.identity)
                        .text(binding.block)
                        .text(binding.key)
                        .scalar(binding.coefficient);
                  })
        .text(plan.solver_route)
        .bytes(plan.hierarchy_policy.exact_contract())
        .bytes(plan.solver_options.exact_contract())
        .presence(plan.has_reaction);
    if (plan.has_reaction)
      contract.scalar(plan.reaction);
    contract.text(plan.nullspace_provider_identity)
        .presence(!plan.nullspace_provider_identity.empty());
    if (!plan.nullspace_provider_identity.empty())
      contract.bytes(plan.nullspace_options.exact_contract());
    contract.text(plan.topology_provider_kind)
        .text(plan.topology_provenance)
        .text(plan.topology_digest)
        .sequence(plan.boundary_kind,
                  [](ExactContractBuilder& item, const std::string& kind) { item.text(kind); })
        .sequence(plan.boundary_alpha)
        .sequence(plan.boundary_beta)
        .sequence(plan.boundary_value)
        .sequence(plan.boundary_state_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_state_components)
        .sequence(plan.boundary_field_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_keys,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_components)
        .sequence(plan.boundary_parameters)
        .presence(plan.output.has_value());
    if (plan.output) {
      contract.scalar(plan.output->gradient_sign())
          .scalar(static_cast<std::uint64_t>(plan.output->component_count()))
          .sequence(plan.output_keys, [](ExactContractBuilder& item,
                                         const runtime::system::AuxiliaryComponentKey& key) {
            key.serialize_exact(item);
          });
    }
    contract.presence(plan.use_prepared_level_rhs).presence(plan.boundary_kernel.has_value());
    if (plan.boundary_kernel)
      contract.text(plan.boundary_kernel->identity)
          .text(plan.boundary_kernel->residual_identity)
          .text(plan.boundary_kernel->jvp_identity)
          .presence(plan.boundary_kernel->observes_iteration);
    contract.presence(plan.newton.has_value());
    if (plan.newton)
      contract.scalar(plan.newton->tolerance)
          .scalar(plan.newton->max_iterations)
          .scalar(plan.newton->linear_tolerance)
          .scalar(plan.newton->linear_max_iterations)
          .scalar(plan.newton->restart)
          .scalar(plan.newton->armijo)
          .scalar(plan.newton->minimum_step);
    return std::move(contract).release();
  }

  FieldPlan& resolve_field_plan(std::string_view field) {
    auto found = field_plans.find(std::string(field));
    if (found != field_plans.end())
      return found->second;
    for (auto& [slot, plan] : field_plans) {
      (void)slot;
      if (plan.output_key == field)
        return plan;
    }
    throw std::out_of_range("AmrSystem has no exact field route '" + std::string(field) + "'");
  }

  const FieldPlan& resolve_field_plan(std::string_view field) const {
    return const_cast<Impl*>(this)->resolve_field_plan(field);
  }

  std::string resolve_field_slot(std::string_view field) const {
    const auto exact = field_plans.find(std::string(field));
    if (exact != field_plans.end())
      return exact->first;
    for (const auto& [slot, plan] : field_plans)
      if (plan.output_key == field)
        return slot;
    throw std::out_of_range("AmrSystem has no exact field route '" + std::string(field) + "'");
  }

  static Extent<Dim> unit_ghosts() {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = 1;
    return result;
  }

  PhysicalBoundaryConditions<Dim> field_boundary(const FieldPlan& plan,
                                                 const Geometry<Dim>& geometry) const {
    const BoundaryTopology<Dim> exact_topology = topology();
    std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
    const std::size_t face_count = static_cast<std::size_t>(2 * Dim);
    if (!plan.boundary_kind.empty() &&
        (plan.boundary_kind.size() != face_count || plan.boundary_alpha.size() != face_count ||
         plan.boundary_beta.size() != face_count || plan.boundary_value.size() != face_count))
      throw std::invalid_argument(
          "AMR field boundary plan must cover both faces of every exact axis");

    for (int axis = 0; axis < Dim; ++axis)
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        const bool periodic = exact_topology.is_periodic(face);
        const std::string kind = plan.boundary_kind.empty() ? (periodic ? "periodic" : "dirichlet")
                                                            : plan.boundary_kind[ordinal];
        if ((kind == "periodic") != periodic)
          throw std::invalid_argument(
              "AMR field boundary periodicity differs from the hierarchy topology");
        if (periodic) {
          faces[ordinal] = {};
          continue;
        }
        const Real value =
            plan.boundary_value.empty() ? Real(0) : static_cast<Real>(plan.boundary_value[ordinal]);
        if (kind == "dirichlet")
          faces[ordinal] = {PhysicalBoundaryKind::dirichlet, value, Real(1), Real(0)};
        else if (kind == "neumann")
          faces[ordinal] = {PhysicalBoundaryKind::neumann, value, Real(0), Real(1)};
        else if (kind == "mixed")
          faces[ordinal] = {PhysicalBoundaryKind::robin, value,
                            static_cast<Real>(plan.boundary_alpha[ordinal]),
                            static_cast<Real>(plan.boundary_beta[ordinal])};
        else
          throw std::invalid_argument("AMR field boundary kind is unknown");
      }

    RealVector<Dim> spacing{};
    for (int axis = 0; axis < Dim; ++axis)
      spacing[axis] = geometry.spacing(axis);
    return PhysicalBoundaryConditions<Dim>(exact_topology, faces, spacing);
  }

  static std::vector<std::shared_ptr<const field_type>> prepare_active_coverage(
      const engine_type& source, std::span<const int> selected_levels) {
    const auto& hierarchy = source.hierarchy();
    if (selected_levels.empty())
      throw std::invalid_argument("AMR composite coverage requires at least one selected level");
    int previous = -1;
    for (const int level : selected_levels) {
      if (level < 0 || static_cast<std::size_t>(level) >= hierarchy.num_levels() ||
          level <= previous)
        throw std::invalid_argument(
            "AMR composite coverage requires strictly increasing live hierarchy levels");
      previous = level;
    }

    std::vector<std::shared_ptr<field_type>> writable;
    writable.reserve(selected_levels.size());
    for (const int level : selected_levels) {
      const field_type& state = hierarchy.state(static_cast<std::size_t>(level));
      auto mask = std::make_shared<field_type>(state.layout(), state.distribution(),
                                               state.local_rank(), 1, Extent<Dim>{});
      mask->set_val(Real(1));
      writable.push_back(std::move(mask));
    }
    for (std::size_t position = 1; position < selected_levels.size(); ++position) {
      const int coarse_level = selected_levels[position - 1];
      const int fine_level = selected_levels[position];
      field_type& parent = *writable[position - 1];
      const auto& fine_layout = hierarchy.layout(static_cast<std::size_t>(fine_level));
      for (const Box<Dim>& fine_patch : fine_layout.patches().boxes()) {
        Box<Dim> footprint = fine_patch;
        for (int current = fine_level; current > coarse_level; --current)
          footprint = amr::hierarchy::coarsen_box(
              footprint, hierarchy.layout(static_cast<std::size_t>(current)).ratio_from_parent());
        for (std::size_t local = 0; local < parent.local_size(); ++local) {
          const Box<Dim> overlap = parent.box(local).intersect(footprint);
          if (overlap.empty())
            continue;
          const auto values = parent.fab(local).view();
          for_each_cell(overlap,
                        [=] POPS_HD(const Index<Dim>& cell) { values(cell, 0) = Real(0); });
        }
      }
    }
    Kokkos::fence();
    std::vector<std::shared_ptr<const field_type>> result;
    result.reserve(writable.size());
    for (auto& mask : writable)
      result.push_back(std::move(mask));
    return result;
  }

  static std::vector<std::shared_ptr<const field_type>> prepare_active_coverage(
      const engine_type& source) {
    std::vector<int> levels;
    levels.reserve(source.hierarchy().num_levels());
    for (std::size_t level = 0; level < source.hierarchy().num_levels(); ++level)
      levels.push_back(static_cast<int>(level));
    return prepare_active_coverage(source, levels);
  }

  const std::vector<std::shared_ptr<const field_type>>& active_coverage() const {
    if (!prepared_hierarchy ||
        prepared_hierarchy->active_coverage.size() != engine->hierarchy().num_levels())
      throw std::logic_error("AMR composite coverage is not prepared for the live hierarchy");
    return prepared_hierarchy->active_coverage;
  }

  static void copy_scalar_component(const field_type& source, int source_component,
                                    field_type& destination, int destination_component) {
    if (source.layout() != destination.layout() ||
        source.distribution() != destination.distribution() ||
        source.local_rank() != destination.local_rank() || source_component < 0 ||
        source_component >= source.ncomp() || destination_component < 0 ||
        destination_component >= destination.ncomp())
      throw std::invalid_argument(
          "AMR field component copy requires one exact layout and valid components");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto input = source.fab(local).view();
      const auto output = destination.fab(local).view();
      for_each_cell(source.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        output(cell, destination_component) = input(cell, source_component);
      });
    }
    Kokkos::fence();
  }

  FieldNullspaceOperatorFacts nullspace_operator_facts(const FieldPlan& plan) const {
    std::vector<FieldBoundaryNullspaceFact> facts;
    facts.reserve(static_cast<std::size_t>(2 * Dim));
    ExactContractBuilder identity;
    identity.text("pops.amr.field-boundary-set").scalar(std::uint32_t{1}).scalar(std::int32_t{Dim});
    const BoundaryTopology<Dim> exact_topology = topology();
    for (int axis = 0; axis < Dim; ++axis)
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        const std::string face_identity =
            "axis:" + std::to_string(axis) + (side == BoundarySide::lower ? ":lower" : ":upper");
        const std::string kind = plan.boundary_kind.empty()
                                     ? (exact_topology.is_periodic(face) ? "periodic" : "dirichlet")
                                     : plan.boundary_kind[ordinal];
        FieldBoundaryNullspaceBehavior behavior =
            FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
        if (kind == "periodic" || kind == "neumann" ||
            (kind == "mixed" && plan.boundary_alpha[ordinal] == 0.0))
          behavior = FieldBoundaryNullspaceBehavior::PreservesConstantMode;
        facts.push_back({face_identity, behavior});
        identity.text(face_identity).text(kind);
      }
    return make_field_nullspace_operator_facts(std::move(identity).release(), std::move(facts),
                                               plan.has_reaction);
  }

  FieldNullspaceProviderRequest<Dim> make_field_nullspace_request(
      const std::string& slot, const FieldPlan& plan, exact_field_solver_type& solver,
      const std::vector<std::shared_ptr<const field_type>>& coverage,
      std::vector<PreparedVectorDistribution<Dim>>& distributions) const {
    FieldNullspaceProviderRequest<Dim> request;
    request.plan_identity = plan.plan_identity;
    request.operator_facts = nullspace_operator_facts(plan);
    request.topology.identity =
        plan.topology_digest.empty() ? plan.plan_identity + ":amr-topology" : plan.topology_digest;
    request.topology.field_component = 0;
    request.topology.first_level = 0;
    request.topology.coverage = coverage;
    request.topology.layouts.reserve(static_cast<std::size_t>(solver.level_count()));
    request.topology.cell_measure.reserve(static_cast<std::size_t>(solver.level_count()));
    request.topology.coverage_contracts.reserve(static_cast<std::size_t>(solver.level_count()));
    distributions.clear();
    distributions.reserve(static_cast<std::size_t>(solver.level_count()));
    ExactContractBuilder layout_contract;
    layout_contract.text("pops.amr.field-nullspace-layout")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(slot)
        .bytes(engine->spatial_contract())
        .scalar(engine->topology_epoch())
        .scalar(engine->materialization_generation());
    ExactContractBuilder connected;
    connected.text("pops.amr.connected-cartesian-hierarchy")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(solver.level_count()));
    for (int level = 0; level < solver.level_count(); ++level) {
      const field_type& layout = solver.rhs_level(level);
      request.topology.layouts.push_back(&layout);
      PreparedVectorDistribution<Dim> distribution =
          layout.distribution().replicated() ? PreparedVectorDistribution<Dim>::replicated()
                                             : PreparedVectorDistribution<Dim>::distributed();
      layout_contract.bytes(distribution.layout_contract(layout));
      distributions.push_back(distribution);
      Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
          engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), cfg.lower,
          cfg.upper);
      Real measure = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        measure *= geometry.spacing(axis);
      request.topology.cell_measure.push_back(measure);
      ExactContractBuilder coverage_contract;
      coverage_contract.text("pops.amr.active-coverage")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(level)
          .scalar(engine->topology_epoch())
          .scalar(engine->materialization_generation())
          .bytes(distribution.layout_contract(*coverage[static_cast<std::size_t>(level)]));
      request.topology.coverage_contracts.push_back(std::move(coverage_contract).release());
      const Box<Dim>& domain = engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
      for (int axis = 0; axis < Dim; ++axis)
        connected.scalar(domain.lo[axis]).scalar(domain.hi[axis]);
    }
    request.topology.exact_layout_contract = std::move(layout_contract).release();
    request.topology.connected_component_contract = std::move(connected).release();
    request.topology.level_distributions = distributions;
    return request;
  }

  void materialize_field(const std::string& slot) {
    if (!embedded_boundary_configuration_contract.empty() &&
        embedded_boundary_mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive)
      throw std::runtime_error(
          "AMR Cartesian field solvers have no authenticated embedded-boundary operator");
    ensure_engine();
    if (!multiblock_hierarchy)
      throw std::logic_error(
          "AMR exact field materialization requires its prepared multi-block hierarchy");
    const ExecutionLane& lane = multiblock_hierarchy->lane();
    auto found = field_plans.find(slot);
    if (found == field_plans.end())
      throw std::out_of_range("AmrSystem has no exact field provider slot '" + slot + "'");
    FieldPlan& plan = found->second;
    if (plan.materialized_for(*engine))
      return;

    std::unique_ptr<exact_field_solver_type> prepared_solver;
    std::vector<std::unique_ptr<field_type>> accepted_potential;
    std::vector<std::unique_ptr<field_type>> candidate_outputs;
    std::vector<std::unique_ptr<field_type>> contribution_scratch;
    std::vector<std::shared_ptr<const field_type>> coverage;
    runtime::amr::ExactAmrFieldSolverBuildRequest<Dim> request;
    std::shared_ptr<const exact_field_provider_type> provider;
    std::optional<FieldNullspaceProviderRequest<Dim>> nullspace_request;
    std::vector<PreparedVectorDistribution<Dim>> distributions;
    std::string expected_contract;
    std::string nullspace_contract;
    std::string provider_contract;
    std::string support_contract;

    const auto finish_local_phase = [&lane](std::exception_ptr local_error,
                                            std::string_view phase) {
      if (all_reduce_max(local_error ? 1L : 0L, lane) == 0)
        return;
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR exact field " + std::string(phase) + " failed collectively");
    };

    std::exception_ptr local_error;
    try {
      if (plan.output && plan.output_keys.size() != plan.output->component_count())
        throw std::logic_error(
            "AMR exact field output keys do not cover its compact publication carrier");
      if ((!plan.boundary_state_blocks.empty() || !plan.boundary_field_blocks.empty()) &&
          !plan.boundary_kernel)
        throw std::logic_error(
            "AMR field boundary dependencies require a compiled dynamic boundary kernel");
      if (!field_solver_providers || !field_nullspace_providers)
        throw std::logic_error("AMR exact field provider registries are absent");
      if (prepared_hierarchy->block_levels.size() != blocks.size() ||
          prepared_hierarchy->block_levels.front().size() != engine->hierarchy().num_levels() ||
          prepared_hierarchy->provider_storage.size() != engine->hierarchy().num_levels())
        throw std::logic_error("AMR field materialization sees an incomplete prepared hierarchy");

      request.mode = plan.hierarchy_policy.policy_id == "pops.field-hierarchy.level-local"
                         ? runtime::amr::ExactFieldHierarchyMode::level_local
                         : runtime::amr::ExactFieldHierarchyMode::composite;
      if (plan.hierarchy_policy.policy_id != "pops.field-hierarchy.level-local" &&
          plan.hierarchy_policy.policy_id != "pops.field-hierarchy.composite")
        throw std::invalid_argument("AMR exact field hierarchy policy is unsupported: " +
                                    plan.hierarchy_policy.policy_id);
      plan.hierarchy_policy.validate();
      request.provider_options = plan.solver_options;
      request.reaction = static_cast<Real>(plan.has_reaction ? plan.reaction : 0.0);
      request.use_contract = exact_field_plan_contract(slot, plan);
      request.spatial_contract.assign(engine->spatial_contract());
      request.hierarchy.levels.reserve(engine->hierarchy().num_levels());
      request.hierarchy.ratios.reserve(engine->hierarchy().num_levels() - 1);
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        const auto& layout = engine->hierarchy().layout(level);
        const field_type& state = engine->hierarchy().state(level);
        const Geometry<Dim> geometry =
            Geometry<Dim>::from_bounds(layout.domain(), cfg.lower, cfg.upper);
        request.hierarchy.levels.push_back(EllipticBuildRequest<Dim>{
            geometry,
            layout.patches(),
            layout.distribution(),
            state.local_rank(),
            field_boundary(plan, geometry),
            Extent<Dim>{},
            unit_ghosts(),
            {layout.patches().size(), checked_pair_count(layout.patches().size())}});
        if (level != 0)
          request.hierarchy.ratios.push_back(layout.ratio_from_parent());
      }

      provider = field_solver_providers->find(plan.solver_route);
      if (!provider)
        throw std::invalid_argument("unknown exact AMR field solver provider '" +
                                    plan.solver_route + "'");
      const PreparedProviderSupport support = provider->supports(request, lane);
      support_contract = exact_prepared_provider_support(support);
      if (!support.well_formed())
        throw std::runtime_error("exact AMR field solver returned malformed support metadata");
      if (!support.accepted())
        throw std::runtime_error("exact AMR field solver rejected the hierarchy: " +
                                 std::string(support.reason));
      ExactContractBuilder provider_declaration;
      provider_declaration.text("pops.amr.exact-field-provider-declaration")
          .scalar(std::uint32_t{1})
          .text(provider->identity())
          .scalar(provider->interface_version())
          .text(provider->collective_contract());
      provider_contract = std::move(provider_declaration).release();
      expected_contract = provider->expected_prepared_contract(request, lane);
      if (expected_contract.empty())
        throw std::runtime_error(
            "exact AMR field solver accepted the request without a prepared contract");
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "provider declaration");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-exact-field-use", request.use_contract},
             {"amr-exact-field-spatial", request.spatial_contract},
             {"amr-exact-field-provider", provider_contract},
             {"amr-exact-field-support", support_contract},
             {"amr-exact-field-expected", expected_contract}},
            lane))
      throw std::runtime_error("AMR exact field provider declaration differs between MPI ranks");

    local_error = {};
    try {
      prepared_solver = provider->build(request, lane);
      if (!prepared_solver || prepared_solver->provider_identity() != provider->identity() ||
          prepared_solver->exact_prepared_contract() != expected_contract ||
          prepared_solver->level_count() != static_cast<int>(engine->hierarchy().num_levels()))
        throw std::runtime_error("exact AMR field solver published an invalid prepared contract");

      coverage = active_coverage();
      nullspace_request.emplace(
          make_field_nullspace_request(slot, plan, *prepared_solver, coverage, distributions));
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "solver preparation");

    const FieldNullspaceProviderSelection selection{
        plan.nullspace_provider_identity.empty() ? default_nullspace_provider_identity
                                                 : plan.nullspace_provider_identity,
        plan.nullspace_provider_identity.empty() ? default_nullspace_options
                                                 : plan.nullspace_options};
    PreparedFieldNullspace<Dim> prepared_nullspace = prepare_field_nullspace_collectively<Dim>(
        *field_nullspace_providers, selection, std::move(*nullspace_request), lane);
    nullspace_request.reset();
    nullspace_contract = prepared_nullspace.exact_prepared_contract;

    local_error = {};
    try {
      prepared_solver->install_nullspace(std::move(prepared_nullspace), std::move(distributions));
      if (plan.newton)
        prepared_solver->install_newton(*plan.newton);
      if (plan.boundary_kernel)
        prepared_solver->install_boundary_kernel(*plan.boundary_kernel);

      accepted_potential.reserve(engine->hierarchy().num_levels());
      candidate_outputs.reserve(engine->hierarchy().num_levels());
      contribution_scratch.reserve(engine->hierarchy().num_levels());
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        field_type& candidate = prepared_solver->candidate_level(static_cast<int>(level));
        auto accepted = std::make_unique<field_type>(candidate.layout(), candidate.distribution(),
                                                     candidate.local_rank(), candidate.ncomp(),
                                                     candidate.ghosts());
        accepted->set_val(Real(0));
        copy_full_field_in_place(*accepted, candidate);
        std::unique_ptr<field_type> output;
        if (plan.output) {
          output = std::make_unique<field_type>(
              candidate.layout(), candidate.distribution(), candidate.local_rank(),
              static_cast<int>(plan.output->component_count()), candidate.ghosts());
          output->set_val(Real(0));
        }
        field_type& rhs = prepared_solver->rhs_level(static_cast<int>(level));
        auto scratch = std::make_unique<field_type>(rhs.layout(), rhs.distribution(),
                                                    rhs.local_rank(), 1, rhs.ghosts());
        scratch->set_val(Real(0));
        accepted_potential.push_back(std::move(accepted));
        candidate_outputs.push_back(std::move(output));
        contribution_scratch.push_back(std::move(scratch));
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "workspace installation");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-exact-field-prepared", expected_contract},
             {"amr-exact-field-nullspace", nullspace_contract}},
            lane))
      throw std::runtime_error("AMR exact field materialization differs between MPI ranks");

    plan.prepared_solver = std::move(prepared_solver);
    plan.accepted_potential = std::move(accepted_potential);
    plan.candidate_outputs = std::move(candidate_outputs);
    plan.contribution_scratch = std::move(contribution_scratch);
    plan.active_coverage = std::move(coverage);
    plan.prepared_contract = std::move(expected_contract);
    plan.prepared_nullspace_contract = std::move(nullspace_contract);
    plan.topology_epoch = engine->topology_epoch();
    plan.materialization_generation = engine->materialization_generation();
    plan.candidate_ready = false;
  }

  static FieldDistribution field_distribution(const field_type& field) noexcept {
    return field.distribution().replicated() ? FieldDistribution::Replicated
                                             : FieldDistribution::Distributed;
  }

  PreparedBoundaryContext prepare_external_boundary_dependencies(
      const PreparedBoundaryComponentSpec& spec, const field_type& owning, std::size_t level,
      const FieldLogicalTimePoint& point, std::string_view clock_identity,
      const ExecutionLane* preparing_lane = nullptr, multiblock_type* preparing_states = nullptr,
      ExternalBoundaryDependencyOperation operation =
          ExternalBoundaryDependencyOperation::field_closure,
      const engine_type* preparing_engine = nullptr,
      PreparedHierarchy* preparing_hierarchy = nullptr) const {
    const ExecutionLane* lane_authority = preparing_lane;
    if (lane_authority == nullptr && prepared_hierarchy && prepared_hierarchy->lane)
      lane_authority = &*prepared_hierarchy->lane;
    const engine_type* engine_authority =
        preparing_engine != nullptr ? preparing_engine : engine.get();
    PreparedHierarchy* hierarchy_authority =
        preparing_hierarchy != nullptr ? preparing_hierarchy : prepared_hierarchy.get();
    if (lane_authority == nullptr || clock_identity.empty() || engine_authority == nullptr ||
        level >= engine_authority->hierarchy().num_levels())
      throw std::logic_error(
          "AMR external boundary dependencies require a prepared level/lane/clock");
    PreparedBoundaryContext context;
    context.level = level;
    context.owning = &owning;
    context.point = point;
    context.point.level = static_cast<int>(level);
    context.clock_identity = std::string(clock_identity);
    context.invocation_point.clock.reserve(kPreparedAmrClockIdentityCapacity);
    context.lane = lane_authority;
    context.topology_epoch = engine_authority->topology_epoch();
    context.materialization_generation = engine_authority->materialization_generation();
    context.detached.reserve(spec.states.size() + spec.fields.size());
    auto exact_route_identity = [&](std::string_view dependency_identity) {
      ExactContractBuilder identity;
      identity.text("pops.amr.external-boundary-dependency-route")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(static_cast<std::uint32_t>(operation))
          .text(spec.target_identity)
          .text(spec.component_id)
          .text(spec.manifest_identity)
          .scalar(spec.interface_version)
          .text(spec.producer_identity)
          .text(spec.state_identity)
          .text(spec.ghost_identity)
          .text(spec.layout_identity)
          .scalar(static_cast<std::int32_t>(spec.region.kind))
          .scalar(std::int32_t{spec.region.dimension})
          .scalar(std::int32_t{spec.region.codimension})
          .sequence(spec.region.axes)
          .sequence(spec.region.sides)
          .text(spec.region.identity)
          .text(dependency_identity);
      return std::move(identity).release();
    };
    auto authenticate = [&](const field_type& dependency, std::string_view identity) {
      const ExecutionLane& lane = *lane_authority;
      if (dependency.layout() != owning.layout() ||
          dependency.local_rank() != owning.local_rank() ||
          lane.size() != static_cast<int>(dependency.rank_space().size()) ||
          lane.rank() !=
              static_cast<int>(dependency.rank_space().linear_rank(dependency.local_rank())))
        throw std::invalid_argument("AMR external boundary dependency " + std::string(identity) +
                                    " differs from its exact level layout/lane");
      for (std::size_t local = 0; local < owning.local_size(); ++local) {
        const std::size_t global = owning.global_index(local);
        Box<Dim> required = owning.fab(local).box();
        if (operation == ExternalBoundaryDependencyOperation::ghost_region)
          for (std::size_t ordinal = 0; ordinal < spec.region.axes.size(); ++ordinal) {
            const int axis = spec.region.axes[ordinal];
            if (spec.region.sides[ordinal] < 0)
              required.lo[axis] -= owning.ghosts()[axis];
            else
              required.hi[axis] += owning.ghosts()[axis];
          }
        if (!dependency.contains_local(global) ||
            !dependency.fab(dependency.local_index_of(global)).grown_box().contains(required))
          throw std::invalid_argument("AMR external boundary dependency " + std::string(identity) +
                                      " lacks exact-level patch colocation/halo");
      }
    };
    auto prepare_ghost_route = [&](std::string route_identity,
                                   std::function<const field_type*(std::size_t)> source_at,
                                   std::optional<std::size_t> source_block,
                                   bool preserve_source_physical_ghosts) {
      auto route = std::make_shared<typename PreparedBoundaryContext::GhostRoute>();
      route->lane = lane_authority;
      route->target_level = level;
      route->preserve_source_physical_ghosts = preserve_source_physical_ghosts;
      route->source_at = std::move(source_at);
      route->route_identity = std::move(route_identity);
      route->source_block = source_block;
      route->images.reserve(level + 1);
      route->level_points.resize(level + 1);
      route->fine_interpolation_kinds.reserve(level);
      for (std::size_t fine_level = 1; fine_level <= level; ++fine_level)
        route->fine_interpolation_kinds.push_back(
            coarse_fine_transfer_kind(static_cast<int>(fine_level - 1)));
      for (std::size_t source_level = 0; source_level <= level; ++source_level) {
        const field_type* source = route->source_at(source_level);
        if (preserve_source_physical_ghosts) {
          if (preparing_states == nullptr || preparing_states->block_count() == 0 ||
              source_level >= preparing_states->level_count())
            throw std::logic_error(
                "AMR GhostBoundary field route lacks its candidate hierarchy prototype");
          const field_type& prototype = preparing_states->state(0, source_level);
          route->images.push_back(
              std::make_unique<field_type>(prototype.layout(), prototype.distribution(),
                                           prototype.local_rank(), 1, unit_ghosts()));
        } else {
          if (source == nullptr || source->ncomp() < 1 ||
              lane_authority->size() != static_cast<int>(source->rank_space().size()) ||
              lane_authority->rank() !=
                  static_cast<int>(source->rank_space().linear_rank(source->local_rank())))
            throw std::invalid_argument(
                "AMR GhostBoundary dependency differs from its exact ancestor lane/layout");
          route->images.push_back(std::make_unique<field_type>(
              source->layout(), source->distribution(), source->local_rank(), source->ncomp(),
              source->ghosts()));
        }
        route->images.back()->set_val(Real(0));
        route->level_points[source_level].clock.reserve(kPreparedAmrClockIdentityCapacity);
      }
      if (source_block) {
        if (hierarchy_authority == nullptr)
          throw std::logic_error("AMR GhostBoundary state route has no prepared hierarchy owner");
        route->prepare_physical = [hierarchy_authority, block = *source_block](
                                      const runtime::multiblock::BoundaryEvaluationPoint& routed,
                                      std::size_t source_level, field_type& image) {
          if (block >= hierarchy_authority->block_levels.size() ||
              source_level >= hierarchy_authority->block_levels[block].size())
            throw std::logic_error(
                "AMR GhostBoundary source physical authority is not materialized");
          hierarchy_authority->block_levels[block][source_level].prepare_physical(routed, image);
        };
      }
      return route;
    };

    for (const std::string& identity : spec.states) {
      const field_type* dependency = nullptr;
      std::optional<std::size_t> source_block;
      if (identity == spec.state_identity)
        dependency = &owning;
      else
        for (std::size_t block = 0; block < blocks.size(); ++block)
          if (boundary_registry.state_route(blocks[block].name) == identity) {
            dependency = preparing_states == nullptr ? &block_state(block, level)
                                                     : &preparing_states->state(block, level);
            source_block = block;
            break;
          }
      if (dependency == nullptr)
        throw std::runtime_error(
            "AMR external boundary state dependency has no sealed block route");
      authenticate(*dependency, identity);
      if (identity == spec.state_identity) {
        context.states.push_back(nullptr);
      } else if (operation == ExternalBoundaryDependencyOperation::ghost_region) {
        if (!source_block)
          throw std::logic_error("AMR GhostBoundary state route lost its exact source block");
        const std::size_t block = *source_block;
        auto source_at = [this, preparing_states, block](std::size_t source_level) {
          return preparing_states == nullptr ? &block_state(block, source_level)
                                             : &preparing_states->state(block, source_level);
        };
        auto route = prepare_ghost_route(exact_route_identity(identity), std::move(source_at),
                                         source_block, false);
        field_type* target_image = &route->target_image();
        context.detached.push_back({nullptr, nullptr, nullptr, std::move(route)});
        context.states.push_back(target_image);
      } else {
        auto image = std::make_unique<field_type>(dependency->layout(), dependency->distribution(),
                                                  dependency->local_rank(), dependency->ncomp(),
                                                  dependency->ghosts());
        image->set_val(Real(0));
        context.detached.push_back({dependency, nullptr, std::move(image), nullptr});
        context.states.push_back(context.detached.back().image.get());
      }
      context.state_distributions.push_back(field_distribution(*dependency));
      context.state_identities.push_back(identity);
    }
    for (const std::string& identity : spec.fields) {
      const std::string& slot = boundary_registry.field_storage_route(identity);
      const auto plan = field_plans.find(slot);
      if (plan == field_plans.end())
        throw std::runtime_error("AMR external boundary field dependency has no sealed route");
      std::unique_ptr<field_type> dependency_prototype;
      const field_type* dependency = nullptr;
      if (plan->second.materialized_for(*engine_authority) &&
          level < plan->second.accepted_potential.size() &&
          plan->second.accepted_potential[level]) {
        dependency = plan->second.accepted_potential[level].get();
      } else {
        // Graph construction precedes exact-field solver materialization. Prepare the canonical
        // scalar solver layout now, then require the eventual accepted/candidate field to match it
        // exactly before any callback may observe the dependency.
        dependency_prototype = std::make_unique<field_type>(owning.layout(), owning.distribution(),
                                                            owning.local_rank(), 1, unit_ghosts());
        dependency_prototype->set_val(Real(0));
        dependency = dependency_prototype.get();
      }
      authenticate(*dependency, identity);
      if (operation == ExternalBoundaryDependencyOperation::ghost_region) {
        const FieldPlan* field_plan = &plan->second;
        const std::uint64_t expected_topology_epoch = engine_authority->topology_epoch();
        const std::uint64_t expected_materialization_generation =
            engine_authority->materialization_generation();
        auto source_at = [field_plan, expected_topology_epoch, expected_materialization_generation](
                             std::size_t source_level) -> const field_type* {
          if (field_plan->topology_epoch != expected_topology_epoch ||
              field_plan->materialization_generation != expected_materialization_generation)
            return nullptr;
          if (source_level >= field_plan->accepted_potential.size() ||
              !field_plan->accepted_potential[source_level])
            return nullptr;
          if (!field_plan->candidate_ready)
            return field_plan->accepted_potential[source_level].get();
          if (!field_plan->prepared_solver)
            throw std::logic_error("AMR GhostBoundary field candidate has no prepared solver");
          return &field_plan->prepared_solver->candidate_level(static_cast<int>(source_level));
        };
        auto route = prepare_ghost_route(exact_route_identity(identity), std::move(source_at),
                                         std::nullopt, true);
        field_type* target_image = &route->target_image();
        context.detached.push_back({nullptr, field_plan, nullptr, std::move(route)});
        context.fields.push_back(target_image);
      } else {
        auto image = std::make_unique<field_type>(dependency->layout(), dependency->distribution(),
                                                  dependency->local_rank(), dependency->ncomp(),
                                                  dependency->ghosts());
        image->set_val(Real(0));
        context.detached.push_back({nullptr, &plan->second, std::move(image), nullptr});
        context.fields.push_back(context.detached.back().image.get());
      }
      context.field_plans.push_back(&plan->second);
      context.field_distributions.push_back(field_distribution(*dependency));
      context.field_identities.push_back(identity);
    }
    return context;
  }

  static void copy_field_outputs_to_provider_candidate(
      const field_type& outputs, const std::vector<runtime::system::AuxiliaryComponentKey>& keys,
      const auxiliary_registry_type& registry, auxiliary_groups_type& candidate) {
    if (keys.size() != static_cast<std::size_t>(outputs.ncomp()))
      throw std::invalid_argument(
          "AMR field-output key count differs from its compact candidate width");
    for (std::size_t slot = 0; slot < keys.size(); ++slot) {
      const auto address = registry.address_of(keys[slot]);
      field_type* destination = candidate.find(address.group);
      if (destination == nullptr ||
          address.component >= static_cast<std::size_t>(destination->ncomp()))
        throw std::logic_error(
            "AMR field-output candidate lacks its resolved provider storage address");
      if (destination->layout() != outputs.layout() ||
          destination->distribution() != outputs.distribution() ||
          destination->local_rank() != outputs.local_rank())
        throw std::invalid_argument(
            "AMR field-output provider group differs from the solved field layout");
      copy_scalar_component(outputs, static_cast<int>(slot), *destination,
                            static_cast<int>(address.component));
    }
    Kokkos::fence();
  }

  runtime::system::AuxiliaryEvaluationPoint field_output_evaluation_point(
      std::size_t level, const runtime::multiblock::BoundaryEvaluationPoint* evaluation) const {
    if (macro_step < 0 || level > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::logic_error("AMR field-output publication has an invalid accepted coordinate");
    runtime::system::AuxiliaryEvaluationPoint point;
    point.clock = evaluation == nullptr ? "pops.amr.accepted" : evaluation->clock;
    point.accepted_step = static_cast<std::uint64_t>(macro_step);
    point.layout_generation = engine->materialization_generation();
    point.level = static_cast<int>(level);
    point.substep = evaluation == nullptr ? 0 : evaluation->substep;
    point.stage = evaluation == nullptr ? 0 : evaluation->stage;
    point.event = runtime::system::AuxiliaryEvaluationEvent::before_field_solve;
    point.validate();
    ExactContractBuilder exact;
    point.serialize_exact(exact);
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-field-output-evaluation-point", std::move(exact).release()}},
            prepared_hierarchy->lane->communicator()))
      throw std::runtime_error("AMR field-output evaluation point differs across MPI ranks");
    return point;
  }

  void prepare_field_boundary_contexts(
      const std::string& slot, FieldPlan& plan, int active_level,
      const std::vector<const field_type*>& stage_overrides,
      const runtime::multiblock::BoundaryEvaluationPoint* evaluation_point) {
    if (!plan.boundary_kernel)
      return;
    if (evaluation_point == nullptr && !plan.boundary_point)
      throw std::logic_error("AMR dynamic field boundary has no exact logical evaluation point");
    if (evaluation_point != nullptr && evaluation_point->tick < 0)
      throw std::overflow_error("AMR dynamic field boundary tick is negative");

    const std::size_t level_count = engine->hierarchy().num_levels();
    const bool prepare_detached = !plan.boundary_context_storage;
    if (prepare_detached != !plan.boundary_contexts)
      throw std::logic_error("AMR prepared field boundary context cache is incomplete");

    auto context_storage = plan.boundary_context_storage;
    auto contexts = plan.boundary_contexts;
    if (prepare_detached) {
      context_storage = std::make_shared<std::vector<PreparedBoundaryContext>>();
      context_storage->reserve(level_count);
      for (std::size_t level = 0; level < level_count; ++level) {
        PreparedBoundaryContext context;
        context.parameters = &plan.boundary_parameters;
        context.point.level = static_cast<int>(level);
        context.point.iteration = 0;

        context.states.reserve(plan.boundary_state_blocks.size());
        context.state_distributions.reserve(plan.boundary_state_blocks.size());
        context.state_identities.reserve(plan.boundary_state_blocks.size());
        for (std::size_t dependency = 0; dependency < plan.boundary_state_blocks.size();
             ++dependency) {
          const auto block =
              std::find_if(blocks.begin(), blocks.end(), [&](const BlockSpec& candidate) {
                return candidate.name == plan.boundary_state_blocks[dependency];
              });
          if (block == blocks.end())
            throw std::runtime_error(
                "AMR dynamic boundary names an unknown state dependency block");
          const std::size_t block_index =
              static_cast<std::size_t>(std::distance(blocks.begin(), block));
          const field_type* state = &block_state(block_index, level);
          const int component = plan.boundary_state_components[dependency];
          if (!same_field_contract(*state, block_state(block_index, level)) || component < 0 ||
              component >= state->ncomp())
            throw std::invalid_argument(
                "AMR dynamic boundary state dependency is not materialized exactly");
          context.states.push_back(state);
          context.state_distributions.push_back(field_distribution(*state));
          context.state_identities.push_back(block->name + "/" +
                                             prepared_hierarchy->state_field_identities[level]);
        }

        context.fields.reserve(plan.boundary_field_blocks.size());
        context.field_distributions.reserve(plan.boundary_field_blocks.size());
        context.field_identities.reserve(plan.boundary_field_blocks.size());
        for (std::size_t dependency = 0; dependency < plan.boundary_field_blocks.size();
             ++dependency) {
          if (plan.boundary_field_components[dependency] != 0)
            throw std::invalid_argument(
                "AMR scalar field boundary dependency must select component zero");
          const auto dependency_plan =
              std::find_if(field_plans.begin(), field_plans.end(), [&](const auto& candidate) {
                return candidate.second.output_block == plan.boundary_field_blocks[dependency] &&
                       candidate.second.output_key == plan.boundary_field_keys[dependency];
              });
          if (dependency_plan == field_plans.end())
            throw std::runtime_error(
                "AMR dynamic boundary names an unknown owner-qualified field dependency");
          if (dependency_plan->first == slot)
            throw std::logic_error(
                "AMR dynamic boundary cannot name its iterate as an external field dependency");
          const FieldPlan& dependency_field = dependency_plan->second;
          if (!dependency_field.materialized_for(*engine) ||
              dependency_field.accepted_potential.size() != level_count)
            throw std::logic_error(
                "AMR dynamic boundary field dependency must be solved and materialized first");
          const field_type& values = *dependency_field.accepted_potential[level];
          context.fields.push_back(&values);
          context.field_distributions.push_back(field_distribution(values));
          context.field_identities.push_back(dependency_plan->first + "/" +
                                             dependency_field.provider_identity);
        }
        context_storage->push_back(std::move(context));
      }
      contexts = std::make_shared<PreparedFieldBoundaryContextSet<Dim>>(
          std::static_pointer_cast<void>(context_storage), level_count);
    }

    if (context_storage->size() != level_count || !contexts || contexts->size() != level_count)
      throw std::logic_error("AMR prepared field boundary context set is stale");

    for (std::size_t level = 0; level < level_count; ++level) {
      PreparedBoundaryContext& context = context_storage->at(level);
      context.point = plan.boundary_point.value_or(FieldLogicalTimePoint{});
      if (evaluation_point != nullptr) {
        context.point.time = static_cast<Real>(evaluation_point->physical_time);
        context.point.dt = static_cast<Real>(evaluation_point->dt);
        context.point.stage_slot = evaluation_point->stage;
        context.point.step = evaluation_point->tick;
        context.point.substep = evaluation_point->substep;
        context.point.stage_fraction_numerator = evaluation_point->stage_fraction.numerator;
        context.point.stage_fraction_denominator = evaluation_point->stage_fraction.denominator;
      }
      context.point.level = static_cast<int>(level);
      context.point.iteration = 0;
      context.refresh_dynamic_fields();
      for (std::size_t dependency = 0; dependency < plan.boundary_state_blocks.size();
           ++dependency) {
        const auto block =
            std::find_if(blocks.begin(), blocks.end(), [&](const BlockSpec& candidate) {
              return candidate.name == plan.boundary_state_blocks[dependency];
            });
        if (block == blocks.end())
          throw std::runtime_error("AMR dynamic boundary lost a prepared state route");
        const std::size_t block_index =
            static_cast<std::size_t>(std::distance(blocks.begin(), block));
        const field_type* state = &block_state(block_index, level);
        if (static_cast<int>(level) == active_level && !stage_overrides.empty() &&
            stage_overrides[block_index] != nullptr)
          state = stage_overrides[block_index];
        if (!same_field_contract(*state, block_state(block_index, level)))
          throw std::invalid_argument(
              "AMR dynamic boundary state invocation differs from its prepared layout");
        context.states[dependency] = state;
      }
      contexts->bind(level, context.view());
    }
    if (prepare_detached) {
      plan.boundary_context_storage = std::move(context_storage);
      plan.boundary_contexts = std::move(contexts);
    }
    plan.prepared_solver->set_boundary_contexts(plan.boundary_contexts);
  }

  SolveReport solve_field_candidate(
      const std::string& slot, int active_level,
      const std::vector<const field_type*>& stage_overrides,
      const runtime::multiblock::BoundaryEvaluationPoint* evaluation_point = nullptr) {
    materialize_field(slot);
    if (!multiblock_hierarchy)
      throw std::logic_error("AMR exact field solve requires its prepared multi-block hierarchy");
    const ExecutionLane& lane = multiblock_hierarchy->lane();
    FieldPlan& plan = field_plans.at(slot);
    if (!active_field_slot.empty())
      throw std::logic_error(
          "AMR exact field solves are sequential until the prior SolveOutcome is consumed");
    if (active_level < 0 ||
        static_cast<std::size_t>(active_level) >= engine->hierarchy().num_levels())
      throw std::out_of_range("AMR exact field solve active level is outside the hierarchy");
    if (!stage_overrides.empty() && stage_overrides.size() != blocks.size())
      throw std::invalid_argument(
          "AMR exact field stage vector must cover the runtime block registry");

    bool has_rhs = false;
    prepare_field_boundary_contexts(slot, plan, active_level, stage_overrides, evaluation_point);
    active_field_slot = slot;
    plan.candidate_ready = false;
    try {
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        field_type& rhs = plan.prepared_solver->rhs_level(static_cast<int>(level));
        field_type& candidate = plan.prepared_solver->candidate_level(static_cast<int>(level));
        rhs.set_val(Real(0));
        copy_full_field_in_place(*plan.accepted_potential[level], candidate);
        const field_type* state = &engine->hierarchy().state(level);
        if (static_cast<int>(level) == active_level && !stage_overrides.empty() &&
            stage_overrides.front() != nullptr)
          state = stage_overrides.front();
        if (!same_field_contract(*state, engine->hierarchy().state(level)))
          throw std::invalid_argument(
              "AMR exact field stage override differs from its live level contract");

        if (plan.use_prepared_level_rhs) {
          prepared_hierarchy->block_levels.front()[level].add_poisson_rhs(*state, rhs);
          has_rhs = true;
        }
        if (!plan.rhs_by_block.empty()) {
          if (plan.rhs_by_block.size() != blocks.size())
            throw std::logic_error("AMR exact field RHS registry changed after preparation");
          for (const PreparedFieldRhs& provider : plan.rhs_by_block.front()) {
            field_type& scratch = *plan.contribution_scratch[level];
            scratch.set_val(Real(0));
            provider.evaluate(*state, scratch);
            saxpy(rhs, provider.coefficient, scratch);
            has_rhs = true;
          }
        }
      }
      if (!has_rhs)
        throw std::runtime_error("AMR exact field has no prepared RHS provider");
      Kokkos::fence();
      SolveReport report = plan.prepared_solver->solve(lane);
      if (!report.solved_value_available()) {
        active_field_slot.clear();
        return report;
      }
      if (plan.output) {
        std::vector<std::string> provider_identities;
        std::exception_ptr setup_error;
        try {
          plan.candidate_provider_storage.clear();
          plan.candidate_provider_publications.clear();
          plan.stale_auxiliary_providers.clear();
          plan.candidate_provider_storage.reserve(engine->hierarchy().num_levels());
          plan.candidate_provider_publications.reserve(engine->hierarchy().num_levels());
          provider_identities.reserve(plan.output_keys.size());
          for (const auto& key : plan.output_keys) {
            const auto& provider = auxiliary_registry.provider_for_key(key);
            if (provider.kind() != runtime::system::AuxiliaryProviderKind::field_output)
              throw std::logic_error("AMR field output lost its field-output provider authority");
            if (std::find(provider_identities.begin(), provider_identities.end(),
                          provider.identity()) == provider_identities.end())
              provider_identities.push_back(provider.identity());
          }
          if (!prepared_hierarchy->lane ||
              prepared_hierarchy->provider_candidate_storage.size() !=
                  engine->hierarchy().num_levels() ||
              prepared_hierarchy->provider_candidate_ghost_fills.size() !=
                  engine->hierarchy().num_levels() ||
              prepared_hierarchy->provider_candidate_physical_boundaries.size() !=
                  engine->hierarchy().num_levels())
            throw std::logic_error("AMR field-output candidate authority is not materialized");
        } catch (...) {
          setup_error = std::current_exception();
        }
        runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
            setup_error, prepared_hierarchy->lane ? &*prepared_hierarchy->lane : nullptr,
            "AMR field-output setup failed collectively before candidate transport");
        for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
          auxiliary_groups_type* candidate_storage = nullptr;
          std::exception_ptr materialization_error;
          try {
            if (!plan.candidate_outputs[level] || !prepared_hierarchy->provider_storage[level] ||
                !prepared_hierarchy->provider_candidate_storage[level])
              throw std::logic_error("AMR field output carrier is not materialized exactly");
            field_type& output = *plan.candidate_outputs[level];
            const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
                engine->hierarchy().layout(level).domain(), cfg.lower, cfg.upper);
            runtime::field::publish_named_field(
                plan.prepared_solver->candidate_level(static_cast<int>(level)), output, geometry,
                *plan.output);
            candidate_storage = prepared_hierarchy->provider_candidate_storage[level].get();
            copy_auxiliary_groups_in_place(*prepared_hierarchy->provider_storage[level],
                                           *candidate_storage);
            auto& registry = prepared_hierarchy->auxiliary_registries[level];
            copy_field_outputs_to_provider_candidate(output, plan.output_keys, registry,
                                                     *candidate_storage);
            plan.candidate_provider_storage.push_back(candidate_storage);
          } catch (...) {
            materialization_error = std::current_exception();
          }
          runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
              materialization_error, &*prepared_hierarchy->lane,
              "AMR field-output materialization failed collectively before candidate transport");

          auto& registry = prepared_hierarchy->auxiliary_registries[level];
          const auto auxiliary_point = field_output_evaluation_point(level, evaluation_point);
          runtime::multiblock::BoundaryEvaluationPoint boundary_point;
          std::exception_ptr publication_error;
          try {
            boundary_point = auxiliary_boundary_evaluation_point(auxiliary_point);
            plan.candidate_provider_publications.emplace_back(
                registry.begin_external_publication(auxiliary_point, provider_identities));
            auto& publication = *plan.candidate_provider_publications.back();
            for (const std::string& identity : provider_identities)
              publication.stage_external(identity);
          } catch (...) {
            publication_error = std::current_exception();
          }
          runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
              publication_error, &*prepared_hierarchy->lane,
              "AMR field-output publication staging failed collectively before candidate "
              "transport");
          auto& publication = *plan.candidate_provider_publications.back();
          const auto refresh_candidate_ghosts = [&] {
            PreparedProviderGroupsGhostFill<Dim>* fill = nullptr;
            runtime::system::PreparedAuxiliaryPhysicalBoundaries<Dim>* physical = nullptr;
            std::exception_ptr authority_error;
            try {
              auto& prepared_fill = prepared_hierarchy->provider_candidate_ghost_fills[level];
              auto& prepared_physical =
                  prepared_hierarchy->provider_candidate_physical_boundaries[level];
              if (!prepared_fill || !prepared_physical)
                throw std::logic_error("AMR field-output candidate ghost authority is incomplete");
              fill = &prepared_fill;
              physical = &*prepared_physical;
            } catch (...) {
              authority_error = std::current_exception();
            }
            runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
                authority_error, &*prepared_hierarchy->lane,
                "AMR field-output candidate ghost authority failed collectively");
            std::exception_ptr hierarchy_error;
            try {
              (*fill)(*candidate_storage, boundary_point);
            } catch (...) {
              hierarchy_error = std::current_exception();
            }
            runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
                hierarchy_error, &*prepared_hierarchy->lane,
                "AMR field-output hierarchy ghost fill failed collectively");
            physical->execute(*candidate_storage);
          };
          refresh_candidate_ghosts();
          publication.launch_ready_native(
              {prepared_hierarchy->provider_storage[level].get(), candidate_storage},
              [&](const auto&, std::exception_ptr local_error) {
                runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
                    local_error, &*prepared_hierarchy->lane,
                    "AMR field-output auxiliary launch failed collectively before ghost fill");
                refresh_candidate_ghosts();
              });
          runtime::system::require_finite_auxiliary_groups(
              *candidate_storage, &*prepared_hierarchy->lane, "AMR field-output publication");
          publication.validate_complete();
          for (const std::string& identity :
               registry.dependent_provider_identities(provider_identities))
            if (std::find(plan.stale_auxiliary_providers.begin(),
                          plan.stale_auxiliary_providers.end(),
                          identity) == plan.stale_auxiliary_providers.end())
              plan.stale_auxiliary_providers.push_back(identity);
        }
      }
      plan.candidate_ready = true;
      return report;
    } catch (...) {
      for (auto& publication : plan.candidate_provider_publications)
        if (publication)
          publication->reject();
      plan.candidate_provider_publications.clear();
      plan.candidate_provider_storage.clear();
      plan.stale_auxiliary_providers.clear();
      plan.candidate_ready = false;
      active_field_slot.clear();
      throw;
    }
  }

  void validate_field_candidate() const {
    if (active_field_slot.empty())
      throw std::logic_error("AMR exact field publication candidate is absent");
    const FieldPlan& plan = field_plans.at(active_field_slot);
    if (!engine || !prepared_hierarchy || !plan.materialized_for(*engine) || !plan.candidate_ready)
      throw std::logic_error("AMR exact field publication candidate is stale");
    if (!plan.output)
      return;
    if (plan.candidate_provider_storage.size() != prepared_hierarchy->provider_storage.size() ||
        plan.candidate_provider_publications.size() != prepared_hierarchy->provider_storage.size())
      throw std::logic_error("AMR field-output publication candidate is incomplete");
    for (std::size_t level = 0; level < plan.candidate_provider_storage.size(); ++level) {
      runtime::system::AuxiliaryCarrierStorage<Dim>{
          prepared_hierarchy->provider_storage[level].get(), plan.candidate_provider_storage[level]}
          .validate();
      if (!plan.candidate_provider_publications[level])
        throw std::logic_error("AMR field-output publication metadata is absent");
      plan.candidate_provider_publications[level]->validate_complete();
    }
  }

  void accept_field_candidate() noexcept {
    try {
      validate_field_candidate();
      FieldPlan& plan = field_plans.at(active_field_slot);
      for (std::size_t level = 0; level < plan.accepted_potential.size(); ++level) {
        copy_full_field_in_place(plan.prepared_solver->candidate_level(static_cast<int>(level)),
                                 *plan.accepted_potential[level]);
        if (plan.output) {
          copy_auxiliary_groups_in_place(*plan.candidate_provider_storage[level],
                                         *prepared_hierarchy->provider_storage[level]);
          plan.candidate_provider_publications[level]->accept();
        }
      }
      for (const std::string& identity : plan.stale_auxiliary_providers)
        if (std::find(dirty_auxiliary_providers.begin(), dirty_auxiliary_providers.end(),
                      identity) == dirty_auxiliary_providers.end())
          dirty_auxiliary_providers.push_back(identity);
      plan.candidate_provider_publications.clear();
      plan.candidate_provider_storage.clear();
      plan.stale_auxiliary_providers.clear();
      plan.candidate_ready = false;
      active_field_slot.clear();
    } catch (...) {
      std::terminate();
    }
  }

  void reject_field_candidate() noexcept {
    try {
      if (!active_field_slot.empty()) {
        FieldPlan& plan = field_plans.at(active_field_slot);
        for (auto& publication : plan.candidate_provider_publications)
          if (publication)
            publication->reject();
        plan.candidate_provider_publications.clear();
        plan.candidate_provider_storage.clear();
        plan.stale_auxiliary_providers.clear();
        plan.candidate_ready = false;
      }
      active_field_slot.clear();
    } catch (...) {
      std::terminate();
    }
  }

  SolveOutcome make_field_outcome(SolveReport report) {
    FieldPlan* plan = active_field_slot.empty() ? nullptr : &field_plans.at(active_field_slot);
    if (!solve_report_is_publishable(report, plan ? plan->prepared_solver->maximum_iterations()
                                                  : std::numeric_limits<int>::max())) {
      reject_field_candidate();
      throw std::runtime_error("AMR exact field solver published a malformed SolveReport");
    }
    ExactSolveReportConsensusScratch consensus;
    if (!consensus.agrees(report)) {
      reject_field_candidate();
      throw std::runtime_error("AMR exact field SolveReport differs between MPI ranks");
    }
    if (!report.solved_value_available())
      return SolveOutcome::collective_lane(std::move(report), *prepared_hierarchy->lane);
    validate_field_candidate();
    return SolveOutcome::collective_lane(
        std::move(report), *prepared_hierarchy->lane,
        SolveOutcome::PublicationHooks{
            this,
            [](void* context) noexcept { static_cast<Impl*>(context)->accept_field_candidate(); },
            [](void* context) { static_cast<Impl*>(context)->reject_field_candidate(); },
            nullptr,
            {},
            [](void* context) { static_cast<Impl*>(context)->validate_field_candidate(); }});
  }

  std::unique_ptr<PreparedHierarchy> prepare_hierarchy_graph(
      engine_type& candidate_engine, multiblock_type& candidate_multiblock,
      const PreparedHierarchy* previous) const {
    if (prepared_blocks.empty())
      throw std::logic_error("AmrSystem has no retained generated package");
    if (std::any_of(prepared_blocks.begin(), prepared_blocks.end(),
                    [](const auto& block) { return block.provider_components != 0; }) &&
        !auxiliary_registry.sealed())
      throw std::logic_error(
          "AMR generated provider consumers require a sealed owner-qualified registry");

    const std::size_t level_count = candidate_engine.hierarchy().num_levels();
    std::unique_ptr<PreparedHierarchy> candidate;
    std::string lane_identity;
    std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> boundary;
    std::string boundary_identity;
    std::exception_ptr allocation_error;
    if (!boundary_execution_context)
      throw std::logic_error(
          "AMR hierarchy materialization requires its authenticated RuntimeInstance lane");
    const PopsExecutionContextV1 execution = boundary_execution_context->view();
#ifdef POPS_HAS_MPI
    const CommunicatorView authenticated_parent{
        MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle))};
#else
    const CommunicatorView authenticated_parent{};
#endif
    // Prepare the hierarchy owner and lane identity before borrowing or duplicating the supplied
    // communicator.  A rank-asymmetric allocation failure is therefore converged on exactly the
    // authenticated parent, while no candidate owns a child communicator yet.
    try {
      candidate = std::make_unique<PreparedHierarchy>();
      candidate->topology_epoch = candidate_engine.topology_epoch();
      candidate->materialization_generation = candidate_engine.materialization_generation();
      lane_identity = "pops.generated-amr-levels/" + std::to_string(candidate->topology_epoch) +
                      "/" + std::to_string(candidate->materialization_generation);
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, authenticated_parent) != 0) {
      if (allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error(
          "generated AMR hierarchy owner preparation failed collectively on its parent lane");
    }
    {
#ifdef POPS_HAS_MPI
      const auto parent = ExecutionCommunicator::borrowed(execution.communicator_identity,
                                                          authenticated_parent.native_handle());
#else
      if (execution.communicator_f_handle != 0 || execution.communicator_datatype_f_handle != 0 ||
          std::string_view(execution.communicator_identity) != "serial")
        throw std::invalid_argument(
            "serial AMR boundary execution requires the exact serial authority");
      const auto parent = ExecutionCommunicator::world();
#endif
      candidate->lane.emplace(ExecutionLane::duplicate_collectively(parent, lane_identity));
    }
    std::string boundary_component_contract;
    std::exception_ptr boundary_component_contract_error;
    try {
      ExactContractBuilder contract;
      contract.text("pops.amr.prepared-boundary-provider-order")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(static_cast<std::uint64_t>(prepared_boundary_components.size()));
      for (const auto& [block, components] : prepared_boundary_components) {
        contract.text(block).scalar(static_cast<std::uint64_t>(components.ghosts.size()));
        for (const auto& provider : components.ghosts) {
          if (!provider)
            throw std::logic_error("AMR GhostBoundary provider order contains a null component");
          append_prepared_boundary_component_contract(contract, provider->spec());
        }
        contract.scalar(static_cast<std::uint64_t>(components.fluxes.size()));
        for (const auto& provider : components.fluxes) {
          if (!provider)
            throw std::logic_error("AMR BoundaryFlux provider order contains a null component");
          append_prepared_boundary_component_contract(contract, provider->spec());
        }
        contract.scalar(static_cast<std::uint64_t>(components.fields.size()));
        for (const auto& [key, pair] : components.fields) {
          if (!pair.residual || !pair.jvp)
            throw std::logic_error("AMR FieldBoundary provider order contains an incomplete pair");
          contract.text(key);
          append_prepared_boundary_component_contract(contract, pair.residual->spec());
          append_prepared_boundary_component_contract(contract, pair.jvp->spec());
        }
      }
      boundary_component_contract = std::move(contract).release();
    } catch (...) {
      boundary_component_contract_error = std::current_exception();
    }
    if (all_reduce_max(boundary_component_contract_error ? 1L : 0L, *candidate->lane) != 0) {
      if (candidate->lane->size() == 1 && boundary_component_contract_error)
        std::rethrow_exception(boundary_component_contract_error);
      throw std::runtime_error("AMR prepared boundary provider order failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-prepared-boundary-provider-order", boundary_component_contract}},
            *candidate->lane))
      throw std::runtime_error("AMR prepared boundary provider order differs across MPI ranks");
    try {
      candidate->spatial_contract.assign(candidate_engine.spatial_contract());
      candidate->package_contract = prepared_package_contract();
      candidate->embedded_boundary_configuration_contract =
          embedded_boundary_configuration_contract;
      candidate->provider_storage.reserve(level_count);
      candidate->provider_candidate_storage.reserve(level_count);
      candidate->auxiliary_registries.reserve(level_count);
      candidate->provider_candidate_ghost_fills.resize(level_count);
      candidate->provider_candidate_physical_boundaries.resize(level_count);
      candidate->embedded_boundary.resize(level_count);
      candidate->block_levels.resize(prepared_blocks.size());
      candidate->block_evaluations.resize(prepared_blocks.size());
      candidate->block_evaluation_candidates.resize(prepared_blocks.size());
      candidate->block_evaluation_published.resize(prepared_blocks.size());
      candidate->block_stage_scratch.resize(prepared_blocks.size());
      candidate->block_state_storage.resize(prepared_blocks.size());
      for (std::size_t block = 0; block < prepared_blocks.size(); ++block) {
        candidate->block_levels[block].reserve(level_count);
        candidate->block_evaluations[block].resize(level_count);
        candidate->block_evaluation_candidates[block].resize(level_count);
        candidate->block_evaluation_published[block].assign(level_count, false);
        candidate->block_stage_scratch[block].resize(level_count);
        candidate->block_state_storage[block].reserve(level_count);
      }
      candidate->provider_storage_identity.reserve(level_count);
      candidate->provider_candidate_storage_identity.reserve(level_count);
      candidate->state_field_identities.reserve(level_count);
      candidate->block_state_field_identities.resize(prepared_blocks.size());
      for (auto& identities : candidate->block_state_field_identities)
        identities.reserve(level_count);
      candidate->provider_storage_field_identities.reserve(level_count);

      const std::string& state_route = boundary_registry.state_route(prepared_blocks.front().name);
      const auto* installed_boundary =
          boundary_registry.find_boundary(prepared_blocks.front().name);
      if (installed_boundary != nullptr) {
        boundary = installed_boundary->authority;
        boundary_identity = installed_boundary->identity;
      }
      if (pending_provider_restore && pending_provider_restore->size() != level_count)
        throw std::invalid_argument(
            "AMR rollback provider image differs from the restored hierarchy depth");
      if (pending_provider_registry_restore &&
          pending_provider_registry_restore->size() != level_count)
        throw std::invalid_argument(
            "AMR rollback provider registry differs from the restored hierarchy depth");
      for (std::size_t level = 0; level < level_count; ++level) {
        field_type& state = candidate_multiblock.state(0, level);
        auto provider_storage = std::make_unique<auxiliary_groups_type>();
        if (auxiliary_registry.sealed())
          for (const auto& group : auxiliary_registry.storage_groups()) {
            if (group.component_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
              throw std::overflow_error("AMR auxiliary storage-group width exceeds int");
            Extent<Dim> ghosts{};
            for (int axis = 0; axis < Dim; ++axis)
              ghosts[axis] = group.shape.halo[axis];
            provider_storage->groups.emplace(
                group.identity, field_type(state.layout(), state.distribution(), state.local_rank(),
                                           static_cast<int>(group.component_count), ghosts));
          }
        const auto restore_provider_groups = [&](const auxiliary_groups_type& restored) {
          if (restored.groups.size() != provider_storage->groups.size())
            throw std::invalid_argument(
                "AMR rollback provider image differs from the resolved storage-group set");
          for (auto& [identity, group] : provider_storage->groups) {
            const auto* source = restored.find(identity);
            if (source == nullptr || !same_field_shape(*source, group))
              throw std::invalid_argument(
                  "AMR rollback provider image differs from its exact level layout");
            copy_valid_field(*source, group);
          }
        };
        if (pending_provider_restore)
          restore_provider_groups((*pending_provider_restore)[level]);
        else if (previous != nullptr && level < previous->provider_storage.size() &&
                 previous->provider_storage[level])
          restore_provider_groups(*previous->provider_storage[level]);
        for (std::size_t block = 0; block < prepared_blocks.size(); ++block)
          candidate->block_state_storage[block].push_back(
              field_storage_identity(candidate_multiblock.state(block, level)));
        std::map<std::string, std::vector<const Real*>> group_storage_identity;
        for (const auto& [identity, group] : provider_storage->groups)
          group_storage_identity.emplace(identity, field_storage_identity(group));
        candidate->provider_storage_identity.push_back(std::move(group_storage_identity));
        candidate->state_field_identities.push_back(state_route + "/level/" +
                                                    std::to_string(level));
        for (std::size_t block = 0; block < prepared_blocks.size(); ++block)
          candidate->block_state_field_identities[block].push_back(
              boundary_registry.state_route(prepared_blocks[block].name) + "/level/" +
              std::to_string(level));
        candidate->provider_storage_field_identities.push_back(
            prepared_blocks.front().provider_identity + "/provider-groups/level/" +
            std::to_string(level));
        auto provider_candidate = std::make_unique<auxiliary_groups_type>(*provider_storage);
        std::map<std::string, std::vector<const Real*>> candidate_storage_identity;
        for (const auto& [identity, group] : provider_candidate->groups)
          candidate_storage_identity.emplace(identity, field_storage_identity(group));
        candidate->provider_candidate_storage_identity.push_back(
            std::move(candidate_storage_identity));
        candidate->provider_candidate_storage.push_back(std::move(provider_candidate));
        candidate->provider_storage.push_back(std::move(provider_storage));
        if (pending_provider_registry_restore)
          candidate->auxiliary_registries.push_back((*pending_provider_registry_restore)[level]);
        else if (previous != nullptr && level < previous->auxiliary_registries.size())
          candidate->auxiliary_registries.push_back(previous->auxiliary_registries[level]);
        else if (auxiliary_registry.sealed())
          candidate->auxiliary_registries.push_back(auxiliary_registry);
        else
          candidate->auxiliary_registries.emplace_back();
      }
      candidate->active_coverage = prepare_active_coverage(candidate_engine);
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, candidate->lane->communicator()) != 0) {
      if (allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error(
          "generated AMR hierarchy allocation failed collectively before lane publication");
    }

    const BoundaryTopology<Dim> exact_topology = topology();

    if (!embedded_boundary_configuration_contract.empty()) {
      ExactContractBuilder materializations;
      materializations.text("pops.amr.embedded-boundary-levels")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(embedded_boundary_configuration_contract)
          .scalar(static_cast<std::uint64_t>(level_count));
      for (std::size_t level = 0; level < level_count; ++level) {
        const Box<Dim>& level_domain = candidate_engine.hierarchy().layout(level).domain();
        const Geometry<Dim> geometry =
            Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper);
        candidate->embedded_boundary[level] =
            runtime::system::prepare_embedded_boundary_geometry_collectively(
                embedded_boundary_opcodes, embedded_boundary_literals, geometry, exact_topology,
                candidate_engine.hierarchy().state(level), embedded_boundary_mode,
                embedded_boundary_thresholds, embedded_boundary_generation, *candidate->lane);
        materializations.text(candidate->embedded_boundary[level]->digest());
      }
      candidate->embedded_boundary_materialization_digest = prefixed_sha256(
          "pops.prepared-eb-geometry.v1:sha256:", std::move(materializations).release());
    }

    struct PreparedExternalFieldInvocation {
      std::string pair_key;
      std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual;
      std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp;
      std::shared_ptr<PreparedBoundaryContext> residual_dependencies;
      std::shared_ptr<PreparedBoundaryContext> jvp_dependencies;
      std::optional<PreparedFieldBoundaryResidualComponent::LocalSessionCandidate<Dim>>
          residual_local_session;
      std::optional<PreparedFieldBoundaryJvpComponent::LocalSessionCandidate<Dim>>
          jvp_local_session;
      std::shared_ptr<std::optional<PreparedFieldBoundaryResidualComponent::Session>>
          residual_session;
      std::shared_ptr<std::optional<PreparedFieldBoundaryJvpComponent::Session>> jvp_session;
      int selected_face = -1;
    };
    struct PreparedExternalGhostInvocation {
      std::shared_ptr<PreparedGhostBoundaryComponent> provider;
      std::shared_ptr<PreparedBoundaryContext> dependencies;
      std::optional<PreparedGhostBoundaryComponent::LocalSessionCandidate<Dim>> local_session;
      std::shared_ptr<std::optional<PreparedGhostBoundaryComponent::Session>> session;
    };
    struct PreparedExternalFluxInvocation {
      std::shared_ptr<PreparedBoundaryFluxComponent> provider;
      std::shared_ptr<PreparedBoundaryContext> dependencies;
      std::optional<PreparedBoundaryFluxComponent::LocalSessionCandidate<Dim>> local_session;
      std::shared_ptr<std::optional<PreparedBoundaryFluxComponent::Session>> session;
    };
    struct PreparedExternalLevelInvocations {
      std::vector<PreparedExternalGhostInvocation> ghosts;
      std::vector<PreparedExternalFluxInvocation> fluxes;
      std::vector<PreparedExternalFieldInvocation> fields;
    };
    std::vector<std::vector<PreparedExternalLevelInvocations>> external_preparations;
    std::exception_ptr external_local_error;
    try {
      external_preparations.resize(prepared_blocks.size());
      for (auto& block : external_preparations)
        block.resize(level_count);
      // Purely local prepass. Canonical provider order is block, operation, provider, level. It
      // allocates every detached all-level image and component/session scratch before any external
      // boundary provider is allowed to enter a collective prepare callback.
      for (std::size_t block_index = 0; block_index < prepared_blocks.size(); ++block_index) {
        const auto component_iter =
            prepared_boundary_components.find(prepared_blocks[block_index].name);
        if (component_iter == prepared_boundary_components.end())
          continue;
        const PreparedBoundaryComponents& components = component_iter->second;
        for (const auto& provider : components.ghosts)
          for (std::size_t level = 0; level < level_count; ++level) {
            field_type& state = candidate_multiblock.state(block_index, level);
            const Box<Dim>& domain = candidate_engine.hierarchy().layout(level).domain();
            auto dependencies =
                std::make_shared<PreparedBoundaryContext>(prepare_external_boundary_dependencies(
                    provider->spec(), state, level, FieldLogicalTimePoint{}, "prepared",
                    &*candidate->lane, &candidate_multiblock,
                    ExternalBoundaryDependencyOperation::ghost_region, &candidate_engine,
                    candidate.get()));
            auto local_session = provider->template make_local_session_candidate<Dim>(
                *candidate->lane, state, Geometry<Dim>::from_bounds(domain, cfg.lower, cfg.upper),
                dependencies->view());
            auto session =
                std::make_shared<std::optional<PreparedGhostBoundaryComponent::Session>>();
            external_preparations[block_index][level].ghosts.push_back(
                {provider, std::move(dependencies), std::move(local_session), std::move(session)});
          }
        for (const auto& provider : components.fluxes)
          for (std::size_t level = 0; level < level_count; ++level) {
            field_type& state = candidate_multiblock.state(block_index, level);
            const Box<Dim>& domain = candidate_engine.hierarchy().layout(level).domain();
            auto dependencies =
                std::make_shared<PreparedBoundaryContext>(prepare_external_boundary_dependencies(
                    provider->spec(), state, level, FieldLogicalTimePoint{}, "prepared",
                    &*candidate->lane, &candidate_multiblock,
                    ExternalBoundaryDependencyOperation::flux_transform, &candidate_engine,
                    candidate.get()));
            auto local_session = provider->template make_local_session_candidate<Dim>(
                *candidate->lane, state, Geometry<Dim>::from_bounds(domain, cfg.lower, cfg.upper),
                dependencies->view());
            auto session =
                std::make_shared<std::optional<PreparedBoundaryFluxComponent::Session>>();
            external_preparations[block_index][level].fluxes.push_back(
                {provider, std::move(dependencies), std::move(local_session), std::move(session)});
          }
        for (const auto& [pair_key, pair] : components.fields) {
          if (!pair.residual || !pair.jvp)
            throw std::logic_error(
                "AMR FieldBoundaryClosure requires one complete residual/JVP pair");
          require_prepared_field_boundary_pair(pair.residual->spec(), pair.jvp->spec());
          const int selected_face = 2 * pair.residual->spec().region.axes.front() +
                                    (pair.residual->spec().region.sides.front() < 0 ? 0 : 1);
          for (std::size_t level = 0; level < level_count; ++level) {
            field_type& state = candidate_multiblock.state(block_index, level);
            const Box<Dim>& domain = candidate_engine.hierarchy().layout(level).domain();
            const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, cfg.lower, cfg.upper);
            auto residual_dependencies =
                std::make_shared<PreparedBoundaryContext>(prepare_external_boundary_dependencies(
                    pair.residual->spec(), state, level, FieldLogicalTimePoint{}, "prepared",
                    &*candidate->lane, &candidate_multiblock,
                    ExternalBoundaryDependencyOperation::field_closure, &candidate_engine,
                    candidate.get()));
            auto residual_local_session = pair.residual->template make_local_session_candidate<Dim>(
                *candidate->lane, state, geometry, residual_dependencies->view());
            auto jvp_dependencies =
                std::make_shared<PreparedBoundaryContext>(prepare_external_boundary_dependencies(
                    pair.jvp->spec(), state, level, FieldLogicalTimePoint{}, "prepared",
                    &*candidate->lane, &candidate_multiblock,
                    ExternalBoundaryDependencyOperation::field_closure, &candidate_engine,
                    candidate.get()));
            auto jvp_local_session = pair.jvp->template make_local_session_candidate<Dim>(
                *candidate->lane, state, geometry, jvp_dependencies->view());
            auto residual_session =
                std::make_shared<std::optional<PreparedFieldBoundaryResidualComponent::Session>>();
            auto jvp_session =
                std::make_shared<std::optional<PreparedFieldBoundaryJvpComponent::Session>>();
            external_preparations[block_index][level].fields.push_back(
                {pair_key, pair.residual, pair.jvp, std::move(residual_dependencies),
                 std::move(jvp_dependencies), std::move(residual_local_session),
                 std::move(jvp_local_session), std::move(residual_session), std::move(jvp_session),
                 selected_face});
          }
        }
      }
      std::size_t transport_contract_count = 0;
      for (std::size_t block_index = 0; block_index < external_preparations.size(); ++block_index)
        for (std::size_t level = 0; level < level_count; ++level) {
          auto& prepared = external_preparations[block_index][level];
          for (auto& invocation : prepared.ghosts) {
            invocation.dependencies->prepare_dependency_transport_requests_locally(
                candidate_engine, exact_topology, *candidate->lane);
            transport_contract_count += invocation.dependencies->dependency_transport_count();
          }
          for (auto& invocation : prepared.fluxes) {
            invocation.dependencies->prepare_dependency_transport_requests_locally(
                candidate_engine, exact_topology, *candidate->lane);
            transport_contract_count += invocation.dependencies->dependency_transport_count();
          }
          for (auto& invocation : prepared.fields) {
            invocation.residual_dependencies->prepare_dependency_transport_requests_locally(
                candidate_engine, exact_topology, *candidate->lane);
            invocation.jvp_dependencies->prepare_dependency_transport_requests_locally(
                candidate_engine, exact_topology, *candidate->lane);
            transport_contract_count +=
                invocation.residual_dependencies->dependency_transport_count() +
                invocation.jvp_dependencies->dependency_transport_count();
          }
        }
      candidate->external_boundary_dependency_contracts.reserve(transport_contract_count);
    } catch (...) {
      external_local_error = std::current_exception();
    }
    if (all_reduce_max(external_local_error ? 1L : 0L, *candidate->lane) != 0) {
      if (candidate->lane->size() == 1 && external_local_error)
        std::rethrow_exception(external_local_error);
      throw std::runtime_error(
          "AMR external boundary local candidate preparation failed collectively");
    }

    // All root/fine identities and request objects now exist on every rank. Converge those
    // retained requests before any ghost-fill collective consumes one of them.
    for (std::size_t block_index = 0; block_index < external_preparations.size(); ++block_index) {
      for (std::size_t level = 0; level < level_count; ++level) {
        const auto& prepared = external_preparations[block_index][level];
        for (const auto& invocation : prepared.ghosts)
          invocation.dependencies->require_collective_dependency_transport_requests(
              *candidate->lane);
        for (const auto& invocation : prepared.fluxes)
          invocation.dependencies->require_collective_dependency_transport_requests(
              *candidate->lane);
        for (const auto& invocation : prepared.fields) {
          invocation.residual_dependencies->require_collective_dependency_transport_requests(
              *candidate->lane);
          invocation.jvp_dependencies->require_collective_dependency_transport_requests(
              *candidate->lane);
        }
      }
    }

    // Collective constructors consume only prepared root/fine requests; canonical provider and
    // level order is unchanged and no broad catch spans a collective constructor.
    for (std::size_t block_index = 0; block_index < external_preparations.size(); ++block_index) {
      const std::size_t ghost_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].ghosts.size();
      for (std::size_t provider = 0; provider < ghost_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level)
          external_preparations[block_index][level]
              .ghosts[provider]
              .dependencies->prepare_dependency_transports_collectively(
                  candidate->external_boundary_dependency_contracts, *candidate->lane);
      const std::size_t flux_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].fluxes.size();
      for (std::size_t provider = 0; provider < flux_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level)
          external_preparations[block_index][level]
              .fluxes[provider]
              .dependencies->prepare_dependency_transports_collectively(
                  candidate->external_boundary_dependency_contracts, *candidate->lane);
      const std::size_t field_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].fields.size();
      for (std::size_t provider = 0; provider < field_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level) {
          auto& invocation = external_preparations[block_index][level].fields[provider];
          invocation.residual_dependencies->prepare_dependency_transports_collectively(
              candidate->external_boundary_dependency_contracts, *candidate->lane);
          invocation.jvp_dependencies->prepare_dependency_transports_collectively(
              candidate->external_boundary_dependency_contracts, *candidate->lane);
        }
    }

    const auto finish_session =
        [&]<class Component>(
            const std::shared_ptr<Component>& provider,
            std::optional<typename Component::template LocalSessionCandidate<Dim>>& local,
            std::shared_ptr<std::optional<typename Component::Session>>& session) {
          runtime::program::collective_boundary_provider_phase(
              *candidate->lane, "AMR external boundary provider preparation failed collectively",
              [&] {
                if (!session || session->has_value() || !local)
                  throw std::logic_error("AMR external boundary session holder is invalid");
                session->emplace(provider->template finish_session<Dim>(std::move(*local)));
                local.reset();
              });
        };
    // Provider callbacks retain full prepare-time collective authority. Enter each exact callback
    // only after the preceding provider and level has converged.
    for (std::size_t block_index = 0; block_index < external_preparations.size(); ++block_index) {
      const std::size_t ghost_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].ghosts.size();
      for (std::size_t provider = 0; provider < ghost_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level) {
          auto& invocation = external_preparations[block_index][level].ghosts[provider];
          finish_session(invocation.provider, invocation.local_session, invocation.session);
        }
      const std::size_t flux_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].fluxes.size();
      for (std::size_t provider = 0; provider < flux_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level) {
          auto& invocation = external_preparations[block_index][level].fluxes[provider];
          finish_session(invocation.provider, invocation.local_session, invocation.session);
        }
      const std::size_t field_count =
          level_count == 0 ? 0 : external_preparations[block_index][0].fields.size();
      for (std::size_t provider = 0; provider < field_count; ++provider)
        for (std::size_t level = 0; level < level_count; ++level) {
          auto& invocation = external_preparations[block_index][level].fields[provider];
          finish_session(invocation.residual, invocation.residual_local_session,
                         invocation.residual_session);
          finish_session(invocation.jvp, invocation.jvp_local_session, invocation.jvp_session);
        }
    }

    for (std::size_t block_index = 0; block_index < prepared_blocks.size(); ++block_index) {
      const prepared_block_type& prepared_block = prepared_blocks[block_index];
      const auto* installed_boundary = boundary_registry.find_boundary(prepared_block.name);
      std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> block_boundary;
      std::string block_boundary_identity;
      if (installed_boundary != nullptr) {
        block_boundary = installed_boundary->authority;
        block_boundary_identity = installed_boundary->identity;
      }
      for (std::size_t level = 0; level < level_count; ++level) {
        std::optional<level_block_type> prepared_level;
        std::exception_ptr level_error;
        long level_failure = 0;
        try {
          field_type& state = candidate_multiblock.state(block_index, level);
          auxiliary_groups_type& provider_storage = *candidate->provider_storage[level];
          auxiliary_groups_type& provider_candidate = *candidate->provider_candidate_storage[level];
          const Box<Dim>& level_domain = candidate_engine.hierarchy().layout(level).domain();
          runtime::amr::PreparedAmrGhostFill<Dim> state_ghost_fill;
          PreparedProviderGroupsGhostFill<Dim> provider_ghost_fill;
          PreparedRootAmrGhostFill<Dim> root_state_ghost_fill;
          PreparedProviderGroupsGhostFill<Dim> root_provider_ghost_fill;
          if (level == 0) {
            root_state_ghost_fill = prepare_root_ghost_fill(
                state, level_domain, exact_topology,
                candidate->block_state_field_identities[block_index][level],
                candidate->topology_epoch, candidate->materialization_generation, *candidate->lane);
            if (prepared_block.provider_components != 0 && !provider_storage.groups.empty())
              root_provider_ghost_fill = prepare_provider_groups_root_ghost_fill(
                  provider_storage, level_domain, exact_topology,
                  candidate->provider_storage_field_identities[level], candidate->topology_epoch,
                  candidate->materialization_generation, *candidate->lane);
            if (block_index == 0 && !provider_candidate.groups.empty())
              candidate->provider_candidate_ghost_fills[level] =
                  prepare_provider_groups_root_ghost_fill(
                      provider_candidate, level_domain, exact_topology,
                      candidate->provider_storage_field_identities[level] + "/candidate",
                      candidate->topology_epoch, candidate->materialization_generation,
                      *candidate->lane);
          } else {
            const Box<Dim>& coarse_domain = candidate_engine.hierarchy().layout(level - 1).domain();
            const auto& ratio = candidate_engine.hierarchy().layout(level).ratio_from_parent();
            state_ghost_fill = runtime::amr::prepare_amr_ghost_fill(
                candidate_multiblock.state(block_index, level - 1), state,
                runtime::amr::AmrGhostFillPreparation<Dim>{
                    .fine_level = static_cast<int>(level),
                    .coarse_domain = coarse_domain,
                    .fine_domain = level_domain,
                    .ratio = ratio,
                    .interpolation_kind = coarse_fine_transfer_kind(static_cast<int>(level - 1)),
                    .topology = exact_topology,
                    .topology_generation = candidate->topology_epoch,
                    .materialization_generation = candidate->materialization_generation,
                    .field_identity = candidate->block_state_field_identities[block_index][level],
                    .budget =
                        exact_amr_ghost_budget(candidate_multiblock.state(block_index, level - 1),
                                               state, coarse_domain, level_domain, exact_topology),
                },
                *candidate->lane);
            if (prepared_block.provider_components != 0 && !provider_storage.groups.empty())
              provider_ghost_fill = prepare_provider_groups_fine_ghost_fill(
                  *candidate->provider_storage[level - 1], provider_storage, coarse_domain,
                  level_domain, ratio, exact_topology,
                  candidate->provider_storage_field_identities[level], static_cast<int>(level),
                  candidate->topology_epoch, candidate->materialization_generation,
                  *candidate->lane);
            if (block_index == 0 && !provider_candidate.groups.empty())
              candidate->provider_candidate_ghost_fills[level] =
                  prepare_provider_groups_fine_ghost_fill(
                      *candidate->provider_candidate_storage[level - 1], provider_candidate,
                      coarse_domain, level_domain, ratio, exact_topology,
                      candidate->provider_storage_field_identities[level] + "/candidate",
                      static_cast<int>(level), candidate->topology_epoch,
                      candidate->materialization_generation, *candidate->lane);
          }
          if (block_index == 0 && !provider_candidate.groups.empty())
            candidate->provider_candidate_physical_boundaries[level].emplace(
                runtime::system::prepare_auxiliary_physical_boundaries(
                    provider_candidate, candidate->auxiliary_registries[level], level_domain,
                    Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper), exact_topology,
                    &*candidate->lane));
          auto& prepared_external = external_preparations[block_index][level];
          std::vector<CompiledFieldBoundaryKernel<Dim>> external_field_boundaries;
          external_field_boundaries.reserve(prepared_external.fields.size());
          for (const auto& invocation : prepared_external.fields) {
            const auto& residual = invocation.residual;
            const auto& jvp = invocation.jvp;
            const auto& residual_dependencies = invocation.residual_dependencies;
            const auto& jvp_dependencies = invocation.jvp_dependencies;
            const auto& residual_session = invocation.residual_session;
            const auto& jvp_session = invocation.jvp_session;
            const int selected_face = invocation.selected_face;
            CompiledFieldBoundaryKernel<Dim> kernel;
            kernel.identity = residual->spec().producer_identity + "/" + invocation.pair_key;
            kernel.residual_identity = residual->spec().outputs.front();
            kernel.jvp_identity = jvp->spec().outputs.front();
            kernel.residual = [provider = residual, residual_session, residual_dependencies,
                               selected_face, lane = &*candidate->lane](
                                  int face, const auto& iterate, auto& output, const auto& geometry,
                                  const auto& context) {
              if (face == selected_face) {
                std::unique_lock invocation_lock(residual_session->value().invocation_mutex(),
                                                 std::defer_lock);
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary residual session admission failed collectively", [&] {
                      if (!invocation_lock.try_lock())
                        throw std::logic_error(
                            "AMR FieldBoundary residual dependency cycle/reentrancy detected");
                    });
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary residual dependency refresh failed collectively",
                    [&] {
                      residual_dependencies->refresh(context.point, context.clock_identity, *lane);
                    });
                FieldBoundaryExecutionContext<Dim> routed;
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary residual context validation failed collectively",
                    [&] {
                      routed = residual_dependencies->view(context.point, context.clock_identity);
                    });
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary residual callback failed collectively", [&] {
                      provider->template evaluate_field_boundary_face<Dim>(
                          residual_session->value(), face, iterate, nullptr, output, geometry,
                          routed);
                    });
              }
            };
            kernel.jvp = [provider = jvp, jvp_session, jvp_dependencies, selected_face,
                          lane = &*candidate->lane](int face, const auto& iterate,
                                                    const auto& direction, auto& output,
                                                    const auto& geometry, const auto& context) {
              if (face == selected_face) {
                std::unique_lock invocation_lock(jvp_session->value().invocation_mutex(),
                                                 std::defer_lock);
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary JVP session admission failed collectively", [&] {
                      if (!invocation_lock.try_lock())
                        throw std::logic_error(
                            "AMR FieldBoundary JVP dependency cycle/reentrancy detected");
                    });
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary JVP dependency refresh failed collectively", [&] {
                      jvp_dependencies->refresh(context.point, context.clock_identity, *lane);
                    });
                FieldBoundaryExecutionContext<Dim> routed;
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary JVP context validation failed collectively", [&] {
                      routed = jvp_dependencies->view(context.point, context.clock_identity);
                    });
                runtime::program::collective_boundary_provider_phase(
                    *lane, "AMR FieldBoundary JVP callback failed collectively", [&] {
                      provider->template evaluate_field_boundary_face<Dim>(
                          jvp_session->value(), face, iterate, &direction, output, geometry,
                          routed);
                    });
              }
            };
            kernel.observes_iteration = !residual->spec().nonlinear_iterate.empty();
            kernel.validate();
            external_field_boundaries.push_back(std::move(kernel));
          }
          struct PreparedGhostInvocation {
            std::shared_ptr<PreparedGhostBoundaryComponent> provider;
            std::shared_ptr<std::optional<PreparedGhostBoundaryComponent::Session>> session;
            std::shared_ptr<PreparedBoundaryContext> dependencies;
          };
          struct PreparedFluxInvocation {
            std::shared_ptr<PreparedBoundaryFluxComponent> provider;
            std::shared_ptr<std::optional<PreparedBoundaryFluxComponent::Session>> session;
            std::shared_ptr<PreparedBoundaryContext> dependencies;
          };
          std::vector<PreparedGhostInvocation> prepared_ghosts;
          std::vector<PreparedFluxInvocation> prepared_fluxes;
          prepared_ghosts.reserve(prepared_external.ghosts.size());
          for (const auto& invocation : prepared_external.ghosts)
            prepared_ghosts.push_back(
                {invocation.provider, invocation.session, invocation.dependencies});
          prepared_fluxes.reserve(prepared_external.fluxes.size());
          for (const auto& invocation : prepared_external.fluxes)
            prepared_fluxes.push_back(
                {invocation.provider, invocation.session, invocation.dependencies});
          GeneratedAmrLevelContext<Dim> context{
              .level = level,
              .lane = &*candidate->lane,
              .state = &state,
              .geometry = Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper),
              .topology = exact_topology,
              .provider_storage =
                  prepared_block.provider_components == 0 || provider_storage.groups.empty()
                      ? nullptr
                      : &provider_storage,
              .provider_plan =
                  prepared_block.provider_components == 0
                      ? nullptr
                      : &auxiliary_registry.consumer_plan(prepared_block.provider_consumer_qid),
              .state_ghost_fill = std::move(state_ghost_fill),
              .provider_ghost_fill = std::move(provider_ghost_fill),
              .root_state_ghost_fill = std::move(root_state_ghost_fill),
              .root_provider_ghost_fill = std::move(root_provider_ghost_fill),
              .external_ghost_boundary =
                  prepared_ghosts.empty()
                      ? decltype(GeneratedAmrLevelContext<Dim>::external_ghost_boundary){}
                      : [providers = std::move(prepared_ghosts)](
                            const auto& point, auto& state, const auto& geometry,
                            const auto& lane) {
                          for (const auto& invocation : providers) {
                            FieldLogicalTimePoint logical;
                            logical.time = static_cast<Real>(point.physical_time);
                            logical.dt = static_cast<Real>(point.dt);
                            logical.stage_slot = point.stage;
                            logical.level = point.level;
                            logical.step = point.tick;
                            logical.substep = point.substep;
                            logical.stage_fraction_numerator = point.stage_fraction.numerator;
                            logical.stage_fraction_denominator = point.stage_fraction.denominator;
                            std::unique_lock invocation_lock(
                                invocation.session->value().invocation_mutex(), std::defer_lock);
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR GhostBoundary session admission failed collectively",
                                [&] {
                                  if (!invocation_lock.try_lock())
                                    throw std::logic_error(
                                        "AMR GhostBoundary dependency cycle/reentrancy detected");
                                });
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR GhostBoundary dependency refresh failed collectively",
                                [&] {
                                  invocation.dependencies->refresh(logical, &point.clock, lane);
                                });
                            FieldBoundaryExecutionContext<Dim> context;
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR GhostBoundary context validation failed collectively",
                                [&] {
                                  context = invocation.dependencies->view(logical, &point.clock);
                                });
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR GhostBoundary callback failed collectively", [&] {
                                  invocation.provider->template apply_ghost_region<Dim>(
                                      invocation.session->value(), point, state, geometry, lane,
                                      context);
                                });
                          }
                        },
              .external_boundary_flux =
                  prepared_fluxes.empty()
                      ? decltype(GeneratedAmrLevelContext<Dim>::external_boundary_flux){}
                      : [providers = std::move(prepared_fluxes)](
                            const auto& point, const auto& state, auto& faces,
                            const auto& geometry, const auto& lane) {
                          for (const auto& invocation : providers) {
                            FieldLogicalTimePoint logical;
                            logical.time = static_cast<Real>(point.physical_time);
                            logical.dt = static_cast<Real>(point.dt);
                            logical.stage_slot = point.stage;
                            logical.level = point.level;
                            logical.step = point.tick;
                            logical.substep = point.substep;
                            logical.stage_fraction_numerator = point.stage_fraction.numerator;
                            logical.stage_fraction_denominator = point.stage_fraction.denominator;
                            std::unique_lock invocation_lock(
                                invocation.session->value().invocation_mutex(), std::defer_lock);
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR BoundaryFlux session admission failed collectively",
                                [&] {
                                  if (!invocation_lock.try_lock())
                                    throw std::logic_error(
                                        "AMR BoundaryFlux dependency cycle/reentrancy detected");
                                });
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR BoundaryFlux dependency refresh failed collectively",
                                [&] {
                                  invocation.dependencies->refresh(logical, &point.clock, lane);
                                });
                            FieldBoundaryExecutionContext<Dim> context;
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR BoundaryFlux context validation failed collectively",
                                [&] {
                                  context = invocation.dependencies->view(logical, &point.clock);
                                });
                            runtime::program::collective_boundary_provider_phase(
                                lane, "AMR BoundaryFlux callback failed collectively", [&] {
                                  invocation.provider->template transform_boundary_flux<Dim>(
                                      invocation.session->value(), point, state, faces, geometry, lane,
                                      context);
                                });
                          }
                        },
              .external_field_boundaries = std::move(external_field_boundaries),
              .physical_boundary = block_boundary,
              .embedded_boundary = candidate->embedded_boundary[level],
              .state_identity = candidate->block_state_field_identities[block_index][level],
              .provider_storage_identity = candidate->provider_storage_field_identities[level],
              .boundary_identity = block_boundary_identity,
              .embedded_boundary_provider_identity =
                  !candidate->embedded_boundary[level] ||
                          embedded_boundary_mode ==
                              runtime::system::PreparedEmbeddedBoundaryMode::inactive
                      ? std::string{}
                  : embedded_boundary_mode ==
                          runtime::system::PreparedEmbeddedBoundaryMode::staircase
                      ? prepared_block.staircase_provider_identity
                      : prepared_block.cut_cell_provider_identity,
              .clock_identity_capacity = kPreparedAmrClockIdentityCapacity,
          };
          prepared_level.emplace(
              prepared_block.prepare_level(candidate_engine, std::move(context)));
          candidate->block_evaluations[block_index][level].emplace(
              make_prepared_level_evaluation_workspace(state, candidate->spatial_contract,
                                                       candidate->topology_epoch,
                                                       candidate->materialization_generation));
          candidate->block_evaluation_candidates[block_index][level].emplace(
              make_prepared_level_evaluation_workspace(state, candidate->spatial_contract,
                                                       candidate->topology_epoch,
                                                       candidate->materialization_generation));
          candidate->block_stage_scratch[block_index][level] =
              std::make_unique<typename PreparedHierarchy::StageScratch>(state);
        } catch (...) {
          level_failure = 1;
          level_error = std::current_exception();
        }
        if (all_reduce_max(level_failure, candidate->lane->communicator()) != 0) {
          if (level_error)
            std::rethrow_exception(level_error);
          throw std::runtime_error("generated AMR level preparation failed collectively");
        }
        candidate->block_levels[block_index].push_back(std::move(*prepared_level));
      }
    }

    std::exception_ptr contract_error;
    long contract_failure = 0;
    try {
      ExactContractBuilder contract;
      contract.text("pops.generated-amr-hierarchy-graph")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(candidate->spatial_contract)
          .bytes(candidate->package_contract)
          .bytes(candidate->embedded_boundary_configuration_contract)
          .text(candidate->embedded_boundary_materialization_digest)
          .text(candidate->lane->identity())
          .scalar(candidate->topology_epoch)
          .scalar(candidate->materialization_generation)
          .scalar(static_cast<std::uint64_t>(candidate->block_levels.size()));
      for (const auto& block : candidate->block_levels) {
        contract.scalar(static_cast<std::uint64_t>(block.size()));
        for (const level_block_type& level : block)
          contract.bytes(level.collective_contract());
      }
      contract.sequence(
          candidate->external_boundary_dependency_contracts,
          [](ExactContractBuilder& route, const std::string& exact) { route.bytes(exact); });
      for (std::size_t level = 0; level < candidate->provider_candidate_ghost_fills.size();
           ++level) {
        contract.optional_collective_contract(candidate->provider_candidate_ghost_fills[level]);
        contract.presence(candidate->provider_candidate_physical_boundaries[level].has_value());
        if (candidate->provider_candidate_physical_boundaries[level])
          contract.bytes(
              candidate->provider_candidate_physical_boundaries[level]->collective_contract());
      }
      candidate->collective_contract = std::move(contract).release();
    } catch (...) {
      contract_failure = 1;
      contract_error = std::current_exception();
    }
    if (all_reduce_max(contract_failure, candidate->lane->communicator()) != 0) {
      if (contract_error)
        std::rethrow_exception(contract_error);
      throw std::runtime_error("generated AMR hierarchy contract failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("generated-amr-hierarchy"),
              std::string_view(candidate->collective_contract)}},
            candidate->lane->communicator()))
      throw std::invalid_argument("generated AMR hierarchy contracts differ between MPI ranks");
    return candidate;
  }

  void refresh_prepared_hierarchy() const {
    if (!engine || prepared_blocks.empty())
      throw std::logic_error("AmrSystem cannot prepare levels before package materialization");
    const long has_graph = prepared_hierarchy ? 1L : 0L;
    std::optional<ExecutionCommunicator> parent;
    if (!prepared_hierarchy)
      parent.emplace(boundary_parent_communicator());
    const long minimum_graph =
        prepared_hierarchy ? all_reduce_min(has_graph, prepared_hierarchy->lane->communicator())
                           : all_reduce_min(has_graph, parent->communicator());
    const long maximum_graph =
        prepared_hierarchy ? all_reduce_max(has_graph, prepared_hierarchy->lane->communicator())
                           : all_reduce_max(has_graph, parent->communicator());
    if (minimum_graph != maximum_graph)
      throw std::runtime_error("prepared AMR graph publication differs between MPI ranks");
    const long stale =
        prepared_hierarchy && prepared_hierarchy->matches(*engine, *multiblock_hierarchy,
                                                          prepared_package_contract(),
                                                          embedded_boundary_configuration_contract)
            ? 0L
            : 1L;
    const long maximum_stale = prepared_hierarchy
                                   ? all_reduce_max(stale, prepared_hierarchy->lane->communicator())
                                   : all_reduce_max(stale, parent->communicator());
    if (maximum_stale == 0)
      return;
    std::unique_ptr<PreparedHierarchy> candidate =
        prepare_hierarchy_graph(*engine, *multiblock_hierarchy, prepared_hierarchy.get());
    std::optional<typename multiblock_type::ProgramBlockMap> block_map_candidate =
        prepare_program_block_map_candidate(*multiblock_hierarchy);
    std::optional<flux_expression_budget_type> flux_budget_candidate;
    if (program_flux_expression_budget) {
      if (!block_map_candidate)
        throw std::logic_error(
            "AMR Program flux-expression budget lost its exact Program block map");
      const bool has_flux_expression =
          flux_expression_budget_is_active(program_flux_expression_budget->blocks);
      flux_budget_candidate.emplace(prepare_program_flux_expression_budget(
          program_flux_expression_budget->program_hash, program_flux_expression_budget->blocks,
          *block_map_candidate, has_flux_expression, *engine, *multiblock_hierarchy));
    }
    prepared_hierarchy.swap(candidate);
    prepared_program_block_map = std::move(block_map_candidate);
    program_flux_expression_budget = std::move(flux_budget_candidate);
    pending_provider_registry_restore.reset();
    for (auto& [slot, plan] : field_plans) {
      (void)slot;
      plan.discard_materialization();
    }
    active_field_slot.clear();
  }

  void discard_level_evaluations() const noexcept {
    if (!prepared_hierarchy)
      return;
    for (auto& block : prepared_hierarchy->block_evaluation_published)
      std::fill(block.begin(), block.end(), false);
  }

  const ResolvedTaggingProgram& resolve_tagging_program() const {
    if (resolved_tagging)
      return *resolved_tagging;
    if (!tagging_spec || prepared_blocks.empty() || blocks.size() != 1)
      throw std::logic_error(
          "AMR prepared tagging requires one installed exact-ranked block and graph");

    ResolvedTaggingProgram candidate;
    candidate.program.stencils = tagging_spec->stencils;
    candidate.program.refine_ops = tagging_spec->refine_ops;
    candidate.program.refine_args = tagging_spec->refine_args;
    candidate.program.coarsen_ops = tagging_spec->coarsen_ops;
    candidate.program.coarsen_args = tagging_spec->coarsen_args;
    candidate.program.minimum_cycles = tagging_spec->min_cycles;
    candidate.program.equality_policy = tagging_spec->equality_policy;
    candidate.program.conflict_policy = tagging_spec->conflict_policy;
    candidate.program.non_finite_policy = POPS_TAGGING_NON_FINITE_REJECT_V1;
    candidate.program.clock_identity = tagging_spec->clock_identity;
    candidate.program.provider_identity = tagging_spec->provider_identity;
    candidate.program.prepared = true;

    const auto bind_field = [&](TaggingFieldKind kind, const std::string& identity,
                                const std::string& provider_group = {}) -> std::size_t {
      for (std::size_t index = 0; index < candidate.fields.size(); ++index)
        if (candidate.fields[index].qualified_identity == identity) {
          if (candidate.fields[index].kind != kind ||
              candidate.fields[index].provider_group != provider_group)
            throw std::invalid_argument(
                "AMR tagging qualified identity resolves to more than one storage authority");
          return index;
        }
      candidate.fields.push_back({kind, identity, provider_group});
      return candidate.fields.size() - 1;
    };

    for (std::size_t leaf_index = 0; leaf_index < tagging_spec->leaf_subject_kinds.size();
         ++leaf_index) {
      const std::string& kind = tagging_spec->leaf_subject_kinds[leaf_index];
      const std::string& identity = tagging_spec->leaf_subject_identities[leaf_index];
      const std::string& block_name = tagging_spec->leaf_blocks[leaf_index];
      const std::string& variable = tagging_spec->leaf_variables[leaf_index];
      std::size_t field_index = 0;
      int component = -1;
      if (kind == "state") {
        if (block_name != blocks.front().name)
          throw std::invalid_argument("AMR tagging state leaf names another native block");
        const std::string& installed = boundary_registry.state_route(block_name);
        if (identity != installed && identity != direct_amr_state_identity(block_name))
          throw std::invalid_argument(
              "AMR tagging state leaf differs from its exact qualified storage route");
        const auto found =
            std::find(prepared_blocks.front().conservative_variables.names.begin(),
                      prepared_blocks.front().conservative_variables.names.end(), variable);
        if (found == prepared_blocks.front().conservative_variables.names.end())
          throw std::invalid_argument("AMR tagging names an unknown conservative variable");
        component = static_cast<int>(
            std::distance(prepared_blocks.front().conservative_variables.names.begin(), found));
        field_index = bind_field(TaggingFieldKind::state, identity);
      } else if (kind == "aux") {
        throw std::invalid_argument(
            "AMR tagging auxiliary leaves must use an owner-qualified provider key; the retired "
            "shared auxiliary route has no valid component mapping");
      } else if (kind == "field") {
        if (!block_name.empty())
          throw std::invalid_argument("AMR tagging field leaf cannot carry a state block route");
        const std::string& slot = boundary_registry.field_storage_route(identity);
        const auto plan = field_plans.find(slot);
        if (plan == field_plans.end() || !plan->second.output)
          throw std::invalid_argument("AMR tagging field leaf has no exact published output");
        const int output_index = tagging_spec->leaf_field_component_indices[leaf_index];
        if (output_index < 0 ||
            static_cast<std::size_t>(output_index) >= plan->second.output_keys.size())
          throw std::out_of_range("AMR tagging field output slot is outside its exact carrier");
        const auto address = auxiliary_registry.address_of(
            plan->second.output_keys[static_cast<std::size_t>(output_index)]);
        component = static_cast<int>(address.component);
        field_index = bind_field(TaggingFieldKind::auxiliary, identity, address.group);
      } else {
        throw std::invalid_argument("AMR tagging leaf has an unknown subject kind");
      }
      const int stencil = tagging_spec->leaf_stencil_indices[leaf_index];
      candidate.program.leaves.push_back(
          {field_index, static_cast<std::size_t>(component),
           static_cast<std::int32_t>(tagging_spec->leaf_ops[leaf_index]),
           tagging_spec->leaf_thresholds[leaf_index],
           stencil < 0 ? POPS_TAGGING_NO_STENCIL_V1 : static_cast<std::size_t>(stencil)});
    }
    resolved_tagging.emplace(std::move(candidate));
    return *resolved_tagging;
  }

  std::vector<Box<Dim>> live_tagging_parent_domains() const {
    if (!engine)
      throw std::logic_error("AMR tagging checkpoint requires a materialized hierarchy");
    const std::size_t configured_parents =
        static_cast<std::size_t>(std::max(cfg.level_count - 1, 0));
    const std::size_t parent_count = std::min(engine->hierarchy().num_levels(), configured_parents);
    std::vector<Box<Dim>> domains;
    domains.reserve(parent_count);
    for (std::size_t level = 0; level < parent_count; ++level)
      domains.push_back(engine->hierarchy().layout(level).domain());
    return domains;
  }

  std::vector<runtime::program::AmrProgramHistoryDescriptor> history_descriptors() const {
    if (!engine)
      throw std::logic_error("AMR history checkpoint requires a materialized hierarchy");
    const auto& manager = program.hist_;
    struct Accumulated {
      runtime::program::AmrProgramHistoryDescriptor descriptor;
      std::set<int> levels;
    };
    std::map<std::string, Accumulated> logical;
    for (const auto& [key, ring] : manager.histories) {
      const auto decoded = decode_exact_amr_history_key(key);
      if (!decoded)
        throw std::invalid_argument(
            "AMR Program history manager contains a non-level-qualified ring");
      const auto& [level, name] = *decoded;
      if (level < 0 || static_cast<std::size_t>(level) >= engine->hierarchy().num_levels() ||
          ring.empty() || ring.front().ncomp() < 1)
        throw std::invalid_argument("AMR Program history ring has an invalid live-level image");
      const auto depth = manager.depth.find(key);
      const auto owner = manager.owner.find(key);
      const auto state = manager.state_identity.find(key);
      const auto space = manager.space_identity.find(key);
      const auto clock = manager.clock_identity.find(key);
      const auto interpolation = manager.interpolation_identity.find(key);
      if (depth == manager.depth.end() || owner == manager.owner.end() ||
          state == manager.state_identity.end() || space == manager.space_identity.end() ||
          clock == manager.clock_identity.end() ||
          interpolation == manager.interpolation_identity.end() || depth->second < 2 ||
          ring.size() != static_cast<std::size_t>(depth->second))
        throw std::invalid_argument("AMR Program history metadata is incomplete");
      int program_owner = -1;
      for (std::size_t index = 0; index < program.block_map_.size(); ++index)
        if (program.block_map_[index] == owner->second) {
          program_owner = static_cast<int>(index);
          break;
        }
      if (program_owner < 0)
        throw std::invalid_argument("AMR Program history has no exact program block owner");
      runtime::program::AmrProgramHistoryDescriptor descriptor{
          name,          program_owner,         state->second, space->second,
          clock->second, interpolation->second, depth->second, ring.front().ncomp()};
      auto [found, inserted] = logical.try_emplace(name, Accumulated{descriptor, {level}});
      if (!inserted) {
        const auto& retained = found->second.descriptor;
        if (retained.program_owner != descriptor.program_owner ||
            retained.state_identity != descriptor.state_identity ||
            retained.space_identity != descriptor.space_identity ||
            retained.clock_identity != descriptor.clock_identity ||
            retained.interpolation_identity != descriptor.interpolation_identity ||
            retained.depth != descriptor.depth || retained.components != descriptor.components ||
            !found->second.levels.insert(level).second)
          throw std::invalid_argument(
              "AMR Program history contract differs between hierarchy levels");
      }
    }
    std::vector<runtime::program::AmrProgramHistoryDescriptor> result;
    result.reserve(logical.size());
    for (auto& [name, accumulated] : logical) {
      (void)name;
      if (accumulated.levels.size() != engine->hierarchy().num_levels())
        throw std::invalid_argument(
            "AMR Program history does not cover every active hierarchy level");
      result.push_back(std::move(accumulated.descriptor));
    }
    return result;
  }

  std::vector<runtime::program::AmrProgramHistorySlotProvenance> history_slot_provenance() const {
    if (!engine)
      throw std::logic_error("AMR history checkpoint requires a materialized hierarchy");
    std::vector<runtime::program::AmrProgramHistorySlotProvenance> result;
    for (const auto& [key, ring] : program.hist_.histories) {
      const auto decoded = decode_exact_amr_history_key(key);
      if (!decoded || ring.empty())
        throw std::invalid_argument("AMR Program history slot has an invalid qualified key");
      const auto& [level, name] = *decoded;
      const auto& dts = program.hist_.slot_dt.at(key);
      if (dts.size() != ring.size())
        throw std::invalid_argument("AMR Program history slot dt has an invalid depth");
      for (std::size_t slot = 0; slot < ring.size(); ++slot)
        result.push_back({name, level, static_cast<int>(slot), static_cast<double>(dts[slot]),
                          program.hist_.initialized.at(key), program.hist_.fill_count.at(key)});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
      return std::tie(left.name, left.level, left.slot) <
             std::tie(right.name, right.level, right.slot);
    });
    return result;
  }

  runtime::program::AmrProgramHistoryDescriptor history_descriptor(std::string_view name) const {
    const auto descriptors = history_descriptors();
    const auto found = std::find_if(descriptors.begin(), descriptors.end(),
                                    [&](const auto& row) { return row.name == name; });
    if (found == descriptors.end())
      throw std::out_of_range("AmrSystem has no exact history ring '" + std::string(name) + "'");
    return *found;
  }

  std::vector<std::string> history_level_keys(std::string_view name) const {
    if (!engine)
      throw std::logic_error("AMR history access requires a materialized hierarchy");
    std::vector<std::string> keys;
    keys.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
      std::string key = exact_amr_history_key(name, static_cast<int>(level));
      if (!program.hist_.histories.contains(key))
        throw std::out_of_range("AMR history ring is missing one active hierarchy level");
      keys.push_back(std::move(key));
    }
    return keys;
  }

  runtime::program::AmrProgramAcceptedState<Dim> minimal_program_accepted_state() const {
    if (!engine)
      throw std::logic_error("AMR Program checkpoint requires a materialized hierarchy");
    runtime::program::AmrProgramAcceptedState<Dim> state;
    state.spatial_contract = engine->spatial_contract();
    state.topology_epoch = engine->topology_epoch();
    state.materialization_generation = engine->materialization_generation();
    state.level_clocks.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level)
      state.level_clocks.push_back(
          {static_cast<int>(level), macro_step, ::pops::amr::Rational(0, 1), accepted_time});
    state.histories = history_descriptors();
    state.history_slots = history_slot_provenance();
    return state;
  }

  void requalify_runtime_owned_program_state(
      runtime::program::AmrProgramAcceptedState<Dim>& state) const {
    if (!engine)
      throw std::logic_error("AMR Program checkpoint requires a materialized hierarchy");
    state.spatial_contract = engine->spatial_contract();
    state.topology_epoch = engine->topology_epoch();
    state.materialization_generation = engine->materialization_generation();
    state.level_clocks.clear();
    state.level_clocks.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level)
      state.level_clocks.push_back(
          {static_cast<int>(level), macro_step, ::pops::amr::Rational(0, 1), accepted_time});
    state.histories = history_descriptors();
    state.history_slots = history_slot_provenance();
  }

  void publish_tagging_checkpoint() const {
    if (!tagging_spec || (tagging_spec->min_cycles == 0 && program_accepted_bytes.empty()))
      return;
    runtime::program::AmrProgramAcceptedState<Dim> state;
    const bool synthesize = program_accepted_bytes.empty();
    if (synthesize) {
      state = minimal_program_accepted_state();
    } else {
      state = runtime::program::deserialize_amr_program_accepted_state<Dim>(program_accepted_bytes);
      if (program_accepted_bytes_runtime_owned)
        requalify_runtime_owned_program_state(state);
      else
        runtime::program::require_live_amr_program_checkpoint(state, *engine);
    }
    state.tagging_hysteresis_state =
        tagging_state.encode(tagging_spec->min_cycles, tagging_spec->provider_identity);
    runtime::program::require_collective_amr_program_checkpoint_consensus(
        state, *prepared_hierarchy->lane);
    std::vector<std::uint8_t> candidate =
        runtime::program::serialize_amr_program_accepted_state(state);
    if (candidate == program_accepted_bytes)
      return;
    if (program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
    program_accepted_bytes = std::move(candidate);
    ++program_accepted_revision;
    if (synthesize)
      program_accepted_bytes_runtime_owned = true;
  }

  amr::transfer::TransferKind regrid_transfer_kind(int parent_level) const {
    if (blocks.size() != 1)
      throw std::logic_error("AMR regrid transfer requires one exact-ranked block package");
    const std::string& subject = boundary_registry.state_route(blocks.front().name);
    const auto subject_route =
        bootstrap_subject_routes.find(std::make_pair(subject, std::string("prolongation")));
    if (subject_route == bootstrap_subject_routes.end())
      return amr::transfer::TransferKind::LinearProlongation;
    const auto route = bootstrap_transfer_routes.find(subject_route->second);
    if (route == bootstrap_transfer_routes.end())
      throw std::logic_error("AMR regrid transfer route lost its exact provider authority");
    const std::size_t transition = static_cast<std::size_t>(parent_level);
    if (transition >= cfg.transition_ratios.size() ||
        route->second.refinement_ratio != cfg.transition_ratios[transition])
      throw std::invalid_argument(
          "AMR regrid transfer route does not authenticate this ranked refinement ratio");
    if (route->second.kernel == "conservative_linear")
      return amr::transfer::TransferKind::LinearProlongation;
    if (route->second.kernel == "conservative_injection")
      return amr::transfer::TransferKind::ConstantInjection;
    throw std::invalid_argument("AMR regrid selected an unsupported prolongation kernel");
  }

  amr::transfer::TransferKind coarse_fine_transfer_kind(int parent_level) const {
    amr::transfer::TransferKind selected =
        amr::transfer::TransferKind::CoarseFineGhostInterpolation;
    const std::size_t transition = static_cast<std::size_t>(parent_level);
    if (transition >= cfg.transition_ratios.size())
      throw std::out_of_range("AMR coarse/fine transfer transition is outside its hierarchy");
    for (const BlockSpec& block : blocks) {
      const std::string& subject = boundary_registry.state_route(block.name);
      const auto subject_route =
          bootstrap_subject_routes.find(std::make_pair(subject, std::string("coarse_fine_fill")));
      if (subject_route == bootstrap_subject_routes.end()) {
        if (block.reconstruction_order <= 2)
          continue;
        if (block.reconstruction_order == 5 && block.required_ghost_depth >= 3) {
          selected = amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation;
          continue;
        }
        throw std::invalid_argument(
            "AMR block has no coarse/fine provider matching its authenticated reconstruction "
            "order");
      }
      if (block.reconstruction_order > 5)
        throw std::invalid_argument(
            "AMR coarse/fine transfer has no provider for the block reconstruction order");
      const auto route = bootstrap_transfer_routes.find(subject_route->second);
      if (route == bootstrap_transfer_routes.end())
        throw std::logic_error("AMR coarse/fine transfer route lost its exact provider authority");
      if (route->second.refinement_ratio != cfg.transition_ratios[transition])
        throw std::invalid_argument(
            "AMR coarse/fine transfer route does not authenticate this ranked refinement ratio");
      if (route->second.kernel == "conservative_coarse_fine") {
        if (block.reconstruction_order > 2)
          throw std::invalid_argument(
              "AMR limited-linear coarse/fine route cannot satisfy the block reconstruction "
              "order");
        continue;
      }
      if (route->second.kernel == "conservative_polynomial5_coarse_fine") {
        if (block.reconstruction_order > 5)
          throw std::invalid_argument(
              "AMR fifth-order coarse/fine route cannot satisfy the block reconstruction order");
        selected = amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation;
        continue;
      }
      throw std::invalid_argument("AMR coarse/fine selected an unsupported native kernel");
    }
    return selected;
  }

  std::uint64_t tagging_generation() const {
    if (!engine)
      throw std::logic_error("AMR tagging generation requires a materialized hierarchy");
    if (engine->materialization_generation() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR tagging generation exceeds uint64_t");
    return engine->materialization_generation() + 1;
  }

  void prepare_tagging_execution() const {
    if ((tagger_component && component_tagging_plan &&
         component_tagging_plan->topology_generation() == tagging_generation()) ||
        (!tagger_component && tagging_plan &&
         tagging_plan->topology_generation() == tagging_generation()))
      return;
    const ResolvedTaggingProgram& resolved = resolve_tagging_program();
    if (!prepared_hierarchy || !prepared_hierarchy->lane ||
        prepared_hierarchy->provider_storage.size() != engine->hierarchy().num_levels())
      throw std::logic_error("AMR tagging requires the exact prepared hierarchy graph");
    using TaggingField =
        runtime::amr::PreparedTaggingField<Dim,
                                           typename Kokkos::DefaultExecutionSpace::memory_space>;
    std::vector<std::vector<TaggingField>> fields_by_level;
    std::vector<amr::hierarchy::LevelLayout<Dim>> layouts;
    std::vector<runtime::amr::PreparedTaggingExecutionBudget> budgets;
    fields_by_level.reserve(engine->hierarchy().num_levels());
    layouts.reserve(engine->hierarchy().num_levels());
    budgets.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
      std::vector<TaggingField> fields;
      fields.reserve(resolved.fields.size());
      for (const ResolvedTaggingField& field : resolved.fields) {
        const field_type* values = &engine->hierarchy().state(level);
        if (field.kind == TaggingFieldKind::auxiliary) {
          values = prepared_hierarchy->provider_storage[level]->find(field.provider_group);
          if (values == nullptr)
            throw std::logic_error("AMR tagging provider group is absent from the live hierarchy");
        }
        fields.push_back({field.qualified_identity, values});
      }
      auto budget = exact_tagging_budget(engine->hierarchy().layout(level),
                                         engine->hierarchy().state(level).local_rank());
      if (tagger_component) {
        budget.scratch_bytes =
            checked_size_product(budget.candidate_mask.owned_cells, std::size_t{8},
                                 "AMR component Tagger candidate staging exceeds size_t");
        if (tagger_component->spec.execution_mode == POPS_TAGGER_EXECUTION_HOST_V2)
          for (const TaggingField& field : fields)
            for (std::size_t local = 0; local < field.values->local_size(); ++local)
              budget.scratch_bytes = checked_size_sum(
                  budget.scratch_bytes,
                  checked_size_product(field.values->fab(local).size(), sizeof(Real),
                                       "AMR component Tagger host staging exceeds size_t"),
                  "AMR component Tagger host staging exceeds size_t");
      }
      fields_by_level.push_back(std::move(fields));
      layouts.push_back(engine->hierarchy().layout(level));
      budgets.push_back(budget);
    }
    if (tagger_component) {
      std::uint32_t periodic_axes = 0;
      for (int axis = 0; axis < Dim; ++axis)
        if (cfg.periodicity[static_cast<std::size_t>(axis)])
          periodic_axes |= std::uint32_t{1} << static_cast<unsigned>(axis);
      auto candidate = runtime::amr::PreparedTaggerComponent<Dim>::prepare(
          tagger_component->component, tagger_component->spec, resolved.program, fields_by_level,
          layouts, budgets, tagging_generation(), periodic_axes, *prepared_hierarchy->lane);
      component_tagging_plan =
          std::make_unique<runtime::amr::PreparedTaggerComponent<Dim>>(std::move(candidate));
      tagging_plan.reset();
    } else {
      auto candidate = runtime::amr::PreparedTaggingExecutionPlan<Dim>::prepare(
          resolved.program, fields_by_level, layouts, budgets, tagging_generation(),
          prepared_hierarchy->lane->communicator());
      tagging_plan =
          std::make_unique<runtime::amr::PreparedTaggingExecutionPlan<Dim>>(std::move(candidate));
      component_tagging_plan.reset();
    }
  }

  runtime::amr::PreparedTaggerCandidates<Dim> execute_tagging(int parent_level) const {
    if (!tagging_spec)
      throw std::logic_error("AMR hierarchy has no prepared tagging authority");
    if (!prepared_hierarchy || !prepared_hierarchy->lane)
      throw std::logic_error("AMR tagging lacks its prepared collective execution lane");
    const CommunicatorView communicator = prepared_hierarchy->lane->communicator();
    const long invalid_parent =
        parent_level < 0 ||
                static_cast<std::size_t>(parent_level) >= engine->hierarchy().num_levels() ||
                parent_level >= cfg.level_count - 1
            ? 1L
            : 0L;
    const long minimum_parent = all_reduce_min(static_cast<long>(parent_level), communicator);
    const long maximum_parent = all_reduce_max(static_cast<long>(parent_level), communicator);
    if (all_reduce_max(invalid_parent, communicator) != 0)
      throw std::out_of_range("AMR tagging parent lies outside the live configured hierarchy");
    if (minimum_parent != maximum_parent)
      throw std::invalid_argument("AMR tagging parent level differs between MPI ranks");
    prepare_tagging_execution();
    runtime::multiblock::BoundaryEvaluationPoint point;
    point.clock = tagging_spec->clock_identity;
    point.tick = macro_step;
    point.level = parent_level;
    point.stage_fraction = {0, 1};
    point.dt = 0.0;
    point.physical_time = accepted_time;
    field_type& state = engine->hierarchy().state(static_cast<std::size_t>(parent_level));
    prepared_hierarchy->block_levels.front()[static_cast<std::size_t>(parent_level)].prepare(point,
                                                                                             state);
    std::array<Real, Dim> spacing{};
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
        engine->hierarchy().layout(static_cast<std::size_t>(parent_level)).domain(), cfg.lower,
        cfg.upper);
    for (int axis = 0; axis < Dim; ++axis)
      spacing[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    if (tagger_component)
      return component_tagging_plan->execute(
          static_cast<std::size_t>(parent_level),
          engine->hierarchy().layout(static_cast<std::size_t>(parent_level)), spacing,
          tagging_generation(), static_cast<std::int64_t>(macro_step), accepted_time);
    return tagging_plan->execute(static_cast<std::size_t>(parent_level),
                                 engine->hierarchy().layout(static_cast<std::size_t>(parent_level)),
                                 spacing, tagging_generation());
  }

  std::vector<amr::tagging::TagMask<Dim>> prepare_cluster_shards(
      int parent_level, const runtime::amr::PreparedTaggerCandidates<Dim>& candidates,
      runtime::amr::PersistentTaggingState<Dim>& staged_state) const {
    const auto& layout = engine->hierarchy().layout(static_cast<std::size_t>(parent_level));
    const CommunicatorView communicator = prepared_hierarchy->lane->communicator();
    std::vector<char> refine = gather_tag_mask(candidates.refine, layout.domain(), communicator);
    std::vector<char> coarsen = gather_tag_mask(candidates.coarsen, layout.domain(), communicator);
    const std::vector<char> refine_equalities =
        gather_tag_mask(candidates.refine_equalities, layout.domain(), communicator);
    const std::vector<char> coarsen_equalities =
        gather_tag_mask(candidates.coarsen_equalities, layout.domain(), communicator);
    if (tagging_spec->equality_policy == 1)
      for (std::size_t index = 0; index < refine.size(); ++index)
        refine[index] = static_cast<char>(refine[index] != 0 || refine_equalities[index] != 0 ||
                                          coarsen_equalities[index] != 0);
    else if (tagging_spec->equality_policy == 2)
      for (std::size_t index = 0; index < coarsen.size(); ++index)
        coarsen[index] = static_cast<char>(coarsen[index] != 0 || refine_equalities[index] != 0 ||
                                           coarsen_equalities[index] != 0);

    std::optional<amr::hierarchy::LevelLayout<Dim>> child;
    if (static_cast<std::size_t>(parent_level + 1) < engine->hierarchy().num_levels())
      child = engine->hierarchy().layout(static_cast<std::size_t>(parent_level + 1));
    const std::vector<char> current = child_coverage_on_parent(layout, child);
    std::vector<char> target = current;
    for (std::size_t ordinal = 0; ordinal < target.size(); ++ordinal) {
      bool requests_refine = refine[ordinal] != 0;
      bool requests_coarsen = coarsen[ordinal] != 0;
      if (requests_refine && requests_coarsen) {
        if (tagging_spec->conflict_policy == 0)
          throw std::runtime_error("AMR tagging refine/coarsen predicates conflict");
        if (tagging_spec->conflict_policy == 1)
          requests_refine = requests_coarsen = false;
        else if (tagging_spec->conflict_policy == 2)
          requests_coarsen = false;
        else
          requests_refine = false;
      }
      const bool desired = requests_refine    ? true
                           : requests_coarsen ? false
                                              : current[ordinal] != 0;
      if (desired == (current[ordinal] != 0))
        continue;
      const Index<Dim> cell = unflatten(layout.domain(), ordinal);
      typename runtime::amr::PersistentTaggingState<Dim>::CellKey key{parent_level, cell};
      if (!staged_state.transition_allowed(key, tagging_spec->min_cycles))
        continue;
      target[ordinal] = desired ? char{1} : char{0};
      staged_state.record(key,
                          desired ? runtime::amr::PersistentTaggingState<Dim>::Decision::Refine
                                  : runtime::amr::PersistentTaggingState<Dim>::Decision::Coarsen,
                          tagging_spec->min_cycles);
    }

    std::array<std::size_t, Dim> reach{};
    std::array<std::size_t, Dim> width{};
    std::size_t neighborhood = 1;
    const std::size_t transition = static_cast<std::size_t>(parent_level);
    for (int axis = 0; axis < Dim; ++axis) {
      reach[axis] =
          checked_size_sum(static_cast<std::size_t>(cfg.transition_buffers[transition][axis]),
                           static_cast<std::size_t>(cfg.transition_lookaheads[transition][axis]),
                           "AMR tagging transition reach exceeds size_t");
      width[axis] =
          checked_size_sum(checked_size_product(std::size_t{2}, reach[axis],
                                                "AMR tagging transition width exceeds size_t"),
                           std::size_t{1}, "AMR tagging transition width exceeds size_t");
      neighborhood = checked_size_product(neighborhood, width[axis],
                                          "AMR tagging transition neighborhood exceeds size_t");
    }
    std::vector<char> buffered = target;
    for (std::size_t ordinal = 0; ordinal < target.size(); ++ordinal) {
      if (target[ordinal] == 0)
        continue;
      const Index<Dim> center = unflatten(layout.domain(), ordinal);
      for (std::size_t neighbor = 0; neighbor < neighborhood; ++neighbor) {
        std::size_t quotient = neighbor;
        std::optional<Index<Dim>> canonical = center;
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t digit = quotient % width[axis];
          quotient /= width[axis];
          const std::int64_t delta =
              static_cast<std::int64_t>(digit) - static_cast<std::int64_t>(reach[axis]);
          const std::int64_t proposed = static_cast<std::int64_t>(center[axis]) + delta;
          if (proposed >= layout.domain().lo[axis] && proposed <= layout.domain().hi[axis]) {
            (*canonical)[axis] = static_cast<int>(proposed);
            continue;
          }
          if (!topology().is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
            canonical.reset();
            break;
          }
          const std::int64_t length = layout.domain().length(axis);
          const std::int64_t shifted = proposed - layout.domain().lo[axis];
          const std::int64_t wrapped = ((shifted % length) + length) % length;
          (*canonical)[axis] =
              static_cast<int>(static_cast<std::int64_t>(layout.domain().lo[axis]) + wrapped);
        }
        if (canonical && layout_contains(layout, *canonical))
          buffered[offset(*canonical, layout.domain())] = char{1};
      }
    }

    std::vector<amr::tagging::TagMask<Dim>> shards;
    const mesh::RankSpace<Dim>& ranks = layout.distribution().rank_space();
    shards.reserve(ranks.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
      const Index<Dim> coordinate = ranks.coordinate(rank);
      shards.emplace_back(layout, coordinate, exact_tag_mask_budget(layout, coordinate));
      for (const auto& patch : shards.back().patches())
        for (std::size_t ordinal = 0; ordinal < patch.tags.size(); ++ordinal) {
          const Index<Dim> cell = unflatten(patch.box, ordinal);
          if (buffered[offset(cell, layout.domain())] != 0)
            shards.back().set(patch.global_patch, cell);
        }
    }
    return shards;
  }

  bool regrid_parent(
      int parent_level, const std::optional<SparseFieldImage<Dim>>& previous_child = std::nullopt,
      runtime::amr::PersistentTaggingState<Dim>* hierarchy_cycle_state = nullptr) const {
    runtime::amr::PreparedTaggerCandidates<Dim> candidates = execute_tagging(parent_level);
    std::optional<SparseFieldImage<Dim>> retained_child = previous_child;
    const std::size_t live_child = static_cast<std::size_t>(parent_level + 1);
    if (!retained_child && live_child < engine->hierarchy().num_levels())
      retained_child = gather_sparse_field(engine->hierarchy().state(live_child),
                                           engine->hierarchy().layout(live_child).domain(),
                                           prepared_hierarchy->lane->communicator());
    runtime::amr::PersistentTaggingState<Dim> staged_state =
        hierarchy_cycle_state ? *hierarchy_cycle_state : tagging_state;
    if (hierarchy_cycle_state == nullptr)
      staged_state.begin_cycle(tagging_spec->min_cycles);
    std::vector<amr::tagging::TagMask<Dim>> shards =
        prepare_cluster_shards(parent_level, candidates, staged_state);
    const auto& parent_layout = engine->hierarchy().layout(static_cast<std::size_t>(parent_level));
    const amr::tagging::ClusterOptions<Dim> cluster_options =
        exact_cluster_options(cfg, parent_layout);
    const amr::tagging::BergerRigoutsosProvider<Dim> clustering;
    amr::tagging::ClusterResult<Dim> clustered = clustering.cluster(shards, cluster_options);
    const amr::RefinementRatio<Dim> ratio =
        refinement_ratio(cfg.transition_ratios[static_cast<std::size_t>(parent_level)]);
    const amr::regridding::RegridPreparationBudget budget =
        exact_regrid_budget(parent_layout, ratio, clustered);
    auto prepared = engine->prepare_regrid(static_cast<std::size_t>(parent_level), ratio,
                                           std::move(clustered), budget, *prepared_hierarchy->lane);
    std::vector<std::optional<field_type>> child_states(multiblock_hierarchy->block_count());
    if (!prepared.removes_fine_level()) {
      for (std::size_t block = 0; block < multiblock_hierarchy->block_count(); ++block) {
        const field_type& parent_state =
            multiblock_hierarchy->state(block, static_cast<std::size_t>(parent_level));
        if (bootstrap_transaction) {
          child_states[block].emplace(
              prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
              parent_state.local_rank(), parent_state.ncomp(), parent_state.ghosts());
          child_states[block]->set_val(Real(0));
          continue;
        }
        std::optional<SparseFieldImage<Dim>> retained = block == 0 ? retained_child : std::nullopt;
        if (!retained && live_child < multiblock_hierarchy->level_count())
          retained = gather_sparse_field(multiblock_hierarchy->state(block, live_child),
                                         engine->hierarchy().layout(live_child).domain(),
                                         prepared_hierarchy->lane->communicator());
        child_states[block].emplace(transfer_regridded_state(
            parent_state, parent_layout, *prepared.fine_layout(), retained,
            prepared_hierarchy->lane->communicator(), regrid_transfer_kind(parent_level)));
      }
    }
    if (checkpoint_regrid_count_value == std::numeric_limits<int>::max())
      throw std::overflow_error("AMR checkpoint regrid count overflow");
    const std::uint64_t prior_topology_epoch = engine->topology_epoch();
    multiblock_hierarchy->publish_regrid(static_cast<std::size_t>(parent_level),
                                         std::move(prepared), std::move(child_states));
    if (engine->topology_epoch() != prior_topology_epoch)
      ++checkpoint_regrid_count_value;
    if (hierarchy_cycle_state != nullptr)
      *hierarchy_cycle_state = std::move(staged_state);
    else
      tagging_state = std::move(staged_state);
    tagging_plan.reset();
    component_tagging_plan.reset();
    refresh_prepared_hierarchy();
    program.refresh_hierarchy_state("AmrSystem::regrid_from_prepared_tagging");
    if (hierarchy_cycle_state == nullptr)
      publish_tagging_checkpoint();
    return static_cast<std::size_t>(parent_level + 1) < engine->hierarchy().num_levels();
  }

  void automatic_bootstrap() const {
    if (automatic_bootstrap_complete || cfg.explicit_bootstrap || !tagging_spec)
      return;
    if (std::any_of(tagging_spec->leaf_subject_kinds.begin(),
                    tagging_spec->leaf_subject_kinds.end(),
                    [](const std::string& kind) { return kind == "field"; }))
      throw std::logic_error(
          "automatic AMR bootstrap cannot evaluate a field tagging leaf before its exact field "
          "plan is materialized; enable explicit bootstrap and publish the field first");
    AcceptedSnapshot snapshot(*this);
    try {
      runtime::amr::PersistentTaggingState<Dim> staged_state = tagging_state;
      staged_state.begin_cycle(tagging_spec->min_cycles);
      for (int parent_level = 0; parent_level < cfg.level_count - 1; ++parent_level)
        if (!regrid_parent(parent_level, std::nullopt, &staged_state))
          break;
      tagging_state = std::move(staged_state);
      publish_tagging_checkpoint();
      automatic_bootstrap_complete = true;
    } catch (...) {
      snapshot.restore(*const_cast<Impl*>(this));
      throw;
    }
  }

  void ensure_engine() const {
    const ExecutionLane world_lane = ExecutionLane::world();
    const long materialized = engine ? 1L : 0L;
    if (all_reduce_min(materialized) != all_reduce_max(materialized))
      throw std::runtime_error("AmrSystem hierarchy materialization differs between MPI ranks");
    if (materialized != 0) {
      refresh_prepared_hierarchy();
      automatic_bootstrap();
      return;
    }
    if (blocks.empty())
      throw std::logic_error(
          "AmrSystem requires a dimension-qualified block before materialization");
    if (prepared_blocks.size() != blocks.size())
      throw std::logic_error("AmrSystem block registry differs from its prepared package registry");

    std::shared_ptr<engine_type> engine_candidate;
    std::unique_ptr<multiblock_type> multiblock_candidate;
    std::unique_ptr<PreparedHierarchy> hierarchy_candidate;
    std::optional<typename multiblock_type::ProgramBlockMap> block_map_candidate;
    std::exception_ptr engine_error;
    long engine_failure = 0;
    try {
      const Box<Dim> domain = cfg.index_domain();
      const mesh::BoxArray<Dim> patches(cfg.materialized_boxes());
      const mesh::RankSpace<Dim> ranks = process_rank_space<Dim>(world_lane);
      const Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(world_lane.rank()));
      mesh::Distribution<Dim> distribution;
      if (cfg.distribute_coarse) {
        const parallel::LoadBalancePreparationBudget budget{patches.size(), ranks.size(),
                                                            checked_layout_cells(patches)};
        distribution =
            load_balance->prepare(patches, ranks, budget, {}, world_lane).plan().distribution();
      } else {
        distribution = mesh::Distribution<Dim>::replicated(patches, ranks);
      }

      const mesh::BoxArrayValidationBudget layout_budget{patches.size(),
                                                         checked_pair_count(patches.size())};
      const amr::hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                                    amr::RefinementRatio<Dim>{}, layout_budget);
      const auto materialize_state = [&](const BlockSpec& block) {
        std::optional<field_type> state;
        std::optional<
            analytic::PreparedAnalyticMaterialization<Dim, typename field_type::memory_space>>
            analytic_materialization;
        std::exception_ptr local_error;
        try {
          state.emplace(patches, distribution, local_rank, block.ncomp, block.ghosts);
          state->set_val(Real(0));
          if (cfg.explicit_bootstrap) {
            if (!block.has_analytic_state && !block.has_state)
              throw std::logic_error(
                  "explicit AMR bootstrap requires one staged initial source per block");
          } else if (block.has_analytic_state) {
            const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, cfg.lower, cfg.upper);
            analytic_materialization.emplace(analytic::prepare_cell_average_materialization(
                *state, geometry, block.analytic_state));
          } else if (block.has_state) {
            write_field(*state, domain, block.state, block.ncomp);
          } else if (block.has_density) {
            write_component(*state, domain, block.density, 0);
          }
        } catch (...) {
          local_error = std::current_exception();
        }
        const long failures = all_reduce_sum(local_error ? 1L : 0L, world_lane.communicator());
        if (failures != 0) {
          if (world_lane.size() == 1 && local_error)
            std::rethrow_exception(local_error);
          throw std::runtime_error(
              "AMR initial materialization preparation failed collectively on " +
              std::to_string(failures) + " rank(s)");
        }
        if (analytic_materialization)
          (void)analytic::materialize_cell_average(*analytic_materialization,
                                                   world_lane.communicator());
        return std::move(*state);
      };
      field_type state = materialize_state(blocks.front());

      const std::size_t hierarchy_pairs = exact_hierarchy_pair_budget(cfg, patches.size());
      auto hierarchy = amr::hierarchy::AmrHierarchy<Dim>::from_coarse(
          coarse, std::move(state),
          amr::hierarchy::HierarchyValidationBudget{static_cast<std::size_t>(cfg.level_count),
                                                    hierarchy_pairs});
      engine_candidate = std::make_shared<engine_type>(std::move(hierarchy), load_balance,
                                                       "pops.amr-system.exact-ranked");
      std::vector<typename multiblock_type::AdditionalBlock> additional;
      additional.reserve(blocks.size() - 1);
      for (std::size_t block = 1; block < blocks.size(); ++block) {
        std::vector<field_type> levels;
        levels.push_back(materialize_state(blocks[block]));
        additional.push_back({blocks[block].name, std::move(levels)});
      }
      multiblock_type prepared_multiblock = multiblock_type::prepare_collectively(
          engine_candidate, blocks.front().name, std::move(additional),
          "pops.amr-system.multiblock");
      multiblock_candidate = std::make_unique<multiblock_type>(std::move(prepared_multiblock));
    } catch (...) {
      engine_failure = 1;
      engine_error = std::current_exception();
    }
    if (all_reduce_max(engine_failure) != 0) {
      if (engine_error)
        std::rethrow_exception(engine_error);
      throw std::runtime_error("AmrSystem hierarchy preparation failed collectively");
    }

    std::exception_ptr coupling_error;
    try {
      for (const auto& coupling : prepared_couplings)
        multiblock_candidate->install_prepared_coupling_operator(coupling.provider_contract,
                                                                 coupling.view, coupling.operation);
      multiblock_candidate->seal_couplings();
    } catch (...) {
      coupling_error = std::current_exception();
    }
    if (all_reduce_max(coupling_error ? 1L : 0L) != 0) {
      if (coupling_error)
        std::rethrow_exception(coupling_error);
      throw std::runtime_error("AmrSystem coupling materialization failed collectively");
    }

    std::exception_ptr graph_error;
    try {
      hierarchy_candidate =
          prepare_hierarchy_graph(*engine_candidate, *multiblock_candidate, nullptr);
    } catch (...) {
      graph_error = std::current_exception();
    }
    if (all_reduce_max(graph_error ? 1L : 0L) != 0) {
      if (graph_error)
        std::rethrow_exception(graph_error);
      throw std::runtime_error("AmrSystem prepared hierarchy graph failed collectively");
    }

    std::exception_ptr map_error;
    try {
      if (!program.block_map_.empty()) {
        if (program.block_map_.size() != blocks.size())
          throw std::invalid_argument(
              "AMR Program block map must cover every prepared carrier exactly once");
        std::vector<std::string> ordered;
        ordered.reserve(program.block_map_.size());
        for (const int runtime_block : program.block_map_) {
          if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= blocks.size())
            throw std::out_of_range("AMR Program block map contains an invalid runtime block");
          ordered.push_back(blocks[static_cast<std::size_t>(runtime_block)].name);
        }
        block_map_candidate.emplace(multiblock_candidate->prepare_program_block_map(ordered));
      }
    } catch (...) {
      map_error = std::current_exception();
    }
    if (all_reduce_max(map_error ? 1L : 0L) != 0) {
      if (map_error)
        std::rethrow_exception(map_error);
      throw std::runtime_error("AmrSystem Program block-map materialization failed collectively");
    }

    hierarchy_tensor_solver_providers->add(
        std::make_shared<runtime::program::CompositeTensorHierarchyProvider<Dim>>(),
        multiblock_candidate->lane());

    // Every allocation, provider installation, seal and exact-map preparation has completed on
    // local candidates.  These ownership moves are the sole publication boundary and cannot throw.
    static_assert(std::is_nothrow_move_assignable_v<decltype(engine)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(multiblock_hierarchy)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(prepared_hierarchy)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(prepared_program_block_map)>);
    engine = std::move(engine_candidate);
    multiblock_hierarchy = std::move(multiblock_candidate);
    prepared_hierarchy = std::move(hierarchy_candidate);
    prepared_program_block_map = std::move(block_map_candidate);
    pending_provider_registry_restore.reset();
    automatic_bootstrap();
  }

  template <class Function>
  decltype(auto) execute_transaction(Function&& function) {
    ensure_engine();
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
      throw;
    }
  }
};

template <int Dim>
AmrSystem<Dim>::AmrSystem(const AmrSystemConfig<Dim>& config) {
  validate_amr_config(config);
  p_ = std::make_unique<Impl>(config);
}

template <int Dim>
AmrSystem<Dim>::~AmrSystem() = default;

template <int Dim>
AmrSystem<Dim>::AmrSystem(AmrSystem&&) noexcept = default;

template <int Dim>
AmrSystem<Dim>& AmrSystem<Dim>::operator=(AmrSystem&&) noexcept = default;

template <int Dim>
void AmrSystem<Dim>::install_prepared_auxiliary_provider(
    runtime::system::PreparedAuxiliaryProvider<Dim> provider) {
  require_amr_assembling(p_->lifecycle, "install_prepared_auxiliary_provider");
  p_->auxiliary_registry.add(std::move(provider));
  p_->auxiliary_registry_consensus_verified = false;
}

template <int Dim>
void AmrSystem<Dim>::install_auxiliary_consumer_plan(
    runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan) {
  require_amr_assembling(p_->lifecycle, "install_auxiliary_consumer_plan");
  if (p_->auxiliary_registry.sealed())
    throw std::logic_error("AMR auxiliary consumer plans must be installed before registry seal");
  p_->auxiliary_registry.add_consumer_plan(std::move(plan));
  p_->auxiliary_registry_consensus_verified = false;
}

template <int Dim>
void AmrSystem<Dim>::seal_auxiliary_providers() {
  if (p_->auxiliary_registry.sealed()) {
    if (!p_->auxiliary_registry_consensus_verified &&
        !all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-auxiliary-registry", p_->auxiliary_registry.collective_contract()}}))
      throw std::runtime_error("AMR auxiliary registry differs across MPI ranks");
    p_->auxiliary_registry_consensus_verified = true;
    return;
  }
  auto candidate = p_->auxiliary_registry;
  std::exception_ptr error;
  try {
    candidate.seal();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("AMR auxiliary registry preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-auxiliary-registry", candidate.collective_contract()}}))
    throw std::runtime_error("AMR auxiliary registry differs across MPI ranks");
  p_->auxiliary_registry = std::move(candidate);
  p_->auxiliary_registry_consensus_verified = true;
}

template <int Dim>
void AmrSystem<Dim>::stage_auxiliary_input(const runtime::system::AuxiliaryComponentKey& key,
                                           const std::vector<double>& values) {
  seal_auxiliary_providers();
  const auto address = p_->auxiliary_registry.address_of(key);
  bool input = false;
  std::string provider_identity;
  for (std::size_t provider = 0; provider < p_->auxiliary_registry.provider_count(); ++provider) {
    const auto& candidate = p_->auxiliary_registry.provider(provider);
    for (const auto& output : candidate.outputs())
      if (output.key.exact_key() == key.exact_key()) {
        input = candidate.kind() == runtime::system::AuxiliaryProviderKind::input;
        provider_identity = candidate.identity();
      }
  }
  if (!input)
    throw std::invalid_argument("AMR auxiliary input key is not owned by an InputAux provider");
  if (values.size() != checked_cells(p_->cfg.index_domain()))
    throw std::invalid_argument("AMR auxiliary input differs from the exact level-zero domain");
  const std::string payload_key = key.exact_key();
  const std::string_view payload(reinterpret_cast<const char*>(values.data()),
                                 values.size() * sizeof(double));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-auxiliary-input-key", payload_key}, {"amr-auxiliary-input-values", payload}}))
    throw std::invalid_argument("AMR auxiliary input differs across MPI ranks");
  p_->staged_auxiliary_inputs[payload_key] = values;
  if (std::find(p_->dirty_auxiliary_providers.begin(), p_->dirty_auxiliary_providers.end(),
                provider_identity) == p_->dirty_auxiliary_providers.end())
    p_->dirty_auxiliary_providers.push_back(std::move(provider_identity));
  (void)address;
}

template <int Dim>
runtime::system::AuxiliaryStorageAddress<Dim> AmrSystem<Dim>::auxiliary_address(
    const runtime::system::AuxiliaryComponentKey& key) const {
  return p_->auxiliary_registry.address_of(key);
}

template <int Dim>
std::string AmrSystem<Dim>::auxiliary_registry_contract() const {
  return std::string(p_->auxiliary_registry.collective_contract());
}

template <int Dim>
const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
AmrSystem<Dim>::prepared_auxiliary_consumer_plan(const std::string& consumer_qid) const {
  return p_->auxiliary_registry.consumer_plan(consumer_qid);
}

template <int Dim>
const runtime::system::AuxiliaryStorageGroups<Dim>*
AmrSystem<Dim>::prepared_amr_provider_storage_groups(int level) const {
  // This accessor is used from rank-local Fab kernels after the Program context has collectively
  // refreshed the hierarchy.  It must not materialize or refresh here: ranks may own different
  // numbers of local Fabs and therefore call this seam a different number of times.
  if (!p_->engine || !p_->prepared_hierarchy)
    throw std::logic_error(
        "AMR provider storage lookup requires a collectively prepared hierarchy");
  if (level < 0 ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->provider_storage.size())
    throw std::out_of_range("AMR provider storage level lies outside the live hierarchy");
  const auto& groups = p_->prepared_hierarchy->provider_storage[static_cast<std::size_t>(level)];
  if (!groups)
    throw std::logic_error("AMR prepared hierarchy has no provider storage groups at this level");
  return groups.get();
}

template <int Dim>
const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
AmrSystem<Dim>::prepared_amr_auxiliary_consumer_plan(const std::string& consumer_qid,
                                                     int level) const {
  // Like the storage lookup above, consumer-plan binding is a rank-local hot-path operation.  The
  // enclosing Program resource traversal owns collective refresh and topology qualification.
  if (!p_->engine || !p_->prepared_hierarchy)
    throw std::logic_error(
        "AMR provider consumer-plan lookup requires a collectively prepared hierarchy");
  if (level < 0 ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->auxiliary_registries.size())
    throw std::out_of_range("AMR provider consumer-plan level lies outside the live hierarchy");
  return p_->prepared_hierarchy->auxiliary_registries[static_cast<std::size_t>(level)]
      .consumer_plan(consumer_qid);
}

template <int Dim>
std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>>
AmrSystem<Dim>::capture_auxiliary_checkpoint_accepted_state() const {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR auxiliary checkpoint requires a materialized hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>> result;
  long local_failure = 0;
  std::string collective_contract;
  try {
    if (!p_->dirty_auxiliary_providers.empty())
      throw std::logic_error(
          "AMR auxiliary checkpoint refuses dirty provider state before accepted publication");
    const auto& groups = p_->prepared_hierarchy->provider_storage;
    const auto& registries = p_->prepared_hierarchy->auxiliary_registries;
    if (groups.size() != registries.size())
      throw std::logic_error(
          "AMR auxiliary checkpoint hierarchy has mismatched groups and registries");
    result.reserve(registries.size());
    ExactContractBuilder exact;
    exact.text("pops.amr-exact-auxiliary-checkpoint")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(registries.size()));
    for (std::size_t level = 0; level < registries.size(); ++level) {
      if (!groups[level])
        throw std::logic_error("AMR auxiliary checkpoint has an empty level carrier");
      auto accepted = runtime::system::capture_auxiliary_checkpoint_state(registries[level]);
      runtime::system::require_auxiliary_checkpoint_storage(accepted, *groups[level]);
      const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
      for (auto& descriptor : accepted.groups) {
        const MultiFab<Dim>* const group = groups[level]->find(descriptor.identity);
        if (group == nullptr)
          throw std::logic_error("AMR auxiliary checkpoint lost a sealed storage group");
        descriptor.payload = gather_field(*group, domain, group->ncomp(), &lane);
      }
      const auto bytes = runtime::system::serialize_auxiliary_checkpoint_state(accepted);
      exact.scalar(static_cast<std::uint64_t>(level))
          .bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
      result.push_back(std::move(accepted));
    }
    collective_contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure, lane) != 0)
    throw std::runtime_error("AMR auxiliary checkpoint capture failed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.amr-auxiliary-checkpoint"), collective_contract}}, lane))
    throw std::runtime_error("AMR auxiliary checkpoint differs between communicator ranks");
  return result;
}

template <int Dim>
void AmrSystem<Dim>::restore_auxiliary_checkpoint_accepted_state(
    const std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>>& state) {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error(
        "AMR auxiliary checkpoint restore requires a materialized hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  long local_failure = 0;
  std::string collective_contract;
  try {
    if (!p_->dirty_auxiliary_providers.empty())
      throw std::logic_error("AMR auxiliary checkpoint restore refuses dirty live provider state");
    const auto& groups = p_->prepared_hierarchy->provider_storage;
    const auto& registries = p_->prepared_hierarchy->auxiliary_registries;
    if (state.size() != groups.size() || state.size() != registries.size())
      throw std::invalid_argument(
          "AMR auxiliary checkpoint level count differs from the hierarchy");
    ExactContractBuilder exact;
    exact.text("pops.amr-exact-auxiliary-checkpoint")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(state.size()));
    for (std::size_t level = 0; level < state.size(); ++level) {
      if (!groups[level])
        throw std::logic_error("AMR auxiliary checkpoint restore has an empty level carrier");
      runtime::system::require_auxiliary_checkpoint_storage(state[level], *groups[level]);
      const std::size_t cells = checked_cells(p_->engine->hierarchy().layout(level).domain());
      for (const auto& descriptor : state[level].groups) {
        if (descriptor.component_count > std::numeric_limits<std::size_t>::max() / cells ||
            descriptor.payload.size() != descriptor.component_count * cells)
          throw std::invalid_argument(
              "AMR auxiliary checkpoint group payload differs from its exact level shape");
      }
      const auto bytes = runtime::system::serialize_auxiliary_checkpoint_state(state[level]);
      exact.scalar(static_cast<std::uint64_t>(level))
          .bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    }
    collective_contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
  }
  if (all_reduce_max(local_failure, lane) != 0)
    throw std::invalid_argument("AMR auxiliary checkpoint preflight failed on at least one rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.amr-auxiliary-checkpoint"), collective_contract}}, lane))
    throw std::runtime_error("AMR auxiliary checkpoint differs between communicator ranks");

  using storage_snapshot_type = std::vector<runtime::system::AuxiliaryStorageGroups<Dim>>;
  using registry_snapshot_type = std::vector<runtime::system::ExactAuxiliaryRegistry<Dim>>;
  std::optional<storage_snapshot_type> storage_snapshot;
  std::optional<registry_snapshot_type> registry_snapshot;
  decltype(p_->dirty_auxiliary_providers) dirty_snapshot;
  std::exception_ptr snapshot_error;
  try {
    storage_snapshot.emplace();
    storage_snapshot->reserve(p_->prepared_hierarchy->provider_storage.size());
    for (const auto& level : p_->prepared_hierarchy->provider_storage) {
      if (!level)
        throw std::logic_error("AMR auxiliary checkpoint snapshot has an empty accepted carrier");
      storage_snapshot->push_back(*level);
    }
    registry_snapshot.emplace(p_->prepared_hierarchy->auxiliary_registries);
    dirty_snapshot = p_->dirty_auxiliary_providers;
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      snapshot_error, &lane,
      "AMR auxiliary checkpoint snapshot failed collectively before registry mutation");

  std::exception_ptr restore_error;
  try {
    // Values are staged and validated in private candidate carriers before accepted generations or
    // evaluation points are touched.  Thus no provider can observe checkpoint provenance paired
    // with the previous numerical payload.
    for (std::size_t level = 0; level < state.size(); ++level) {
      auto& candidate = *p_->prepared_hierarchy->provider_candidate_storage[level];
      copy_auxiliary_groups_in_place(*p_->prepared_hierarchy->provider_storage[level], candidate);
      const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
      for (const auto& descriptor : state[level].groups) {
        MultiFab<Dim>* const group = candidate.find(descriptor.identity);
        if (group == nullptr)
          throw std::logic_error("AMR auxiliary checkpoint candidate lost a storage group");
        write_field(*group, domain, descriptor.payload,
                    static_cast<int>(descriptor.component_count));
      }
      runtime::system::require_finite_auxiliary_groups(candidate, &lane,
                                                       "AMR auxiliary checkpoint candidate");
    }
    auto candidate_registries = *registry_snapshot;
    for (std::size_t level = 0; level < state.size(); ++level)
      runtime::system::restore_auxiliary_checkpoint_state(state[level], candidate_registries[level],
                                                          lane);
    static_assert(std::is_nothrow_swappable_v<
                  typename decltype(p_->prepared_hierarchy->provider_storage)::value_type>);
    for (std::size_t level = 0; level < state.size(); ++level)
      p_->prepared_hierarchy->provider_candidate_storage[level].swap(
          p_->prepared_hierarchy->provider_storage[level]);
    p_->prepared_hierarchy->auxiliary_registries.swap(candidate_registries);
  } catch (...) {
    restore_error = std::current_exception();
  }
  const long restore_failed = all_reduce_max(restore_error ? 1L : 0L, lane.communicator());
  if (restore_failed != 0) {
    std::exception_ptr rollback_error;
    try {
      for (std::size_t level = 0; level < storage_snapshot->size(); ++level)
        copy_auxiliary_groups_in_place((*storage_snapshot)[level],
                                       *p_->prepared_hierarchy->provider_storage[level]);
    } catch (...) {
      rollback_error = std::current_exception();
    }
    try {
      for (std::size_t level = 0; level < storage_snapshot->size(); ++level)
        copy_auxiliary_groups_in_place(*p_->prepared_hierarchy->provider_storage[level],
                                       *p_->prepared_hierarchy->provider_candidate_storage[level]);
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    try {
      p_->prepared_hierarchy->auxiliary_registries.swap(*registry_snapshot);
      p_->dirty_auxiliary_providers.swap(dirty_snapshot);
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        rollback_error, &lane, "AMR auxiliary checkpoint rollback failed collectively");
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        restore_error, &lane, "AMR auxiliary checkpoint restore failed collectively");
  }
  p_->dirty_auxiliary_providers.clear();
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::auxiliary_component(
    const runtime::system::AuxiliaryComponentKey& key, int level) const {
  p_->ensure_engine();
  if (level < 0 ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->provider_storage.size())
    throw std::out_of_range("AMR auxiliary component level lies outside the live hierarchy");
  const auto address = p_->auxiliary_registry.address_of(key);
  const auto* group =
      p_->prepared_hierarchy->provider_storage[static_cast<std::size_t>(level)]->find(
          address.group);
  if (group == nullptr || address.component >= static_cast<std::size_t>(group->ncomp()))
    throw std::logic_error("AMR auxiliary carrier lost its resolved storage address");
  const Box<Dim>& domain = p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
  const std::vector<double> packed = gather_field(*group, domain, group->ncomp());
  const std::size_t cells = checked_cells(domain);
  return {packed.begin() + static_cast<std::ptrdiff_t>(address.component * cells),
          packed.begin() + static_cast<std::ptrdiff_t>((address.component + 1) * cells)};
}

template <int Dim>
void AmrSystem<Dim>::refresh_auxiliary(const runtime::system::AuxiliaryEvaluationPoint& point) {
  seal_auxiliary_providers();
  p_->ensure_engine();
  std::exception_ptr hierarchy_error;
  try {
    if (!p_->prepared_hierarchy || p_->prepared_hierarchy->auxiliary_registries.size() !=
                                       p_->prepared_hierarchy->provider_storage.size())
      throw std::logic_error("AMR auxiliary hierarchy lost its per-level registries");
    if (!p_->prepared_hierarchy->lane ||
        p_->prepared_hierarchy->provider_candidate_storage.size() !=
            p_->prepared_hierarchy->provider_storage.size() ||
        p_->prepared_hierarchy->provider_candidate_ghost_fills.size() !=
            p_->prepared_hierarchy->provider_storage.size() ||
        p_->prepared_hierarchy->provider_candidate_physical_boundaries.size() !=
            p_->prepared_hierarchy->provider_storage.size())
      throw std::logic_error("AMR auxiliary candidate authorities are not materialized exactly");
  } catch (...) {
    hierarchy_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      hierarchy_error, nullptr, "AMR auxiliary hierarchy preflight failed collectively");
  auto& hierarchy = *p_->prepared_hierarchy;
  const ExecutionLane& lane = *hierarchy.lane;
  using transaction_type =
      typename runtime::system::ExactAuxiliaryRegistry<Dim>::PublicationTransaction;
  using storage_snapshot_type = std::vector<runtime::system::AuxiliaryStorageGroups<Dim>>;
  using registry_snapshot_type = std::vector<runtime::system::ExactAuxiliaryRegistry<Dim>>;
  std::optional<storage_snapshot_type> storage_snapshot;
  std::optional<registry_snapshot_type> registry_snapshot;
  std::vector<std::optional<transaction_type>> transactions;
  decltype(p_->dirty_auxiliary_providers) dirty_snapshot;
  const std::size_t level_count = hierarchy.provider_storage.size();
  std::exception_ptr snapshot_error;
  try {
    storage_snapshot.emplace();
    storage_snapshot->reserve(level_count);
    for (const auto& level : hierarchy.provider_storage) {
      if (!level)
        throw std::logic_error("AMR auxiliary snapshot has an empty accepted carrier");
      storage_snapshot->push_back(*level);
    }
    registry_snapshot.emplace(hierarchy.auxiliary_registries);
    transactions.resize(level_count);
    dirty_snapshot = p_->dirty_auxiliary_providers;
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      snapshot_error, &lane,
      "AMR auxiliary snapshot failed collectively before multi-level publication");

  const auto rollback_all_levels = [&]() -> std::exception_ptr {
    for (auto& transaction : transactions)
      if (transaction)
        transaction->reject();
    std::exception_ptr rollback_error;
    try {
      for (std::size_t level = 0; level < level_count; ++level)
        copy_auxiliary_groups_in_place((*storage_snapshot)[level],
                                       *hierarchy.provider_storage[level]);
    } catch (...) {
      rollback_error = std::current_exception();
    }
    try {
      for (std::size_t level = 0; level < level_count; ++level)
        copy_auxiliary_groups_in_place(*hierarchy.provider_storage[level],
                                       *hierarchy.provider_candidate_storage[level]);
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    try {
      hierarchy.auxiliary_registries.swap(*registry_snapshot);
      p_->dirty_auxiliary_providers.swap(dirty_snapshot);
    } catch (...) {
      if (!rollback_error)
        rollback_error = std::current_exception();
    }
    return rollback_error;
  };
  const auto rollback_and_rethrow = [&](std::exception_ptr operation_error,
                                        const char* operation_message) {
    const std::exception_ptr rollback_error = rollback_all_levels();
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        rollback_error, &lane, "AMR auxiliary multi-level rollback failed collectively");
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(operation_error, &lane,
                                                                        operation_message);
  };

  std::exception_ptr preparation_error;
  try {
    for (std::size_t level = 0; level < level_count; ++level) {
      runtime::system::ExactAuxiliaryRegistry<Dim>* registry = nullptr;
      runtime::system::AuxiliaryStorageGroups<Dim>* candidate = nullptr;
      runtime::system::AuxiliaryEvaluationPoint level_point;
      std::exception_ptr candidate_error;
      try {
        registry = &hierarchy.auxiliary_registries.at(level);
        candidate = hierarchy.provider_candidate_storage.at(level).get();
        if (candidate == nullptr)
          throw std::logic_error("AMR auxiliary candidate carrier is empty");
        level_point = point;
        level_point.level = static_cast<int>(level);
        transactions[level].emplace(
            registry->begin_publication(level_point, p_->dirty_auxiliary_providers));
      } catch (...) {
        candidate_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          candidate_error, &lane,
          "AMR auxiliary candidate preparation failed collectively before input staging");
      auto& transaction = *transactions[level];
      if (hierarchy.provider_storage[level]->groups.empty())
        continue;

      std::vector<std::size_t> due_inputs;
      std::exception_ptr staging_error;
      try {
        copy_auxiliary_groups_in_place(*hierarchy.provider_storage[level], *candidate);
        due_inputs.reserve(registry->topological_order().size());
        for (std::size_t provider_index : registry->topological_order()) {
          const auto& provider = registry->provider(provider_index);
          if (provider.kind() != runtime::system::AuxiliaryProviderKind::input ||
              !transaction.requires_staging(provider.identity()))
            continue;
          due_inputs.push_back(provider_index);
          for (const auto& output : provider.outputs()) {
            const auto staged = p_->staged_auxiliary_inputs.find(output.key.exact_key());
            if (staged == p_->staged_auxiliary_inputs.end())
              throw std::runtime_error("AMR InputAux provider is due but has no staged value");
            const auto address = registry->address_of(output.key);
            if (candidate->find(address.group) == nullptr)
              throw std::logic_error("AMR auxiliary candidate lacks a resolved storage group");
            if (level != 0 &&
                hierarchy.provider_candidate_storage[level - 1]->find(address.group) == nullptr)
              throw std::logic_error("AMR auxiliary transfer lost a parent candidate group");
          }
        }
      } catch (...) {
        staging_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          staging_error, &lane,
          "AMR auxiliary input staging preflight failed collectively before ghost preparation");

      for (std::size_t provider_index : due_inputs) {
        const auto& provider = registry->provider(provider_index);
        std::exception_ptr provider_staging_error;
        try {
          for (const auto& output : provider.outputs()) {
            const auto staged = p_->staged_auxiliary_inputs.find(output.key.exact_key());
            const auto address = registry->address_of(output.key);
            auto* group = candidate->find(address.group);
            if (level == 0) {
              write_component(*group, p_->engine->hierarchy().layout(0).domain(), staged->second,
                              static_cast<int>(address.component));
            } else {
              const auto* parent =
                  hierarchy.provider_candidate_storage[level - 1]->find(address.group);
              auto transferred = transfer_regridded_state(
                  *parent, p_->engine->hierarchy().layout(level - 1),
                  p_->engine->hierarchy().layout(level), std::optional<SparseFieldImage<Dim>>{},
                  lane.communicator(), amr::transfer::TransferKind::ConstantInjection);
              copy_full_field_in_place(transferred, *group);
            }
          }
          transaction.stage_external(provider.identity());
        } catch (...) {
          provider_staging_error = std::current_exception();
        }
        runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
            provider_staging_error, &lane,
            "AMR auxiliary input staging failed collectively before provider ghost fill");
      }
      runtime::multiblock::BoundaryEvaluationPoint boundary_point;
      std::exception_ptr point_error;
      try {
        boundary_point = auxiliary_boundary_evaluation_point(level_point);
      } catch (...) {
        point_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          point_error, &lane,
          "AMR auxiliary boundary point failed collectively before provider ghost fill");
      const auto refresh_candidate_ghosts = [&] {
        PreparedProviderGroupsGhostFill<Dim>* fill = nullptr;
        runtime::system::PreparedAuxiliaryPhysicalBoundaries<Dim>* physical = nullptr;
        std::exception_ptr authority_error;
        try {
          auto& prepared_fill = hierarchy.provider_candidate_ghost_fills[level];
          auto& prepared_physical = hierarchy.provider_candidate_physical_boundaries[level];
          if (!prepared_fill || !prepared_physical)
            throw std::logic_error("AMR auxiliary candidate ghost authority is incomplete");
          fill = &prepared_fill;
          physical = &*prepared_physical;
        } catch (...) {
          authority_error = std::current_exception();
        }
        runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
            authority_error, &lane, "AMR auxiliary candidate ghost authority failed collectively");
        std::exception_ptr hierarchy_ghost_error;
        try {
          (*fill)(*candidate, boundary_point);
        } catch (...) {
          hierarchy_ghost_error = std::current_exception();
        }
        runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
            hierarchy_ghost_error, &lane, "AMR auxiliary hierarchy ghost fill failed collectively");
        physical->execute(*candidate);
      };
      refresh_candidate_ghosts();
      transaction.launch_ready_native(
          {hierarchy.provider_storage[level].get(), candidate},
          [&](const auto&, std::exception_ptr local_error) {
            runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
                local_error, &lane,
                "AMR auxiliary native provider launch failed collectively before ghost fill");
            refresh_candidate_ghosts();
          });
      std::exception_ptr fence_error;
      try {
        Kokkos::fence();
      } catch (...) {
        fence_error = std::current_exception();
      }
      runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
          fence_error, &lane, "AMR auxiliary device fence failed collectively");
      runtime::system::require_finite_auxiliary_groups(*candidate, &lane,
                                                       "AMR auxiliary publication");
    }
    for (const auto& transaction : transactions)
      transaction->validate_complete();
  } catch (...) {
    preparation_error = std::current_exception();
  }
  const long preparation_failed = all_reduce_max(preparation_error ? 1L : 0L, lane.communicator());
  if (preparation_failed != 0) {
    rollback_and_rethrow(preparation_error,
                         "AMR auxiliary multi-level preparation failed collectively");
    return;
  }

  std::exception_ptr storage_commit_error;
  try {
    for (std::size_t level = 0; level < level_count; ++level)
      copy_auxiliary_groups_in_place(*hierarchy.provider_candidate_storage[level],
                                     *hierarchy.provider_storage[level]);
  } catch (...) {
    storage_commit_error = std::current_exception();
  }
  const long storage_commit_failed =
      all_reduce_max(storage_commit_error ? 1L : 0L, lane.communicator());
  if (storage_commit_failed != 0) {
    rollback_and_rethrow(storage_commit_error,
                         "AMR auxiliary multi-level storage commit failed collectively");
    return;
  }

  std::exception_ptr metadata_commit_error;
  try {
    for (auto& transaction : transactions)
      transaction->accept();
  } catch (...) {
    metadata_commit_error = std::current_exception();
  }
  const long metadata_commit_failed =
      all_reduce_max(metadata_commit_error ? 1L : 0L, lane.communicator());
  if (metadata_commit_failed != 0) {
    rollback_and_rethrow(metadata_commit_error,
                         "AMR auxiliary multi-level metadata commit failed collectively");
    return;
  }
  p_->dirty_auxiliary_providers.clear();
}

template <int Dim>
void AmrSystem<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                               const std::string&, const std::string&, const std::string&, int, int,
                               const std::vector<std::string>&, const std::vector<std::string>&,
                               const NewtonOptions&, bool, double, double, bool) {
  throw std::runtime_error(
      "AmrSystem ModelSpec installation requires a dimension-qualified compiled block provider");
}

template <int Dim>
void AmrSystem<Dim>::set_compiled_block(int ncomp, double gamma, int substeps,
                                        AmrCompiledBlockBuilder<Dim> runtime_builder,
                                        const std::string& name, bool, const std::string& time,
                                        int stride, const std::vector<std::string>& implicit_vars,
                                        const std::vector<std::string>& implicit_roles, double,
                                        double, bool) {
  (void)ncomp;
  (void)gamma;
  (void)substeps;
  (void)runtime_builder;
  (void)name;
  (void)time;
  (void)stride;
  (void)implicit_vars;
  (void)implicit_roles;
  throw std::logic_error(
      "AmrSystem::set_compiled_block is retired: install one complete "
      "PreparedAmrSystemBlock<Dim> through the exact generated-package seam");
}

template <int Dim>
void AmrSystem<Dim>::add_native_block(const std::string& name, const std::string& so_path,
                                      const std::string& limiter, const std::string& riemann,
                                      const std::string& recon, const std::string& time,
                                      double gamma, int substeps, const std::vector<double>& params,
                                      double positivity_floor, double weno_epsilon,
                                      bool wave_speed_cache) {
  using register_auxiliary_type = void (*)(AmrSystem<Dim>*);
  using install_type = void (*)(void*, const char*, const char*, const char*, const char*,
                                const char*, double, int, const double*, int, double, double, bool);

  struct FieldPlanSnapshot {
    std::optional<runtime::field::NamedFieldOutput<Dim>> output;
    std::vector<runtime::system::AuxiliaryComponentKey> output_keys;
    std::vector<std::vector<typename Impl::PreparedFieldRhs>> rhs_by_block;
    bool use_prepared_level_rhs = false;
  };

  std::shared_ptr<pops::dynlib::handle> package_lifetime;
  register_auxiliary_type register_auxiliary = nullptr;
  install_type install = nullptr;
  std::optional<typename Impl::auxiliary_registry_type> auxiliary_snapshot;
  std::optional<typename Impl::boundary_registry_type> boundary_snapshot;
  std::vector<typename Impl::BlockSpec> blocks_snapshot;
  std::vector<typename Impl::prepared_block_type> prepared_blocks_snapshot;
  std::map<std::string, FieldPlanSnapshot> field_plan_snapshots;
  bool auxiliary_consensus_snapshot = false;
  std::string preparation_contract;
  std::exception_ptr preparation_error;

  try {
    require_amr_assembling(p_->lifecycle, "add_native_block");
    if (p_->engine || p_->prepared_hierarchy)
      throw std::logic_error("AmrSystem native packages must be installed before materialization");
    if (std::any_of(p_->blocks.begin(), p_->blocks.end(),
                    [&](const auto& block) { return block.name == name; }))
      throw std::logic_error("AmrSystem native package block identities must be unique");
    if (name.empty() || so_path.empty())
      throw std::invalid_argument(
          "AmrSystem native package requires non-empty block and artifact identities");
    if (!std::isfinite(gamma) || !(gamma > 0.0) || substeps < 1)
      throw std::invalid_argument(
          "AmrSystem native package requires finite positive gamma and substeps");
    if (!std::isfinite(positivity_floor) || positivity_floor < 0.0)
      throw std::invalid_argument(
          "AmrSystem native package positivity floor must be finite and non-negative");
    if (!std::isfinite(weno_epsilon) || !(weno_epsilon > 0.0))
      throw std::invalid_argument(
          "AmrSystem native package WENO epsilon must be finite and positive");
    if (limiter != "weno5" && weno_epsilon != static_cast<double>(kWenoEpsilon))
      throw std::invalid_argument("AmrSystem native package WENO epsilon requires limiter='weno5'");
    if (wave_speed_cache)
      throw std::invalid_argument(
          "AmrSystem native package has no prepared exact-ranked wave-speed cache provider");
    (void)parse_limiter_route(limiter, "AmrSystem native package");
    (void)parse_riemann_route(riemann, "AmrSystem native package");
    (void)parse_recon_route(recon, "AmrSystem native package");
    (void)parse_time_route(time, "AmrSystem native package");

#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&detail::abi_key_string), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
#endif
    const pops::dynlib::handle handle = pops::dynlib::open(so_path);
    if (!pops::dynlib::valid(handle))
      throw std::runtime_error("AmrSystem::add_native_block: cannot load '" + so_path +
                               "': " + pops::dynlib::last_error());
    package_lifetime = std::shared_ptr<pops::dynlib::handle>(new pops::dynlib::handle(handle),
                                                             [](pops::dynlib::handle* owned) {
                                                               pops::dynlib::close(*owned);
                                                               delete owned;
                                                             });

    const auto key =
        reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_native_abi_key"));
    if (key == nullptr || key() == nullptr)
      throw std::runtime_error(
          "AmrSystem::add_native_block: pops_native_abi_key is missing; rebuild the artifact");
    const std::string artifact_key = key();
    const std::string module_key = detail::abi_key_string();
    if (artifact_key != module_key)
      throw std::runtime_error(
          "AmrSystem::add_native_block: compiled package ABI differs from the native module");

    const NativeAmrPackageMetadata metadata = inspect_native_amr_package(handle, params);
    register_auxiliary = reinterpret_cast<register_auxiliary_type>(
        pops::dynlib::sym(handle, "pops_register_provider_routes_amr"));
    install = reinterpret_cast<install_type>(pops::dynlib::sym(handle, "pops_install_native_amr"));
    if (install == nullptr)
      throw std::runtime_error(
          "AmrSystem::add_native_block: pops_install_native_amr is missing; compile the package "
          "with target='amr_system'");

    auxiliary_snapshot.emplace(p_->auxiliary_registry);
    boundary_snapshot.emplace(p_->boundary_registry);
    blocks_snapshot = p_->blocks;
    prepared_blocks_snapshot = p_->prepared_blocks;
    auxiliary_consensus_snapshot = p_->auxiliary_registry_consensus_verified;
    for (const auto& [slot, plan] : p_->field_plans)
      field_plan_snapshots.emplace(
          slot, FieldPlanSnapshot{plan.output, plan.output_keys, plan.rhs_by_block,
                                  plan.use_prepared_level_rhs});
    p_->native_package_lifetimes.reserve(p_->native_package_lifetimes.size() + 1);

    ExactContractBuilder exact;
    exact.text("pops.amr-system.native-package-preflight")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(name)
        .text(limiter)
        .text(riemann)
        .text(recon)
        .text(time)
        .scalar(gamma)
        .scalar(std::int32_t{substeps})
        .sequence(params)
        .scalar(positivity_floor)
        .scalar(weno_epsilon)
        .scalar(wave_speed_cache)
        .text(artifact_key)
        .text(metadata.route_manifest)
        .text(metadata.parameter_names)
        .scalar(std::int32_t{metadata.parameter_count})
        .presence(register_auxiliary != nullptr);
    preparation_contract = std::move(exact).release();
  } catch (...) {
    preparation_error = std::current_exception();
  }

  if (all_reduce_max(preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AmrSystem native package preflight failed on at least one MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-native-package"), std::string_view(preparation_contract)}}))
    throw std::invalid_argument("AmrSystem native package contracts differ between MPI ranks");

  auto rollback = [&] {
    p_->auxiliary_registry = *auxiliary_snapshot;
    p_->auxiliary_registry_consensus_verified = auxiliary_consensus_snapshot;
    p_->boundary_registry = *boundary_snapshot;
    p_->blocks = blocks_snapshot;
    p_->prepared_blocks = prepared_blocks_snapshot;
    for (const auto& [slot, snapshot] : field_plan_snapshots) {
      auto found = p_->field_plans.find(slot);
      if (found == p_->field_plans.end())
        throw std::logic_error("AMR native package changed the exact field-plan registry");
      found->second.output = snapshot.output;
      found->second.output_keys = snapshot.output_keys;
      found->second.rhs_by_block = snapshot.rhs_by_block;
      found->second.use_prepared_level_rhs = snapshot.use_prepared_level_rhs;
    }
  };

  std::exception_ptr registration_error;
  try {
    if (register_auxiliary != nullptr)
      register_auxiliary(this);
  } catch (...) {
    registration_error = std::current_exception();
  }
  if (all_reduce_max(registration_error ? 1L : 0L) != 0) {
    std::exception_ptr rollback_error;
    try {
      rollback();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    if (all_reduce_max(rollback_error ? 1L : 0L) != 0) {
      p_->native_package_lifetimes.push_back(std::move(package_lifetime));
      throw std::runtime_error(
          "AmrSystem native auxiliary registration rollback failed collectively");
    }
    if (n_ranks() == 1 && registration_error)
      std::rethrow_exception(registration_error);
    throw std::runtime_error(
        "AmrSystem native auxiliary registration failed on at least one MPI rank");
  }

  std::exception_ptr installation_error;
  try {
    install(static_cast<void*>(this), name.c_str(), limiter.c_str(), riemann.c_str(), recon.c_str(),
            time.c_str(), gamma, substeps, params.empty() ? nullptr : params.data(),
            static_cast<int>(params.size()), positivity_floor, weno_epsilon, wave_speed_cache);
    if (p_->prepared_blocks.empty() || p_->blocks.empty() || p_->blocks.back().name != name ||
        p_->prepared_blocks.back().name != name)
      throw std::logic_error(
          "AmrSystem native package did not publish one complete prepared block");
  } catch (...) {
    installation_error = std::current_exception();
  }
  if (all_reduce_max(installation_error ? 1L : 0L) != 0) {
    std::exception_ptr rollback_error;
    try {
      rollback();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    if (all_reduce_max(rollback_error ? 1L : 0L) != 0) {
      p_->native_package_lifetimes.push_back(std::move(package_lifetime));
      throw std::runtime_error("AmrSystem native package rollback failed collectively");
    }
    if (n_ranks() == 1 && installation_error)
      std::rethrow_exception(installation_error);
    throw std::runtime_error("AmrSystem native package installation failed collectively");
  }
  p_->native_package_lifetimes.push_back(std::move(package_lifetime));
}

template <int Dim>
void AmrSystem<Dim>::register_external_riemann_package(
    const std::string& name, const std::string& so_path, const std::string& brick_id,
    const std::string& expected_sha256, int expected_nvars, int expected_provider_count,
    const std::string& expected_model_identity, const std::string& provider_consumer_qid,
    const std::string& limiter, const std::string& recon, const std::string& time, double gamma,
    int substeps, int stride, double positivity_floor, double weno_epsilon) {
  std::shared_ptr<runtime::program::ExternalBrickHandle> authority;
  std::optional<typename Impl::auxiliary_registry_type> provider_registry_snapshot;
  std::optional<typename Impl::boundary_registry_type> boundary_snapshot;
  std::vector<typename Impl::BlockSpec> blocks_snapshot;
  std::vector<typename Impl::prepared_block_type> prepared_blocks_snapshot;
  bool provider_consensus_snapshot = false;
  std::string preflight_contract;
  std::exception_ptr preflight_error;

  try {
    require_amr_assembling(p_->lifecycle, "register_external_riemann_package");
    runtime::program::detail::validate_external_install(name, limiter, recon, time,
                                                        provider_consumer_qid, gamma, substeps,
                                                        stride, positivity_floor, weno_epsilon);
    if (p_->engine || p_->prepared_hierarchy)
      throw std::logic_error(
          "AMR external Riemann packages must be installed before materialization");
    if (so_path.empty() || brick_id.empty() || expected_nvars < 1 || expected_provider_count < 0 ||
        expected_model_identity.empty())
      throw std::invalid_argument(
          "AMR external Riemann package requires complete artifact and model authority");
    if (std::any_of(p_->blocks.begin(), p_->blocks.end(),
                    [&](const auto& block) { return block.name == name; }))
      throw std::logic_error("AMR external Riemann block identities must be unique");
    if (expected_sha256.size() != 64 ||
        !std::all_of(expected_sha256.begin(), expected_sha256.end(), [](char value) {
          return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        }))
      throw std::invalid_argument(
          "AMR external Riemann package requires one lowercase SHA-256 digest");

    authority = std::make_shared<runtime::program::ExternalBrickHandle>(
        so_path, brick_id, expected_nvars, expected_provider_count, expected_model_identity,
        expected_sha256);

    provider_registry_snapshot.emplace(p_->auxiliary_registry);
    boundary_snapshot.emplace(p_->boundary_registry);
    blocks_snapshot = p_->blocks;
    prepared_blocks_snapshot = p_->prepared_blocks;
    provider_consensus_snapshot = p_->auxiliary_registry_consensus_verified;
    p_->external_package_lifetimes.reserve(p_->external_package_lifetimes.size() + 1);

    ExactContractBuilder exact;
    exact.text("pops.external-riemann.amr-package")
        .scalar(std::uint32_t{4})
        .scalar(std::int32_t{Dim})
        .text(name)
        .text(brick_id)
        .text(expected_sha256)
        .scalar(std::int32_t{expected_nvars})
        .scalar(std::int32_t{expected_provider_count})
        .text(expected_model_identity)
        .text(provider_consumer_qid)
        .text(limiter)
        .text(recon)
        .text(time)
        .scalar(gamma)
        .scalar(std::int32_t{substeps})
        .scalar(std::int32_t{stride})
        .scalar(positivity_floor)
        .scalar(weno_epsilon);
    preflight_contract = std::move(exact).release();
  } catch (...) {
    preflight_error = std::current_exception();
  }

  if (all_reduce_max(preflight_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && preflight_error)
      std::rethrow_exception(preflight_error);
    throw std::runtime_error(
        "AMR external Riemann package preflight failed on at least one MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs({{std::string_view("amr-external-riemann-package"),
                                                  std::string_view(preflight_contract)}}))
    throw std::invalid_argument("AMR external Riemann package contracts differ between MPI ranks");

  const auto rollback = [&] {
    p_->auxiliary_registry = *provider_registry_snapshot;
    p_->auxiliary_registry_consensus_verified = provider_consensus_snapshot;
    p_->boundary_registry = *boundary_snapshot;
    p_->blocks = blocks_snapshot;
    p_->prepared_blocks = prepared_blocks_snapshot;
  };

  std::exception_ptr registration_error;
  try {
    authority->register_amr_routes(*this);
  } catch (...) {
    registration_error = std::current_exception();
  }
  if (all_reduce_max(registration_error ? 1L : 0L) != 0) {
    std::exception_ptr rollback_error;
    try {
      rollback();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    if (all_reduce_max(rollback_error ? 1L : 0L) != 0) {
      p_->external_package_lifetimes.push_back(authority);
      throw std::runtime_error("AMR external Riemann provider-route rollback failed collectively");
    }
    if (n_ranks() == 1 && registration_error)
      std::rethrow_exception(registration_error);
    throw std::runtime_error(
        "AMR external Riemann provider-route registration failed collectively");
  }

  std::exception_ptr installation_error;
  try {
    authority->install_amr(this, name, provider_consumer_qid, limiter, recon, time, gamma, substeps,
                           stride, positivity_floor, weno_epsilon);
    if (p_->blocks.size() != blocks_snapshot.size() + 1 ||
        p_->prepared_blocks.size() != prepared_blocks_snapshot.size() + 1 ||
        p_->blocks.back().name != name || p_->prepared_blocks.back().name != name)
      throw std::logic_error(
          "AMR external Riemann package did not publish one complete prepared block");
  } catch (...) {
    installation_error = std::current_exception();
  }
  if (all_reduce_max(installation_error ? 1L : 0L) != 0) {
    std::exception_ptr rollback_error;
    try {
      rollback();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    if (all_reduce_max(rollback_error ? 1L : 0L) != 0) {
      p_->external_package_lifetimes.push_back(authority);
      throw std::runtime_error("AMR external Riemann package rollback failed collectively");
    }
    if (n_ranks() == 1 && installation_error)
      std::rethrow_exception(installation_error);
    throw std::runtime_error("AMR external Riemann package installation failed collectively");
  }
  p_->external_package_lifetimes.push_back(std::move(authority));
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_block(PreparedBlock prepared) {
  std::exception_ptr preparation_error;
  long preparation_failure = 0;
  std::vector<typename Impl::BlockSpec> block_candidate;
  std::vector<PreparedBlock> prepared_candidates;
  std::shared_ptr<const HyperbolicBoundary> converted_boundary;
  std::string install_contract;
  bool has_boundary = false;
  try {
    require_amr_assembling(p_->lifecycle, "install_prepared_amr_block");
    validate_prepared_amr_block(prepared);
    if (p_->engine || p_->prepared_hierarchy)
      throw std::logic_error("prepared AMR blocks must be installed before materialization");
    if (std::any_of(p_->blocks.begin(), p_->blocks.end(),
                    [&](const auto& block) { return block.name == prepared.name; }))
      throw std::logic_error("prepared AMR block identities must be unique");

    const auto route = p_->boundary_registry.state_routes().find(prepared.name);
    if (route == p_->boundary_registry.state_routes().end())
      throw std::runtime_error("prepared AMR block lacks its exact pre-installed state identity");
    const auto* installed_boundary = p_->boundary_registry.find_boundary(prepared.name);
    const BoundaryTopology<Dim> exact_topology =
        BoundaryTopology<Dim>::axis_periodic(p_->cfg.periodicity);
    if (generated_amr_detail::has_physical_faces(exact_topology) && installed_boundary == nullptr)
      throw std::runtime_error(
          "prepared AMR block with physical faces requires a model-qualified boundary");
    if (installed_boundary != nullptr) {
      if (installed_boundary->state_identity != route->second ||
          installed_boundary->authority->ncomp() != prepared.ncomp ||
          installed_boundary->authority->periodic_axes() != p_->cfg.periodicity)
        throw std::invalid_argument(
            "prepared AMR block boundary differs from its exact state/domain contract");
      for (int axis = 0; axis < Dim; ++axis)
        if (prepared.ghosts[axis] < installed_boundary->required_depth)
          throw std::invalid_argument(
              "prepared AMR block ghosts are narrower than its boundary requirement");
      converted_boundary = std::make_shared<HyperbolicBoundary>(
          installed_boundary->authority->with_converted_fixed_states(
              prepared.primitive_to_conservative));
      has_boundary = true;
    }

    typename Impl::BlockSpec block;
    block.name = prepared.name;
    block.ncomp = prepared.ncomp;
    block.gamma = prepared.gamma;
    block.substeps = prepared.substeps;
    block.stride = prepared.stride;
    block.ghosts = prepared.ghosts;
    block.time = prepared.time_route;
    block.reconstruction_order = prepared.reconstruction_order;
    block.required_ghost_depth = 0;
    for (int axis = 0; axis < Dim; ++axis)
      block.required_ghost_depth =
          std::max(block.required_ghost_depth, static_cast<int>(prepared.ghosts[axis]));
    block_candidate = p_->blocks;
    block_candidate.reserve(block_candidate.size() + 1);
    block_candidate.push_back(std::move(block));

    ExactContractBuilder contract;
    contract.text("pops.amr-system.prepared-install")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(prepared.collective_contract)
        .text(prepared.provider_consumer_qid)
        .text(route->second)
        .scalar(has_boundary);
    if (installed_boundary != nullptr)
      contract.text(installed_boundary->identity)
          .scalar(std::int32_t{installed_boundary->required_depth})
          .scalar(std::int32_t{installed_boundary->authority->ncomp()})
          .bytes(exact_hyperbolic_boundary_contract(*converted_boundary));
    install_contract = std::move(contract).release();
    prepared_candidates = p_->prepared_blocks;
    prepared_candidates.reserve(prepared_candidates.size() + 1);
    prepared_candidates.push_back(std::move(prepared));
  } catch (...) {
    preparation_failure = 1;
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_failure) != 0) {
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("prepared AMR block validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("prepared-amr-install"), std::string_view(install_contract)}}))
    throw std::invalid_argument("prepared AMR block install contracts differ between MPI ranks");

  p_->blocks.swap(block_candidate);
  p_->prepared_blocks.swap(prepared_candidates);
  if (has_boundary)
    p_->boundary_registry.boundary(p_->blocks.back().name).authority =
        std::move(converted_boundary);
}

template <int Dim>
void AmrSystem<Dim>::install_tagger_component(
    std::shared_ptr<component::LoadedComponent> component, const std::string& component_id,
    const std::string& manifest_identity, std::uint32_t interface_version,
    const std::string& provider_identity, const std::string& tagging_graph_identity,
    const std::string& layout_identity, const std::string& clock_identity,
    const std::string& execution_mode, const std::string& parameters_json,
    const std::string& target_json,
    std::shared_ptr<const component::PreparedExecutionContextV1> execution) {
  typename Impl::TaggerComponentAuthority candidate;
  require_amr_assembling(p_->lifecycle, "install_tagger_component");
  if (p_->engine || p_->tagger_component || !component || !execution || component_id.empty() ||
      manifest_identity.empty() || provider_identity.empty() || tagging_graph_identity.empty() ||
      layout_identity.empty() || clock_identity.empty() || parameters_json.empty() ||
      target_json.empty() || interface_version != 2)
    throw std::invalid_argument(
        "AMR native Tagger requires one complete unique pre-materialization authority");
  const PopsTaggerExecutionModeV2 mode =
      execution_mode == "native_backend" ? POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2
      : execution_mode == "host"         ? POPS_TAGGER_EXECUTION_HOST_V2
                                         : static_cast<PopsTaggerExecutionModeV2>(0);
  if (mode == static_cast<PopsTaggerExecutionModeV2>(0))
    throw std::invalid_argument("AMR native Tagger execution mode is unknown");
  const PopsComponentApiV1& api = component->api();
  if (api.component_id == nullptr || api.manifest_identity == nullptr ||
      component_id != api.component_id || manifest_identity != api.manifest_identity)
    throw std::invalid_argument("AMR native Tagger loaded component identity differs");
  (void)component->table<PopsTaggerApiV2>(POPS_NATIVE_INTERFACE_TAGGER_V2, interface_version);
  candidate.component = std::move(component);
  candidate.spec = {
      component_id,         manifest_identity, provider_identity, tagging_graph_identity,
      layout_identity,      clock_identity,    interface_version, mode,
      std::move(execution), parameters_json,   target_json};
  // Installation is deliberately rank-local staging. The complete authority is authenticated and
  // the component state is prepared only after PreparedHierarchy publishes its explicit lane.
  p_->tagger_component.emplace(std::move(candidate));
}

template <int Dim>
void AmrSystem<Dim>::set_bootstrap_tagging(
    const std::vector<std::string>& leaf_subject_kinds,
    const std::vector<std::string>& leaf_subject_identities,
    const std::vector<std::string>& leaf_blocks, const std::vector<std::string>& leaf_variables,
    const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
    const std::vector<double>& leaf_thresholds, const std::vector<int>& leaf_stencil_indices,
    const std::vector<typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil>& stencils,
    const std::vector<std::int32_t>& refine_ops, const std::vector<std::int32_t>& refine_args,
    const std::vector<std::int32_t>& coarsen_ops, const std::vector<std::int32_t>& coarsen_args,
    int min_cycles, const std::string& equality_policy, const std::string& conflict_policy,
    const std::string& clock_identity, const std::string& provider_identity) {
  typename Impl::TaggingSpec candidate;
  std::string contract;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_bootstrap_tagging");
    const std::size_t leaves = leaf_subject_kinds.size();
    if (p_->engine || p_->tagging_spec || leaves == 0 || leaf_subject_identities.size() != leaves ||
        leaf_blocks.size() != leaves || leaf_variables.size() != leaves ||
        leaf_field_component_indices.size() != leaves || leaf_ops.size() != leaves ||
        leaf_thresholds.size() != leaves || leaf_stencil_indices.size() != leaves ||
        refine_ops.empty() || refine_ops.size() != refine_args.size() ||
        coarsen_ops.size() != coarsen_args.size() || min_cycles < 0 || clock_identity.empty() ||
        provider_identity.empty() ||
        (p_->tagger_component && p_->tagger_component->spec.clock_identity != clock_identity))
      throw std::invalid_argument(
          "AMR prepared tagging requires one complete unique pre-materialization graph");
    const int equality = equality_policy == "hold"      ? 0
                         : equality_policy == "refine"  ? 1
                         : equality_policy == "coarsen" ? 2
                                                        : -1;
    const int conflict = conflict_policy == "error"          ? 0
                         : conflict_policy == "hold"         ? 1
                         : conflict_policy == "refine_wins"  ? 2
                         : conflict_policy == "coarsen_wins" ? 3
                                                             : -1;
    if (equality < 0 || conflict < 0)
      throw std::invalid_argument("AMR prepared tagging has an unknown decision policy");
    for (std::size_t index = 0; index < leaves; ++index) {
      const std::string& kind = leaf_subject_kinds[index];
      if ((kind != "state" && kind != "aux" && kind != "field") ||
          leaf_subject_identities[index].empty() || leaf_variables[index].empty() ||
          !std::isfinite(leaf_thresholds[index]) ||
          !pops_tagging_opcode_is_leaf_v1(leaf_ops[index]) ||
          (kind == "state" && leaf_blocks[index].empty()) ||
          (kind != "state" && !leaf_blocks[index].empty()) ||
          (kind == "field" && leaf_field_component_indices[index] < 0) ||
          (kind != "field" && leaf_field_component_indices[index] != -1))
        throw std::invalid_argument("AMR prepared tagging has an invalid qualified leaf");
      if (kind == "field" && !p_->cfg.explicit_bootstrap)
        throw std::invalid_argument(
            "AMR field tagging requires explicit bootstrap so its exact field plan can be "
            "materialized before the first predicate evaluation");
      const bool gradient = leaf_ops[index] == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf_ops[index] == POPS_TAGGING_GRADIENT_BELOW_V1;
      const int stencil = leaf_stencil_indices[index];
      if (gradient != (stencil >= 0) ||
          (stencil >= 0 && static_cast<std::size_t>(stencil) >= stencils.size()))
        throw std::invalid_argument("AMR prepared tagging leaf/stencil relation is invalid");
    }
    const auto validate_bytecode = [&](const std::vector<std::int32_t>& ops,
                                       const std::vector<std::int32_t>& args, bool required) {
      if (ops.empty()) {
        if (required)
          throw std::invalid_argument("AMR prepared tagging has no refine predicate");
        return;
      }
      int depth = 0;
      for (std::size_t instruction = 0; instruction < ops.size(); ++instruction) {
        const int opcode = ops[instruction];
        const int argument = args[instruction];
        if (pops_tagging_opcode_is_leaf_v1(opcode)) {
          if (argument < 0 || static_cast<std::size_t>(argument) >= leaves ||
              leaf_ops[static_cast<std::size_t>(argument)] != opcode)
            throw std::invalid_argument("AMR prepared tagging bytecode names an invalid leaf");
          ++depth;
        } else if (opcode == POPS_TAGGING_NOT_V1) {
          if (argument != 1 || depth < 1)
            throw std::invalid_argument("AMR prepared tagging bytecode has an invalid NOT");
        } else if (opcode == POPS_TAGGING_ANY_OF_V1 || opcode == POPS_TAGGING_ALL_OF_V1) {
          if (argument < 2 || depth < argument)
            throw std::invalid_argument("AMR prepared tagging bytecode has an invalid arity");
          depth -= argument - 1;
        } else {
          throw std::invalid_argument("AMR prepared tagging bytecode has an unknown opcode");
        }
      }
      if (depth != 1)
        throw std::invalid_argument("AMR prepared tagging bytecode has an invalid final depth");
    };
    validate_bytecode(refine_ops, refine_args, true);
    validate_bytecode(coarsen_ops, coarsen_args, false);
    candidate = {leaf_subject_kinds,
                 leaf_subject_identities,
                 leaf_blocks,
                 leaf_variables,
                 leaf_field_component_indices,
                 leaf_ops,
                 leaf_thresholds,
                 leaf_stencil_indices,
                 stencils,
                 refine_ops,
                 refine_args,
                 coarsen_ops,
                 coarsen_args,
                 min_cycles,
                 equality,
                 conflict,
                 clock_identity,
                 provider_identity};
    ExactContractBuilder exact;
    exact.text("pops.amr-system.prepared-tagging-authoring")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .sequence(leaf_subject_kinds,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_subject_identities,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_blocks,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_variables,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_field_component_indices)
        .sequence(leaf_ops)
        .sequence(leaf_thresholds)
        .sequence(leaf_stencil_indices)
        .scalar(static_cast<std::uint64_t>(stencils.size()));
    for (const auto& stencil : stencils) {
      exact.text(stencil.identity)
          .text(stencil.route)
          .text(stencil.norm)
          .text(stencil.scale)
          .text(stencil.boundary_mode);
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& row = stencil.axes[static_cast<std::size_t>(axis)];
        exact.scalar(row.axis)
            .scalar(row.derivative_order)
            .scalar(row.formal_order)
            .scalar(static_cast<std::uint64_t>(row.ghost_lower))
            .scalar(static_cast<std::uint64_t>(row.ghost_upper))
            .sequence(row.offsets)
            .sequence(row.coefficients);
      }
    }
    exact.sequence(refine_ops)
        .sequence(refine_args)
        .sequence(coarsen_ops)
        .sequence(coarsen_args)
        .scalar(min_cycles)
        .scalar(equality)
        .scalar(conflict)
        .text(clock_identity)
        .text(provider_identity);
    contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR prepared tagging authoring failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("prepared-amr-tagging"), std::string_view(contract)}}))
    throw std::invalid_argument("AMR prepared tagging contracts differ between MPI ranks");
  p_->tagging_spec.emplace(std::move(candidate));
  p_->resolved_tagging.reset();
  p_->tagging_plan.reset();
  p_->component_tagging_plan.reset();
  p_->tagging_state.clear();
  p_->automatic_bootstrap_complete = false;
}

template <int Dim>
void AmrSystem<Dim>::set_temporal_relations(const std::vector<std::int64_t>& numerators,
                                            const std::vector<std::int64_t>& denominators,
                                            const std::vector<std::string>& remainder_policies) {
  std::vector<::pops::amr::ParentChildClockRelation> candidate;
  std::string contract;
  std::exception_ptr preparation_error;
  try {
    require_amr_assembling(p_->lifecycle, "set_temporal_relations");
    const std::size_t transitions = static_cast<std::size_t>(p_->cfg.level_count - 1);
    if (p_->engine || !p_->temporal_relations.empty() || numerators.size() != transitions ||
        denominators.size() != transitions || remainder_policies.size() != transitions)
      throw std::invalid_argument(
          "AMR temporal authority requires exactly one unique pre-materialization relation per "
          "level transition");
    candidate.reserve(transitions);
    ExactContractBuilder exact;
    exact.text("pops.amr-system.temporal-relations")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(transitions));
    for (std::size_t index = 0; index < transitions; ++index) {
      const auto policy = remainder_policies[index] == "integral_only"
                              ? ::pops::amr::RemainderPolicy::IntegralOnly
                          : remainder_policies[index] == "explicit_final_substep"
                              ? ::pops::amr::RemainderPolicy::ExplicitFinalSubstep
                              : throw std::invalid_argument(
                                    "AMR temporal relation has an unknown remainder policy");
      const ::pops::amr::Rational ratio(numerators[index], denominators[index]);
      candidate.emplace_back(static_cast<int>(index), static_cast<int>(index + 1), ratio, policy);
      exact.scalar(static_cast<std::int32_t>(index))
          .scalar(static_cast<std::int32_t>(index + 1))
          .scalar(ratio.numerator)
          .scalar(ratio.denominator)
          .text(remainder_policies[index]);
    }
    contract = std::move(exact).release();
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR temporal relation preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-temporal-relations"), std::string_view(contract)}}))
    throw std::invalid_argument("AMR temporal relations differ between MPI ranks");
  p_->temporal_relations = std::move(candidate);
}

template <int Dim>
void AmrSystem<Dim>::set_analytic_level_set(const std::vector<std::string>& opcodes,
                                            const std::vector<double>& literals,
                                            const std::string& mode, double kappa_min,
                                            double face_open_eps, double cut_theta_min) {
  runtime::system::PreparedEmbeddedBoundaryMode prepared_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  EbThresholds thresholds{};
  std::uint64_t generation = 0;
  std::vector<std::string> staged_opcodes;
  std::vector<double> staged_literals;
  PreparedAmrEbAuthoring authored;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_analytic_level_set");
    if (p_->prepared_blocks.empty())
      throw std::logic_error(
          "AmrSystem embedded geometry requires an installed exact generated block provider");
    if (p_->engine || p_->prepared_hierarchy)
      throw std::logic_error(
          "AmrSystem embedded geometry must be authored before hierarchy materialization");
    prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
    thresholds = resolved_amr_eb_thresholds(kappa_min, face_open_eps, cut_theta_min);
    generation = next_amr_eb_generation(p_->embedded_boundary_generation);
    staged_opcodes = opcodes;
    staged_literals = literals;
    authored = prepare_amr_eb_authoring(p_->cfg, p_->prepared_blocks.front(), staged_opcodes,
                                        staged_literals, prepared_mode, thresholds, generation);
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR embedded-boundary authoring failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-eb-configuration", authored.configuration_contract},
           {"amr-eb-semantic", authored.semantic_digest}}))
    throw std::invalid_argument(
        "AMR embedded-boundary authoring contracts differ between MPI ranks");

  p_->embedded_boundary_opcodes = std::move(staged_opcodes);
  p_->embedded_boundary_literals = std::move(staged_literals);
  p_->embedded_boundary_mode = prepared_mode;
  p_->embedded_boundary_thresholds = thresholds;
  p_->embedded_boundary_generation = generation;
  p_->embedded_boundary_configuration_contract = std::move(authored.configuration_contract);
  p_->embedded_boundary_semantic_digest = std::move(authored.semantic_digest);
}

template <int Dim>
void AmrSystem<Dim>::set_geometry_mode(const std::string& mode) {
  runtime::system::PreparedEmbeddedBoundaryMode prepared_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  std::uint64_t generation = 0;
  PreparedAmrEbAuthoring authored;
  bool unchanged_inactive = false;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_geometry_mode");
    prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
    if (p_->embedded_boundary_configuration_contract.empty()) {
      if (prepared_mode == runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
        unchanged_inactive = true;
      } else {
        throw std::logic_error("AmrSystem geometry mode requires an analytic level set");
      }
    } else {
      if (p_->prepared_blocks.empty() || p_->engine || p_->prepared_hierarchy)
        throw std::logic_error(
            "AmrSystem geometry mode must be selected on an assembled exact block before build");
      generation = next_amr_eb_generation(p_->embedded_boundary_generation);
      authored =
          prepare_amr_eb_authoring(p_->cfg, p_->prepared_blocks.front(),
                                   p_->embedded_boundary_opcodes, p_->embedded_boundary_literals,
                                   prepared_mode, p_->embedded_boundary_thresholds, generation);
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR embedded-boundary mode selection failed collectively");
  }
  if (all_reduce_min(unchanged_inactive ? 1L : 0L) != all_reduce_max(unchanged_inactive ? 1L : 0L))
    throw std::invalid_argument("AMR embedded-boundary mode state differs between MPI ranks");
  if (unchanged_inactive)
    return;
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-eb-configuration", authored.configuration_contract},
           {"amr-eb-semantic", authored.semantic_digest}}))
    throw std::invalid_argument("AMR embedded-boundary mode contracts differ between MPI ranks");
  p_->embedded_boundary_mode = prepared_mode;
  p_->embedded_boundary_generation = generation;
  p_->embedded_boundary_configuration_contract = std::move(authored.configuration_contract);
  p_->embedded_boundary_semantic_digest = std::move(authored.semantic_digest);
}

template <int Dim>
void AmrSystem<Dim>::refresh_prepared_amr_levels() {
  p_->ensure_engine();
}

template <int Dim>
const MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_block_state(int runtime_block, int level) const {
  p_->ensure_engine();
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    throw std::out_of_range("prepared AMR block state block is out of range");
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR block state level is out of range");
  return p_->block_state(static_cast<std::size_t>(runtime_block), static_cast<std::size_t>(level));
}

template <int Dim>
MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_block_state(int runtime_block, int level) {
  return const_cast<MultiFab<Dim>&>(
      std::as_const(*this).prepared_amr_block_state(runtime_block, level));
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_interface_flux_provider(
    std::string provider_contract,
    std::function<void(runtime::multiblock::InterfaceFluxScheduler<Dim>&)> installer) {
  require_amr_assembling(p_->lifecycle, "install_prepared_amr_interface_flux_provider");
  p_->ensure_engine();
  p_->multiblock_hierarchy->install_interface_flux_provider(
      std::move(provider_contract), prepared_amr_level_geometry(0), std::move(installer));
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_coupling_operator(std::string provider_contract,
                                                            CouplingOperatorView view,
                                                            PreparedCouplingOperator operation) {
  require_amr_assembling(p_->lifecycle, "install_prepared_amr_coupling_operator");
  std::vector<typename Impl::PreparedCouplingInstall> candidate;
  std::string exact;
  std::exception_ptr local_error;
  try {
    if (p_->engine || p_->multiblock_hierarchy)
      throw std::logic_error("AMR coupling providers must be installed before materialization");
    if (provider_contract.empty() || view.label.empty() || !operation ||
        !std::isfinite(view.frequency.constant_mu) || view.frequency.constant_mu < 0.0)
      throw std::invalid_argument(
          "AMR coupling provider requires owner identity, executable, and finite frequency");
    ExactContractBuilder contract;
    contract.text("pops.amr-system.prepared-coupling")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_contract)
        .text(view.label)
        .scalar(view.frequency.constant_mu)
        .scalar(view.frequency.per_cell)
        .sequence(view.conservation.conserved_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
        .sequence(view.conservation.created_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
    contract.sequence(operation.conservation_groups(),
                      [](ExactContractBuilder& group,
                         const runtime::system::PreparedCouplingConservationGroup& conservation) {
                        group.text(conservation.identity)
                            .scalar(conservation.absolute_tolerance)
                            .scalar(conservation.relative_tolerance)
                            .sequence(
                                conservation.members,
                                [](ExactContractBuilder& member,
                                   const runtime::system::PreparedCouplingStateRole& role) {
                                  member.text(role.owner)
                                      .scalar(static_cast<std::uint64_t>(role.canonical_block))
                                      .scalar(std::int32_t{role.component})
                                      .text(role.state_role);
                                });
                      });
    exact = std::move(contract).release();
    candidate = p_->prepared_couplings;
    candidate.push_back({std::move(provider_contract), std::move(view), std::move(operation)});
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR coupling provider staging failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-system-prepared-coupling"), std::string_view(exact)}}))
    throw std::invalid_argument("AMR coupling provider contracts differ between MPI ranks");
  p_->prepared_couplings.swap(candidate);
}

template <int Dim>
const typename AmrSystem<Dim>::ProgramBlockMap& AmrSystem<Dim>::prepared_amr_program_block_map()
    const {
  p_->ensure_engine();
  if (!p_->prepared_program_block_map)
    throw std::logic_error(
        "prepared AMR Program block map requires one exact all-block Program mapping");
  return *p_->prepared_program_block_map;
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_program_flux_expression_budget(
    std::string program_hash, std::vector<PreparedAmrProgramFluxExpressionBlockBudget> blocks) {
  require_amr_assembling(p_->lifecycle, "install_prepared_amr_program_flux_expression_budget");
  p_->program.require_step_installed(
      "AmrSystem::install_prepared_amr_program_flux_expression_budget");
  if (p_->program.artifact_backed_)
    throw std::logic_error(
        "artifact-backed AMR Programs must publish flux-expression budgets from exact DSO "
        "metadata");
  p_->ensure_engine();
  if (!p_->prepared_program_block_map)
    throw std::logic_error(
        "manual AMR Program flux-expression budget requires an exact all-block Program map");
  const bool has_flux_expression = Impl::flux_expression_budget_is_active(blocks);
  auto candidate = p_->prepare_program_flux_expression_budget(
      std::move(program_hash), std::move(blocks), *p_->prepared_program_block_map,
      has_flux_expression, *p_->engine, *p_->multiblock_hierarchy);

  // Both candidates own their storage already. String swap and optional move are the no-throw
  // publication boundary after prepared-lane consensus.
  std::string hash_candidate = candidate.program_hash;
  p_->program.installed_hash_.swap(hash_candidate);
  static_assert(std::is_nothrow_move_assignable_v<decltype(p_->program_flux_expression_budget)>);
  p_->program_flux_expression_budget = std::move(candidate);
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBudget&
AmrSystem<Dim>::prepared_amr_program_flux_expression_budget() const {
  p_->ensure_engine();
  if (!p_->program_flux_expression_budget || !p_->prepared_program_block_map)
    throw std::logic_error("installed AMR Program has no prepared flux-expression budget");
  const auto& budget = *p_->program_flux_expression_budget;
  const auto& block_map = *p_->prepared_program_block_map;
  if (budget.program_hash.empty() || budget.program_hash != p_->program.installed_hash_ ||
      budget.generation != p_->engine->materialization_generation() ||
      budget.exact_contract.empty() ||
      budget.program_block_map.canonical_indices != block_map.canonical_indices ||
      budget.program_block_map.hierarchy_contract != block_map.hierarchy_contract ||
      budget.program_block_map.exact_contract != block_map.exact_contract ||
      budget.blocks.size() != block_map.canonical_indices.size())
    throw std::logic_error(
        "installed AMR Program flux-expression budget is not authentic for the prepared carrier");
  for (const auto& block : budget.blocks) {
    const bool active = block.rhs_basis_bound != 0 || block.coefficient_term_bound != 0;
    if ((active && (block.rhs_basis_bound == 0 || block.coefficient_term_bound == 0)) ||
        block.rhs_basis_bound >
            std::numeric_limits<std::size_t>::max() - block.coefficient_term_bound)
      throw std::logic_error(
          "installed AMR Program flux-expression budget contains an invalid per-block bound");
  }
  return budget;
}

template <int Dim>
std::size_t AmrSystem<Dim>::apply_prepared_amr_program_candidates(
    int level, Real dt, std::span<MultiFab<Dim>* const> program_candidates,
    const runtime::multiblock::BoundaryEvaluationPoint& point,
    runtime::multiblock::InterfaceFluxFragmentPublication* interface_publication) {
  p_->ensure_engine();
  if (level < 0)
    throw std::out_of_range("prepared AMR Program candidate level is out of range");
  return p_->multiblock_hierarchy->apply_program_candidates(
      prepared_amr_program_block_map(), static_cast<std::size_t>(level), dt, program_candidates,
      point, interface_publication);
}

template <int Dim>
void AmrSystem<Dim>::publish_prepared_amr_program_candidates(
    int level, std::span<MultiFab<Dim>* const> program_candidates) {
  p_->ensure_engine();
  const ExecutionLane& lane = p_->multiblock_hierarchy->lane();
  const auto communicator = lane.communicator();
  std::exception_ptr pack_error;
  try {
    if (level < 0 || static_cast<std::size_t>(level) >= p_->multiblock_hierarchy->level_count())
      throw std::out_of_range("prepared AMR Program publication level is out of range");
    if (!p_->prepared_program_block_map ||
        p_->prepared_program_block_map->canonical_indices.size() !=
            p_->multiblock_hierarchy->block_count() ||
        p_->program.block_map_.size() != p_->multiblock_hierarchy->block_count() ||
        program_candidates.size() != p_->multiblock_hierarchy->block_count())
      throw std::invalid_argument(
          "prepared AMR Program publication requires one complete exact block pack");
    if (std::any_of(program_candidates.begin(), program_candidates.end(),
                    [](const MultiFab<Dim>* candidate) { return candidate == nullptr; }))
      throw std::invalid_argument("prepared AMR Program publication pack contains a null state");
  } catch (...) {
    pack_error = std::current_exception();
  }
  if (all_reduce_max(pack_error ? 1L : 0L, communicator) != 0) {
    if (lane.size() == 1 && pack_error)
      std::rethrow_exception(pack_error);
    throw std::runtime_error("prepared AMR Program publication pack failed collectively");
  }

  std::exception_ptr layout_error;
  std::string publication_contract;
  try {
    ExactContractBuilder exact;
    exact.text("pops.amr-system.program-publication-pack")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(p_->multiblock_hierarchy->collective_contract())
        .bytes(p_->prepared_program_block_map->exact_contract)
        .scalar(std::int32_t{level})
        .scalar(static_cast<std::uint64_t>(program_candidates.size()));
    for (std::size_t program = 0; program < program_candidates.size(); ++program) {
      const int runtime_block = p_->program.block_map_[program];
      if (runtime_block < 0 ||
          static_cast<std::size_t>(runtime_block) >= p_->multiblock_hierarchy->block_count() ||
          p_->prepared_program_block_map->canonical_indices[program] !=
              static_cast<std::size_t>(runtime_block))
        throw std::invalid_argument(
            "prepared AMR Program publication map differs from its canonical block owner");
      const MultiFab<Dim>& candidate = *program_candidates[program];
      const MultiFab<Dim>& accepted =
          p_->block_state(static_cast<std::size_t>(runtime_block), static_cast<std::size_t>(level));
      if (!same_field_contract(candidate, accepted))
        throw std::invalid_argument(
            "prepared AMR Program publication candidate differs from its exact block layout");
      for (std::size_t local = 0; local < candidate.local_size(); ++local)
        if (candidate.global_index(local) != accepted.global_index(local) ||
            candidate.fab(local).box() != accepted.fab(local).box() ||
            candidate.fab(local).grown_box() != accepted.fab(local).grown_box())
          throw std::invalid_argument(
              "prepared AMR Program publication candidate differs from local patch ownership");
      exact.scalar(static_cast<std::uint64_t>(program))
          .scalar(std::int32_t{runtime_block})
          .text(p_->blocks[static_cast<std::size_t>(runtime_block)].name)
          .scalar(std::int32_t{candidate.ncomp()});
      for (int axis = 0; axis < Dim; ++axis)
        exact.scalar(candidate.ghosts()[axis]);
    }
    publication_contract = std::move(exact).release();
  } catch (...) {
    layout_error = std::current_exception();
  }
  if (all_reduce_max(layout_error ? 1L : 0L, communicator) != 0) {
    if (lane.size() == 1 && layout_error)
      std::rethrow_exception(layout_error);
    throw std::runtime_error("prepared AMR Program publication layout failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-system-program-publication-pack"), publication_contract}}, lane))
    throw std::invalid_argument("prepared AMR Program publication identities differ between ranks");

  for (std::size_t program = 0; program < program_candidates.size(); ++program) {
    validate_prepared_amr_state_publication_candidate(p_->program.block_map_[program], level,
                                                      *program_candidates[program]);
  }
  p_->multiblock_hierarchy->publish_program_candidates(
      *p_->prepared_program_block_map, static_cast<std::size_t>(level), program_candidates);
  p_->discard_level_evaluations();
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation& AmrSystem<Dim>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  p_->ensure_engine();
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR evaluation level lies outside the live hierarchy");
  return evaluate_prepared_amr_level_at(
      point, p_->engine->hierarchy().state(static_cast<std::size_t>(point.level)));
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_level_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state) {
  return evaluate_prepared_amr_block_level_at(0, point, state);
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_block_level_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    throw std::out_of_range("prepared AMR evaluation block lies outside the registry");
  authenticate_generated_block_point<Dim>("combined", runtime_block,
                                          p_->blocks[static_cast<std::size_t>(runtime_block)].name,
                                          point, p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  std::string point_contract;
  std::exception_ptr point_error;
  long point_failure = 0;
  try {
    ExactContractBuilder contract;
    contract.text("pops.generated-amr-evaluation-point")
        .scalar(std::uint32_t{1})
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    point_contract = std::move(contract).release();
  } catch (...) {
    point_failure = 1;
    point_error = std::current_exception();
  }
  if (all_reduce_max(point_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (point_error)
      std::rethrow_exception(point_error);
    throw std::runtime_error("prepared AMR evaluation point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-evaluation-point"), std::string_view(point_contract)}},
          p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR evaluation points differ between MPI ranks");
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >=
          p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)].size())
    throw std::out_of_range("prepared AMR evaluation level lies outside the live hierarchy");
  const std::size_t level_index = static_cast<std::size_t>(point.level);
  const std::size_t block_index = static_cast<std::size_t>(runtime_block);
  const auto* hierarchy_candidates =
      static_cast<std::size_t>(runtime_block) < p_->program_hierarchy_candidates.size()
          ? p_->program_hierarchy_candidates[static_cast<std::size_t>(runtime_block)]
          : nullptr;
  MultiFab<Dim>& live = p_->block_state(block_index, level_index);
  std::optional<PreparedLevelEvaluation>& candidate =
      p_->prepared_hierarchy
          ->block_evaluation_candidates[static_cast<std::size_t>(runtime_block)][level_index];
  std::optional<PreparedLevelEvaluation>& published =
      p_->prepared_hierarchy
          ->block_evaluations[static_cast<std::size_t>(runtime_block)][level_index];
  if (!candidate || !published)
    throw std::logic_error("prepared AMR evaluation workspaces are unavailable");
  const auto restore_ancestors = [&] {
    std::exception_ptr first_error;
    for (std::size_t ancestor = level_index; ancestor-- > 0;) {
      auto& ancestor_stage = *p_->prepared_hierarchy->block_stage_scratch[block_index][ancestor];
      try {
        restore_exact_field_collectively(ancestor_stage.staged, ancestor_stage.backup,
                                         p_->block_state(block_index, ancestor),
                                         p_->prepared_hierarchy->lane->communicator());
      } catch (...) {
        if (!first_error)
          first_error = std::current_exception();
      }
    }
    if (first_error)
      std::rethrow_exception(first_error);
  };
  if (hierarchy_candidates != nullptr) {
    if (hierarchy_candidates->size() != p_->engine->hierarchy().num_levels())
      throw std::logic_error("prepared AMR Program hierarchy candidate registry is incomplete");
    try {
      for (std::size_t ancestor = 0; ancestor < level_index; ++ancestor) {
        MultiFab<Dim>& ancestor_live = p_->block_state(block_index, ancestor);
        auto& scratch = *p_->prepared_hierarchy->block_stage_scratch[block_index][ancestor];
        scratch.staged = stage_exact_field_collectively(
            hierarchy_candidates->at(ancestor), ancestor_live, scratch.backup,
            p_->prepared_hierarchy->lane->communicator());
      }
    } catch (...) {
      restore_ancestors();
      throw;
    }
  }
  auto& stage = *p_->prepared_hierarchy->block_stage_scratch[block_index][level_index];
  try {
    stage.staged = stage_exact_field_collectively(state, live, stage.backup,
                                                  p_->prepared_hierarchy->lane->communicator());
  } catch (...) {
    restore_ancestors();
    throw;
  }
  std::exception_ptr evaluation_error;
  long evaluation_failure = 0;
  try {
    p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)][level_index]
        .evaluate(point, live, *candidate);
  } catch (...) {
    evaluation_failure = 1;
    evaluation_error = std::current_exception();
  }
  std::exception_ptr restoration_error;
  try {
    restore_exact_field_collectively(stage.staged, stage.backup, live,
                                     p_->prepared_hierarchy->lane->communicator());
  } catch (...) {
    restoration_error = std::current_exception();
  }
  try {
    restore_ancestors();
  } catch (...) {
    if (!restoration_error)
      restoration_error = std::current_exception();
  }
  if (restoration_error)
    std::rethrow_exception(restoration_error);
  if (all_reduce_max(evaluation_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (evaluation_error)
      std::rethrow_exception(evaluation_error);
    throw std::runtime_error("prepared AMR level evaluation failed collectively");
  }
  std::swap(*published, *candidate);
  p_->prepared_hierarchy
      ->block_evaluation_published[static_cast<std::size_t>(runtime_block)][level_index] = true;
  return *published;
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_block_level_flux_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR flux evaluation target is out of range");
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  const std::size_t level = static_cast<std::size_t>(point.level);
  authenticate_generated_block_point<Dim>("flux", runtime_block, p_->blocks[block].name, point,
                                          p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  MultiFab<Dim>& live = p_->block_state(block, level);
  auto& candidate = p_->prepared_hierarchy->block_evaluation_candidates[block][level];
  auto& published = p_->prepared_hierarchy->block_evaluations[block][level];
  if (!candidate || !published)
    throw std::logic_error("prepared AMR flux workspaces are unavailable");
  auto& stage = *p_->prepared_hierarchy->block_stage_scratch[block][level];
  stage.staged = stage_exact_field_collectively(state, live, stage.backup,
                                                p_->prepared_hierarchy->lane->communicator());
  std::exception_ptr local_error;
  try {
    p_->prepared_hierarchy->block_levels[block][level].evaluate_flux(point, live, *candidate);
  } catch (...) {
    local_error = std::current_exception();
  }
  restore_exact_field_collectively(stage.staged, stage.backup, live,
                                   p_->prepared_hierarchy->lane->communicator());
  if (all_reduce_max(local_error ? 1L : 0L, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared AMR flux evaluation failed collectively");
  }
  std::swap(*published, *candidate);
  p_->prepared_hierarchy->block_evaluation_published[block][level] = true;
  return *published;
}

template <int Dim>
bool AmrSystem<Dim>::requires_prepared_amr_block_boundary_session(int runtime_block) const {
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    return false;
  return p_->boundary_registry.find_boundary(
             p_->blocks[static_cast<std::size_t>(runtime_block)].name) != nullptr;
}

template <int Dim>
bool AmrSystem<Dim>::has_prepared_amr_block_boundary_linearization(int runtime_block) const {
  if (!requires_prepared_amr_block_boundary_session(runtime_block))
    return false;
  const auto found = p_->prepared_boundary_components.find(
      p_->blocks[static_cast<std::size_t>(runtime_block)].name);
  if (found == p_->prepared_boundary_components.end())
    return true;
  return std::all_of(found->second.fields.begin(), found->second.fields.end(),
                     [](const auto& row) { return row.second.residual && row.second.jvp; });
}

template <int Dim, class Evaluate>
void evaluate_amr_prepared_boundary_candidate(
    typename AmrSystem<Dim>::PreparedLevelEvaluation& evaluation, MultiFab<Dim>& state,
    MultiFab<Dim>& result, MultiFab<Dim>& live, MultiFab<Dim>& backup, const ExecutionLane& lane,
    Evaluate&& evaluate, const char* operation) {
  bool staged = stage_exact_field_collectively(state, live, backup, lane.communicator());
  std::exception_ptr local_error;
  try {
    runtime::program::collective_boundary_provider_phase(lane, operation,
                                                         [&] { evaluate(live, evaluation); });
  } catch (...) {
    local_error = std::current_exception();
  }
  restore_exact_field_collectively(staged, backup, live, lane.communicator());
  if (local_error)
    std::rethrow_exception(local_error);
  generated_system_detail::copy_valid(evaluation.residual, result);
}

template <int Dim>
void AmrSystem<Dim>::prepared_amr_block_level_rhs_core_into_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, MultiFab<Dim>& result, bool flux_only) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  const std::size_t level = static_cast<std::size_t>(point.level);
  if (runtime_block < 0 || block >= p_->blocks.size() || point.level < 0 ||
      level >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR core evaluation target is out of range");
  authenticate_generated_block_point<Dim>("core", runtime_block, p_->blocks[block].name, point,
                                          p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  auto& prepared = p_->prepared_hierarchy->block_levels[block][level];
  MultiFab<Dim>& live = p_->block_state(block, level);
  auto& evaluation = p_->prepared_hierarchy->block_evaluation_candidates[block][level];
  if (!evaluation)
    throw std::logic_error("prepared AMR core evaluation workspace is unavailable");
  evaluate_amr_prepared_boundary_candidate<Dim>(
      *evaluation, state, result, live,
      p_->prepared_hierarchy->block_stage_scratch[block][level]->backup,
      *p_->prepared_hierarchy->lane,
      [&](MultiFab<Dim>& image, PreparedLevelEvaluation& output) {
        prepared.evaluate_core(point, image, flux_only, output);
      },
      "prepared AMR core evaluation");
}

template <int Dim>
void AmrSystem<Dim>::prepared_amr_block_level_boundary_residual_into_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, MultiFab<Dim>& result) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  const std::size_t level = static_cast<std::size_t>(point.level);
  if (!has_prepared_amr_block_boundary_linearization(runtime_block) || point.level < 0 ||
      block >= p_->blocks.size() || level >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR boundary residual target is unavailable");
  authenticate_generated_block_point<Dim>("boundary-residual", runtime_block,
                                          p_->blocks[block].name, point,
                                          p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  auto& prepared = p_->prepared_hierarchy->block_levels[block][level];
  MultiFab<Dim>& live = p_->block_state(block, level);
  auto& evaluation = p_->prepared_hierarchy->block_evaluation_candidates[block][level];
  if (!evaluation)
    throw std::logic_error("prepared AMR boundary evaluation workspace is unavailable");
  evaluate_amr_prepared_boundary_candidate<Dim>(
      *evaluation, state, result, live,
      p_->prepared_hierarchy->block_stage_scratch[block][level]->backup,
      *p_->prepared_hierarchy->lane,
      [&](MultiFab<Dim>& image, PreparedLevelEvaluation& output) {
        prepared.evaluate_boundary(point, image, output);
      },
      "prepared AMR boundary residual");
}

template <int Dim>
void AmrSystem<Dim>::prepared_amr_block_level_boundary_jvp_into_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, const MultiFab<Dim>& direction, MultiFab<Dim>& result) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  const std::size_t level = static_cast<std::size_t>(point.level);
  if (!has_prepared_amr_block_boundary_linearization(runtime_block) || point.level < 0 ||
      block >= p_->blocks.size() || level >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR boundary JVP target is unavailable");
  authenticate_generated_block_point<Dim>("boundary-jvp", runtime_block, p_->blocks[block].name,
                                          point, p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  auto& prepared = p_->prepared_hierarchy->block_levels[block][level];
  MultiFab<Dim>& live = p_->block_state(block, level);
  auto& stage = *p_->prepared_hierarchy->block_stage_scratch[block][level];
  stage.staged = stage_exact_field_collectively(state, live, stage.backup,
                                                p_->prepared_hierarchy->lane->communicator());
  std::exception_ptr local_error;
  try {
    runtime::program::collective_boundary_provider_phase(
        *p_->prepared_hierarchy->lane, "prepared AMR boundary JVP failed collectively", [&] {
          copy_full_field_in_place(result, stage.candidate);
          prepared.boundary_jvp(point, live, direction, stage.candidate);
        });
  } catch (...) {
    local_error = std::current_exception();
  }
  restore_exact_field_collectively(stage.staged, stage.backup, live,
                                   p_->prepared_hierarchy->lane->communicator());
  if (local_error)
    std::rethrow_exception(local_error);
  copy_full_field_in_place(stage.candidate, result);
}

template <int Dim>
void AmrSystem<Dim>::prepared_amr_block_level_source_into_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR source evaluation target is out of range");
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  const std::size_t level = static_cast<std::size_t>(point.level);
  authenticate_generated_block_point<Dim>("source", runtime_block, p_->blocks[block].name, point,
                                          p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());
  MultiFab<Dim>& live = p_->block_state(block, level);
  auto& stage = *p_->prepared_hierarchy->block_stage_scratch[block][level];
  stage.staged = stage_exact_field_collectively(state, live, stage.backup,
                                                p_->prepared_hierarchy->lane->communicator());
  std::exception_ptr local_error;
  try {
    copy_full_field_in_place(rhs, stage.candidate);
    p_->prepared_hierarchy->block_levels[block][level].source_into(point, live, stage.candidate);
    Kokkos::fence();
  } catch (...) {
    local_error = std::current_exception();
  }
  restore_exact_field_collectively(stage.staged, stage.backup, live,
                                   p_->prepared_hierarchy->lane->communicator());
  if (all_reduce_max(local_error ? 1L : 0L, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared AMR source evaluation failed collectively");
  }
  copy_full_field_in_place(stage.candidate, rhs);
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_prepared_amr_block_level_source_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, Real dt, const NewtonOptions& options) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  const ExecutionLane& lane = p_->multiblock_hierarchy->lane();
  const auto communicator = lane.communicator();
  std::exception_ptr local_error;
  std::size_t block = 0;
  std::size_t level = 0;
  try {
    if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
        point.level < 0 ||
        static_cast<std::size_t>(point.level) >= p_->engine->hierarchy().num_levels())
      throw std::out_of_range("prepared AMR implicit-source target is out of range");
    if (!std::isfinite(static_cast<double>(dt)) || !(dt > Real(0)) ||
        dt != static_cast<Real>(point.dt))
      throw std::invalid_argument(
          "prepared AMR implicit-source dt must match its finite positive evaluation point");
    block = static_cast<std::size_t>(runtime_block);
    level = static_cast<std::size_t>(point.level);
    if (!same_field_contract(state, p_->block_state(block, level)))
      throw std::invalid_argument(
          "prepared AMR implicit-source state differs from its exact block/level carrier");
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared AMR implicit-source preflight failed collectively");
  }
  authenticate_generated_block_point<Dim>("implicit-source", runtime_block, p_->blocks[block].name,
                                          point, p_->multiblock_hierarchy->collective_contract(),
                                          communicator);
  // The generated solver targets this detached Program candidate directly. Its returned
  // lane-qualified SolveOutcome remains the sole owner of acceptance publication and consensus.
  return p_->prepared_hierarchy->block_levels[block][level].solve_implicit_source(point, state, dt,
                                                                                  options);
}

template <int Dim>
void AmrSystem<Dim>::prepare_generated_amr_level_state(
    const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state) {
  prepare_generated_amr_block_level_state(0, point, state);
}

template <int Dim>
void AmrSystem<Dim>::prepare_generated_amr_block_level_state(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    throw std::out_of_range("prepared AMR state-preparation block is out of range");
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >=
          p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)].size())
    throw std::out_of_range("prepared AMR state-preparation level lies outside the live hierarchy");

  authenticate_generated_block_point<Dim>("prepare", runtime_block,
                                          p_->blocks[static_cast<std::size_t>(runtime_block)].name,
                                          point, p_->multiblock_hierarchy->collective_contract(),
                                          p_->prepared_hierarchy->lane->communicator());

  std::string point_contract;
  std::exception_ptr point_error;
  long point_failure = 0;
  try {
    ExactContractBuilder contract;
    contract.text("pops.generated-amr-state-preparation-point")
        .scalar(std::uint32_t{1})
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    point_contract = std::move(contract).release();
  } catch (...) {
    point_failure = 1;
    point_error = std::current_exception();
  }
  if (all_reduce_max(point_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (point_error)
      std::rethrow_exception(point_error);
    throw std::runtime_error("prepared AMR state-preparation point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-state-preparation-point"),
            std::string_view(point_contract)}},
          p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR state-preparation points differ between MPI ranks");

  const std::size_t level_index = static_cast<std::size_t>(point.level);
  const std::size_t block_index = static_cast<std::size_t>(runtime_block);
  const auto* hierarchy_candidates =
      static_cast<std::size_t>(runtime_block) < p_->program_hierarchy_candidates.size()
          ? p_->program_hierarchy_candidates[static_cast<std::size_t>(runtime_block)]
          : nullptr;
  MultiFab<Dim>& live = p_->block_state(static_cast<std::size_t>(runtime_block), level_index);
  const long staged = &state == &live ? 0L : 1L;
  if (all_reduce_min(staged, p_->prepared_hierarchy->lane->communicator()) !=
      all_reduce_max(staged, p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR state-preparation target differs between MPI ranks");
  const auto restore_ancestors = [&] {
    std::exception_ptr first_error;
    for (std::size_t ancestor = level_index; ancestor-- > 0;) {
      auto& ancestor_stage = *p_->prepared_hierarchy->block_stage_scratch[block_index][ancestor];
      try {
        restore_exact_field_collectively(ancestor_stage.staged, ancestor_stage.backup,
                                         p_->block_state(block_index, ancestor),
                                         p_->prepared_hierarchy->lane->communicator());
      } catch (...) {
        if (!first_error)
          first_error = std::current_exception();
      }
    }
    if (first_error)
      std::rethrow_exception(first_error);
  };
  if (hierarchy_candidates != nullptr) {
    if (hierarchy_candidates->size() != p_->engine->hierarchy().num_levels())
      throw std::logic_error("prepared AMR Program hierarchy candidate registry is incomplete");
    try {
      for (std::size_t ancestor = 0; ancestor < level_index; ++ancestor) {
        MultiFab<Dim>& ancestor_live = p_->block_state(block_index, ancestor);
        auto& ancestor_stage = *p_->prepared_hierarchy->block_stage_scratch[block_index][ancestor];
        ancestor_stage.staged = stage_exact_field_collectively(
            hierarchy_candidates->at(ancestor), ancestor_live, ancestor_stage.backup,
            p_->prepared_hierarchy->lane->communicator());
      }
    } catch (...) {
      restore_ancestors();
      throw;
    }
  }

  auto& level_stage = *p_->prepared_hierarchy->block_stage_scratch[block_index][level_index];
  try {
    level_stage.staged = stage_exact_field_collectively(
        state, live, level_stage.backup, p_->prepared_hierarchy->lane->communicator());
  } catch (...) {
    restore_ancestors();
    throw;
  }
  std::exception_ptr preparation_error;
  long preparation_failure = 0;
  try {
    p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)][level_index]
        .prepare(point, live);
    if (staged != 0)
      copy_full_field_in_place(live, level_stage.candidate);
  } catch (...) {
    preparation_failure = 1;
    preparation_error = std::current_exception();
  }
  std::exception_ptr restoration_error;
  try {
    restore_exact_field_collectively(level_stage.staged, level_stage.backup, live,
                                     p_->prepared_hierarchy->lane->communicator());
  } catch (...) {
    restoration_error = std::current_exception();
  }
  try {
    restore_ancestors();
  } catch (...) {
    if (!restoration_error)
      restoration_error = std::current_exception();
  }
  if (restoration_error)
    std::rethrow_exception(restoration_error);
  if (all_reduce_max(preparation_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("prepared AMR state preparation failed collectively");
  }
  if (staged != 0)
    copy_full_field_in_place(level_stage.candidate, state);
}

template <int Dim>
void AmrSystem<Dim>::prepare_generated_amr_block_level_state(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (point.level == 0) {
    if (parent_level != -1 || staged_parent != nullptr)
      throw std::invalid_argument("root AMR provider call cannot bind a staged parent");
    prepare_generated_amr_block_level_state(runtime_block, point, state);
    return;
  }
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      parent_level < 0 ||
      static_cast<std::size_t>(parent_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("subcycled AMR state-preparation parent target is out of range");
  if (p_->program_hierarchy_candidates.size() != p_->blocks.size())
    p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  invoke_with_staged_parent<Dim>(
      runtime_block, p_->blocks[block].name, point.level, parent_level, staged_parent,
      p_->block_state(block, static_cast<std::size_t>(parent_level)),
      p_->multiblock_hierarchy->collective_contract(), p_->prepared_hierarchy->lane->communicator(),
      p_->program_hierarchy_candidates[block],
      p_->prepared_hierarchy->block_stage_scratch[block][static_cast<std::size_t>(parent_level)]
          ->backup,
      [&] { prepare_generated_amr_block_level_state(runtime_block, point, state); });
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_block_level_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (point.level == 0) {
    if (parent_level != -1 || staged_parent != nullptr)
      throw std::invalid_argument("root AMR provider call cannot bind a staged parent");
    return evaluate_prepared_amr_block_level_at(runtime_block, point, state);
  }
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      parent_level < 0 ||
      static_cast<std::size_t>(parent_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("subcycled AMR evaluation parent target is out of range");
  if (p_->program_hierarchy_candidates.size() != p_->blocks.size())
    p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  return invoke_with_staged_parent<Dim>(
      runtime_block, p_->blocks[block].name, point.level, parent_level, staged_parent,
      p_->block_state(block, static_cast<std::size_t>(parent_level)),
      p_->multiblock_hierarchy->collective_contract(), p_->prepared_hierarchy->lane->communicator(),
      p_->program_hierarchy_candidates[block],
      p_->prepared_hierarchy->block_stage_scratch[block][static_cast<std::size_t>(parent_level)]
          ->backup,
      [&]() -> const PreparedLevelEvaluation& {
        return evaluate_prepared_amr_block_level_at(runtime_block, point, state);
      });
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_block_level_flux_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (point.level == 0) {
    if (parent_level != -1 || staged_parent != nullptr)
      throw std::invalid_argument("root AMR flux call cannot bind a staged parent");
    return evaluate_prepared_amr_block_level_flux_at(runtime_block, point, state);
  }
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      parent_level < 0 ||
      static_cast<std::size_t>(parent_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("subcycled AMR flux parent target is out of range");
  if (p_->program_hierarchy_candidates.size() != p_->blocks.size())
    p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  return invoke_with_staged_parent<Dim>(
      runtime_block, p_->blocks[block].name, point.level, parent_level, staged_parent,
      p_->block_state(block, static_cast<std::size_t>(parent_level)),
      p_->multiblock_hierarchy->collective_contract(), p_->prepared_hierarchy->lane->communicator(),
      p_->program_hierarchy_candidates[block],
      p_->prepared_hierarchy->block_stage_scratch[block][static_cast<std::size_t>(parent_level)]
          ->backup,
      [&]() -> const PreparedLevelEvaluation& {
        return evaluate_prepared_amr_block_level_flux_at(runtime_block, point, state);
      });
}

template <int Dim>
void AmrSystem<Dim>::prepared_amr_block_level_source_into_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, MultiFab<Dim>& rhs, int parent_level,
    const MultiFab<Dim>* staged_parent) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (point.level == 0) {
    if (parent_level != -1 || staged_parent != nullptr)
      throw std::invalid_argument("root AMR source call cannot bind a staged parent");
    prepared_amr_block_level_source_into_at(runtime_block, point, state, rhs);
    return;
  }
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      parent_level < 0 ||
      static_cast<std::size_t>(parent_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("subcycled AMR source parent target is out of range");
  if (p_->program_hierarchy_candidates.size() != p_->blocks.size())
    p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  invoke_with_staged_parent<Dim>(
      runtime_block, p_->blocks[block].name, point.level, parent_level, staged_parent,
      p_->block_state(block, static_cast<std::size_t>(parent_level)),
      p_->multiblock_hierarchy->collective_contract(), p_->prepared_hierarchy->lane->communicator(),
      p_->program_hierarchy_candidates[block],
      p_->prepared_hierarchy->block_stage_scratch[block][static_cast<std::size_t>(parent_level)]
          ->backup,
      [&] { prepared_amr_block_level_source_into_at(runtime_block, point, state, rhs); });
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_prepared_amr_block_level_source_at(
    int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab<Dim>& state, Real dt, const NewtonOptions& options, int parent_level,
    const MultiFab<Dim>* staged_parent) {
  p_->ensure_engine();
  std::lock_guard execution_lock(p_->prepared_hierarchy->execution_mutex);
  if (point.level == 0) {
    if (parent_level != -1 || staged_parent != nullptr)
      throw std::invalid_argument("root AMR implicit-source call cannot bind a staged parent");
    return solve_prepared_amr_block_level_source_at(runtime_block, point, state, dt, options);
  }
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size() ||
      parent_level < 0 ||
      static_cast<std::size_t>(parent_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("subcycled AMR implicit-source parent target is out of range");
  if (p_->program_hierarchy_candidates.size() != p_->blocks.size())
    p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
  const std::size_t block = static_cast<std::size_t>(runtime_block);
  return invoke_with_staged_parent<Dim>(
      runtime_block, p_->blocks[block].name, point.level, parent_level, staged_parent,
      p_->block_state(block, static_cast<std::size_t>(parent_level)),
      p_->multiblock_hierarchy->collective_contract(), p_->prepared_hierarchy->lane->communicator(),
      p_->program_hierarchy_candidates[block],
      p_->prepared_hierarchy->block_stage_scratch[block][static_cast<std::size_t>(parent_level)]
          ->backup,
      [&] {
        return solve_prepared_amr_block_level_source_at(runtime_block, point, state, dt, options);
      });
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::prepared_amr_level_evaluation(int level) const {
  p_->ensure_engine();
  if (level < 0 || p_->prepared_hierarchy->block_evaluations.empty() ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->block_evaluations.front().size())
    throw std::out_of_range("prepared AMR ledger level lies outside the live hierarchy");
  const std::optional<PreparedLevelEvaluation>& evaluation =
      p_->prepared_hierarchy->block_evaluations.front()[static_cast<std::size_t>(level)];
  if (!evaluation ||
      !p_->prepared_hierarchy->block_evaluation_published.front()[static_cast<std::size_t>(level)])
    throw std::logic_error("prepared AMR level has no published residual/flux evaluation");
  return *evaluation;
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation*
AmrSystem<Dim>::prepared_amr_level_evaluation_if_present(int level) const noexcept {
  try {
    if (!p_->prepared_hierarchy || p_->prepared_hierarchy->block_evaluations.empty() || level < 0 ||
        static_cast<std::size_t>(level) >= p_->prepared_hierarchy->block_evaluations.front().size())
      return nullptr;
    const std::optional<PreparedLevelEvaluation>& evaluation =
        p_->prepared_hierarchy->block_evaluations.front()[static_cast<std::size_t>(level)];
    return evaluation && p_->prepared_hierarchy->block_evaluation_published
                             .front()[static_cast<std::size_t>(level)]
               ? &*evaluation
               : nullptr;
  } catch (...) {
    return nullptr;
  }
}

template <int Dim>
void AmrSystem<Dim>::clear_prepared_amr_level_evaluations() const noexcept {
  p_->discard_level_evaluations();
}

template <int Dim>
void AmrSystem<Dim>::bind_program_hierarchy_candidates(
    const std::vector<MultiFab<Dim>>* candidates) const {
  bind_program_block_hierarchy_candidates(0, candidates);
}

template <int Dim>
void AmrSystem<Dim>::unbind_program_hierarchy_candidates(
    const std::vector<MultiFab<Dim>>* candidates) const noexcept {
  unbind_program_block_hierarchy_candidates(0, candidates);
}

template <int Dim>
void AmrSystem<Dim>::bind_program_block_hierarchy_candidates(
    int runtime_block, const std::vector<MultiFab<Dim>>* candidates) const {
  p_->ensure_engine();
  std::exception_ptr validation_error;
  std::string binding_contract;
  try {
    if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
      throw std::out_of_range("AMR Program hierarchy candidate block is out of range");
    if (!p_->program_hierarchy_candidates.empty() &&
        p_->program_hierarchy_candidates.size() != p_->blocks.size())
      throw std::logic_error("AMR Program hierarchy candidate registry is malformed");
    if (p_->program_hierarchy_candidates.empty())
      p_->program_hierarchy_candidates.resize(p_->blocks.size(), nullptr);
    const std::size_t block = static_cast<std::size_t>(runtime_block);
    if (candidates == nullptr || p_->program_hierarchy_candidates[block] != nullptr ||
        candidates->size() != p_->multiblock_hierarchy->level_count())
      throw std::invalid_argument(
          "AMR Program block hierarchy binding requires one unique complete image");

    ExactContractBuilder exact;
    exact.text("pops.amr-system.program-block-hierarchy-candidates")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(p_->multiblock_hierarchy->collective_contract())
        .scalar(std::int32_t{runtime_block})
        .text(p_->blocks[block].name)
        .scalar(static_cast<std::uint64_t>(candidates->size()));
    for (std::size_t level = 0; level < candidates->size(); ++level) {
      const MultiFab<Dim>& live = p_->block_state(block, level);
      const MultiFab<Dim>& candidate = candidates->at(level);
      if (!same_field_contract(candidate, live) || candidate.local_size() != live.local_size())
        throw std::invalid_argument(
            "AMR Program hierarchy candidate differs from its exact live block/level contract");
      for (std::size_t local = 0; local < candidate.local_size(); ++local)
        if (candidate.global_index(local) != live.global_index(local) ||
            candidate.fab(local).box() != live.fab(local).box() ||
            candidate.fab(local).grown_box() != live.fab(local).grown_box())
          throw std::invalid_argument(
              "AMR Program hierarchy candidate differs from exact local patch ownership");
      exact.scalar(static_cast<std::uint64_t>(level)).scalar(std::int32_t{live.ncomp()});
      for (int axis = 0; axis < Dim; ++axis)
        exact.scalar(live.ghosts()[axis]);
    }
    binding_contract = std::move(exact).release();
  } catch (...) {
    validation_error = std::current_exception();
  }
  const CommunicatorView communicator = p_->prepared_hierarchy->lane->communicator();
  if (all_reduce_max(validation_error ? 1L : 0L, communicator) != 0) {
    if (validation_error)
      std::rethrow_exception(validation_error);
    throw std::runtime_error(
        "AMR Program block hierarchy candidate binding failed on another MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-program-block-hierarchy-candidates"), binding_contract}},
          communicator))
    throw std::invalid_argument(
        "AMR Program block hierarchy candidate identities differ between MPI ranks");
  p_->program_hierarchy_candidates[static_cast<std::size_t>(runtime_block)] = candidates;
}

template <int Dim>
void AmrSystem<Dim>::unbind_program_block_hierarchy_candidates(
    int runtime_block, const std::vector<MultiFab<Dim>>* candidates) const noexcept {
  if (runtime_block < 0 ||
      static_cast<std::size_t>(runtime_block) >= p_->program_hierarchy_candidates.size())
    return;
  auto& bound = p_->program_hierarchy_candidates[static_cast<std::size_t>(runtime_block)];
  if (bound == candidates)
    bound = nullptr;
}

template <int Dim>
Geometry<Dim> AmrSystem<Dim>::prepared_amr_level_geometry(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR geometry level lies outside the live hierarchy");
  return Geometry<Dim>::from_bounds(
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), p_->cfg.lower,
      p_->cfg.upper);
}

template <int Dim>
typename AmrSystem<Dim>::PreparedMultiBlockHierarchy&
AmrSystem<Dim>::prepared_amr_multiblock_hierarchy_() {
  p_->ensure_engine();
  if (!p_->multiblock_hierarchy)
    throw std::logic_error("AmrSystem has no prepared multi-block hierarchy carrier");
  return *p_->multiblock_hierarchy;
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedMultiBlockHierarchy&
AmrSystem<Dim>::prepared_amr_multiblock_hierarchy_() const {
  p_->ensure_engine();
  if (!p_->multiblock_hierarchy)
    throw std::logic_error("AmrSystem has no prepared multi-block hierarchy carrier");
  return *p_->multiblock_hierarchy;
}

template <int Dim>
BoundaryTopology<Dim> AmrSystem<Dim>::prepared_amr_boundary_topology() const {
  return p_->topology();
}

template <int Dim>
Real AmrSystem<Dim>::prepared_amr_level_maximum_speed(int level, const MultiFab<Dim>& state) const {
  return prepared_amr_block_level_maximum_speed(0, level, state);
}

template <int Dim>
Real AmrSystem<Dim>::prepared_amr_block_level_maximum_speed(int runtime_block, int level,
                                                            const MultiFab<Dim>& state) const {
  p_->ensure_engine();
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    throw std::out_of_range("prepared AMR speed block is out of range");
  if (level < 0 ||
      static_cast<std::size_t>(level) >=
          p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)].size())
    throw std::out_of_range("prepared AMR speed level lies outside the live hierarchy");
  return p_->prepared_hierarchy
      ->block_levels[static_cast<std::size_t>(runtime_block)][static_cast<std::size_t>(level)]
      .maximum_speed(state);
}

template <int Dim>
void AmrSystem<Dim>::validate_prepared_amr_state_publication_candidate(
    int runtime_block, int level, const MultiFab<Dim>& candidate) const {
  p_->ensure_engine();
  const ExecutionLane& lane = p_->multiblock_hierarchy->lane();
  const auto communicator = lane.communicator();
  if (all_reduce_min(static_cast<long>(runtime_block), communicator) !=
          all_reduce_max(static_cast<long>(runtime_block), communicator) ||
      all_reduce_min(static_cast<long>(level), communicator) !=
          all_reduce_max(static_cast<long>(level), communicator))
    throw std::invalid_argument(
        "AMR Program state publication block/level identities differ between ranks");
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->prepared_blocks.size())
      throw std::out_of_range("AMR Program state publication block is out of range");
    if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
      throw std::out_of_range(
          "AMR Program state publication level lies outside the live hierarchy");

    const MultiFab<Dim>& accepted =
        p_->block_state(static_cast<std::size_t>(runtime_block), static_cast<std::size_t>(level));
    if (!same_field_contract(candidate, accepted))
      throw std::invalid_argument(
          "AMR Program state publication candidate differs from its exact live level contract");
    const PreparedBlock& block = p_->prepared_blocks[static_cast<std::size_t>(runtime_block)];
    if (candidate.ncomp() != block.ncomp)
      throw std::invalid_argument(
          "AMR Program state publication candidate differs from its prepared block width");
    if (!finite_valid_field_local(candidate))
      throw std::invalid_argument("AMR Program state publication candidate has non-finite values");

    std::vector<double> conservative(static_cast<std::size_t>(block.ncomp));
    std::vector<double> primitive(static_cast<std::size_t>(block.ncomp));
    for (std::size_t local = 0; local < candidate.local_size(); ++local) {
      const Fab<Dim>& fab = candidate.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const Box<Dim>& valid = fab.box();
      const Box<Dim>& grown = fab.grown_box();
      const std::size_t component_stride = checked_cells(grown);
      for (std::size_t ordinal = 0; ordinal < checked_cells(valid); ++ordinal) {
        const Index<Dim> cell = unflatten(valid, ordinal);
        for (int component = 0; component < block.ncomp; ++component)
          conservative[static_cast<std::size_t>(component)] = static_cast<double>(
              host(static_cast<std::size_t>(component) * component_stride + offset(cell, grown)));
        const RecoveryReport report =
            block.conservative_to_primitive(conservative.data(), primitive.data());
        if (!report.publication_permitted())
          throw std::runtime_error(
              "AMR Program state publication candidate failed prepared variable recovery");
        if (!std::all_of(primitive.begin(), primitive.end(),
                         [](double value) { return std::isfinite(value); }))
          throw std::runtime_error(
              "AMR Program state publication candidate recovered non-finite primitives");
      }
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure, communicator) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(
        "AMR Program state publication candidate failed prepared recovery collectively");
  }
}

template <int Dim>
void AmrSystem<Dim>::add_prepared_amr_poisson_rhs(int level, MultiFab<Dim>& rhs) {
  p_->ensure_engine();
  for (std::size_t block = 0; block < p_->blocks.size(); ++block)
    add_prepared_amr_block_poisson_rhs(static_cast<int>(block), level,
                                       p_->block_state(block, static_cast<std::size_t>(level)),
                                       rhs);
}

template <int Dim>
void AmrSystem<Dim>::add_prepared_amr_block_poisson_rhs(int runtime_block, int level,
                                                        const MultiFab<Dim>& state,
                                                        MultiFab<Dim>& rhs) {
  p_->ensure_engine();
  if (all_reduce_min(static_cast<long>(level)) != all_reduce_max(static_cast<long>(level)))
    throw std::invalid_argument("prepared AMR Poisson RHS levels differ between MPI ranks");
  if (runtime_block < 0 || static_cast<std::size_t>(runtime_block) >= p_->blocks.size())
    throw std::out_of_range("prepared AMR Poisson RHS block is out of range");
  if (level < 0 ||
      static_cast<std::size_t>(level) >=
          p_->prepared_hierarchy->block_levels[static_cast<std::size_t>(runtime_block)].size())
    throw std::out_of_range("prepared AMR Poisson RHS level lies outside the live hierarchy");
  p_->prepared_hierarchy
      ->block_levels[static_cast<std::size_t>(runtime_block)][static_cast<std::size_t>(level)]
      .add_poisson_rhs(state, rhs);
}

template <int Dim>
void AmrSystem<Dim>::install_block_state_route(const std::string& name,
                                               const std::string& state_identity) {
  require_amr_assembling(p_->lifecycle, "install_block_state_route");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem state routes must be installed before their block");
  p_->boundary_registry.install_state_route(name, state_identity);
}

template <int Dim>
void AmrSystem<Dim>::install_field_storage_route(const std::string& field_identity,
                                                 const std::string& provider_slot) {
  require_amr_assembling(p_->lifecycle, "install_field_storage_route");
  p_->boundary_registry.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_boundary_execution_context(
    std::shared_ptr<const component::PreparedExecutionContextV1> execution) {
  require_amr_assembling(p_->lifecycle, "install_prepared_boundary_execution_context");
  if (!execution || p_->engine || p_->prepared_hierarchy)
    throw std::invalid_argument(
        "AMR boundary execution context must precede hierarchy materialization");
  if (p_->boundary_execution_context)
    throw std::logic_error("AMR boundary execution context is already installed");
  p_->boundary_execution_context = std::move(execution);
}

template <int Dim>
void AmrSystem<Dim>::stage_prepared_ghost_boundary_component(
    const std::string& block, std::shared_ptr<PreparedGhostBoundaryComponent> component) {
  require_amr_assembling(p_->lifecycle, "stage_prepared_ghost_boundary_component");
  if (block.empty() || !component || component->spec().region.dimension != Dim ||
      component->spec().state_identity != p_->boundary_registry.state_route(block))
    throw std::invalid_argument("AMR GhostBoundary differs from its exact block/rank route");
  auto candidate = p_->prepared_boundary_components;
  auto& providers = candidate[block].ghosts;
  providers.push_back(std::move(component));
  canonicalize_prepared_boundary_chain(providers, "AMR GhostBoundary");
  p_->prepared_boundary_components.swap(candidate);
}

template <int Dim>
void AmrSystem<Dim>::stage_prepared_boundary_flux_component(
    const std::string& block, std::shared_ptr<PreparedBoundaryFluxComponent> component) {
  require_amr_assembling(p_->lifecycle, "stage_prepared_boundary_flux_component");
  if (block.empty() || !component || component->spec().region.dimension != Dim ||
      component->spec().state_identity != p_->boundary_registry.state_route(block))
    throw std::invalid_argument("AMR BoundaryFlux differs from its exact block/rank route");
  auto candidate = p_->prepared_boundary_components;
  auto& providers = candidate[block].fluxes;
  providers.push_back(std::move(component));
  canonicalize_prepared_boundary_chain(providers, "AMR BoundaryFlux");
  p_->prepared_boundary_components.swap(candidate);
}

namespace {

std::string prepared_field_boundary_pair_key(const PreparedBoundaryComponentSpec& spec) {
  return spec.producer_identity + "\n" + spec.region.identity;
}

void require_prepared_field_boundary_pair(const PreparedBoundaryComponentSpec& residual,
                                          const PreparedBoundaryComponentSpec& jvp) {
  if (residual.target_identity != jvp.target_identity || residual.target_json != jvp.target_json ||
      residual.component_id != jvp.component_id ||
      residual.manifest_identity != jvp.manifest_identity ||
      residual.interface_version != jvp.interface_version ||
      residual.producer_identity != jvp.producer_identity ||
      residual.state_identity != jvp.state_identity ||
      residual.ghost_identity != jvp.ghost_identity ||
      residual.layout_identity != jvp.layout_identity ||
      residual.region.identity != jvp.region.identity || residual.region.axes != jvp.region.axes ||
      residual.region.sides != jvp.region.sides || residual.region.kind != jvp.region.kind ||
      residual.region.codimension != jvp.region.codimension || residual.states != jvp.states ||
      residual.fields != jvp.fields || residual.parameter_ids != jvp.parameter_ids ||
      residual.parameter_values != jvp.parameter_values ||
      residual.parameters_json != jvp.parameters_json || residual.rate != jvp.rate ||
      residual.nonlinear_iterate != jvp.nonlinear_iterate || !residual.directions.empty() ||
      jvp.directions != std::vector<std::string>{residual.state_identity} ||
      residual.outputs.size() != 1 || jvp.outputs.size() != 1 ||
      residual.outputs.front() == jvp.outputs.front())
    throw std::invalid_argument(
        "AMR FieldBoundaryClosure residual/JVP pair changed its exact provider contract");
}

}  // namespace

template <int Dim>
void AmrSystem<Dim>::stage_prepared_field_boundary_component_pair(
    const std::string& block, std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual,
    std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp) {
  require_amr_assembling(p_->lifecycle, "stage_prepared_field_boundary_component_pair");
  if (block.empty() || !residual || !jvp || residual->spec().region.dimension != Dim ||
      jvp->spec().region.dimension != Dim ||
      residual->spec().state_identity != p_->boundary_registry.state_route(block) ||
      jvp->spec().state_identity != p_->boundary_registry.state_route(block))
    throw std::invalid_argument(
        "AMR FieldBoundaryClosure pair differs from its exact block/rank route");
  require_prepared_field_boundary_pair(residual->spec(), jvp->spec());
  if (residual->package_owner_identity() != jvp->package_owner_identity() ||
      !residual->spec().execution->equivalent_to(*jvp->spec().execution))
    throw std::invalid_argument(
        "AMR FieldBoundaryClosure pair changed its exact package/execution authority");
  auto candidate = p_->prepared_boundary_components;
  auto& pair = candidate[block].fields[prepared_field_boundary_pair_key(residual->spec())];
  if (pair.residual || pair.jvp)
    throw std::logic_error("AMR FieldBoundaryClosure pair route is already installed");
  pair.residual = std::move(residual);
  pair.jvp = std::move(jvp);
  p_->prepared_boundary_components.swap(candidate);
}

template <int Dim>
void AmrSystem<Dim>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_field_nullspace_provider");
  if (!p_->field_nullspace_providers)
    throw std::logic_error("AmrSystem field-nullspace registry is absent");
  p_->field_nullspace_providers->add(std::move(provider));
}

template <int Dim>
void AmrSystem<Dim>::register_field_solver_provider(
    std::shared_ptr<const runtime::amr::ExactAmrFieldSolverProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_field_solver_provider");
  if (!p_->field_solver_providers)
    throw std::logic_error("AmrSystem exact field-solver registry is absent");
  p_->field_solver_providers->add(std::move(provider));
}

template <int Dim>
std::string AmrSystem<Dim>::register_field_solver_provider(
    const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec,
    std::shared_ptr<component::LoadedComponent>, std::shared_ptr<component::LoadedComponent>) {
  require_amr_assembling(p_->lifecycle, "register_field_solver_provider");
  if (provider_slot.empty())
    throw std::invalid_argument("AMR component field provider slot must be non-empty");
  throw std::logic_error(
      "AMR component FieldTopology/FieldSolver pairs require an exact-ranked hierarchy provider");
}

template <int Dim>
void AmrSystem<Dim>::set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                                 const PreparedProviderOptions& options) {
  require_amr_assembling(p_->lifecycle, "set_default_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("AMR default field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  p_->default_nullspace_provider_identity = nullspace_provider_identity;
  p_->default_nullspace_options = options;
}

template <int Dim>
void AmrSystem<Dim>::register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_hierarchy_tensor_solver_provider");
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  if (!p_->multiblock_hierarchy)
    throw std::logic_error(
        "AmrSystem hierarchy tensor-solver registration requires its prepared hierarchy");
  p_->hierarchy_tensor_solver_providers->add(std::move(provider), p_->multiblock_hierarchy->lane());
}

template <int Dim>
void AmrSystem<Dim>::register_program_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider) {
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  if (!p_->multiblock_hierarchy)
    throw std::logic_error(
        "AMR Program hierarchy tensor-solver registration requires its prepared hierarchy");
  p_->hierarchy_tensor_solver_providers->add(std::move(provider), p_->multiblock_hierarchy->lane());
}

template <int Dim>
std::shared_ptr<const runtime::program::HierarchyTensorSolverProviderRegistry<Dim>>
AmrSystem<Dim>::hierarchy_tensor_solver_provider_registry() const {
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  return p_->hierarchy_tensor_solver_providers;
}

template <int Dim>
void AmrSystem<Dim>::set_poisson(const std::string& rhs, const std::string& solver,
                                 const std::string& bc,
                                 const AmrFieldSolverOptions& solver_options) {
  require_amr_assembling(p_->lifecycle, "set_poisson");
  if (rhs != "charge_density" && rhs != "composite")
    throw std::invalid_argument(
        "AMR exact Poisson supports charge_density or composite RHS routes");
  if (solver != "geometric_mg")
    throw std::invalid_argument("AMR exact Poisson requires the geometric_mg provider");
  if (bc != "auto" && bc != "periodic" && bc != "dirichlet" && bc != "neumann")
    throw std::invalid_argument("AMR exact Poisson boundary mode is unknown");
  if (!p_->default_field_slot.empty())
    throw std::logic_error("AMR default field is already configured");
  if (bc == "periodic" && std::any_of(p_->cfg.periodicity.begin(), p_->cfg.periodicity.end(),
                                      [](bool periodic) { return !periodic; }))
    throw std::invalid_argument(
        "AMR periodic Poisson requires every exact topology axis to be periodic");

  typename Impl::FieldPlan plan;
  plan.plan_identity = "pops.amr.default-field-plan";
  plan.provider_identity = "pops.amr.default-field";
  plan.output_owner_identity = "pops.amr.default-field";
  plan.output_block = "amr";
  plan.output_key = "fields_from_state";
  plan.solver_route = solver;
  plan.hierarchy_policy = {
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  plan.solver_options =
      solver_options.schema_identity.empty()
          ? geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{})
          : solver_options;
  (void)plan.solver_options.exact_contract();
  plan.use_prepared_level_rhs = true;
  if (bc != "auto") {
    const std::size_t faces = static_cast<std::size_t>(2 * Dim);
    plan.boundary_kind.assign(faces, bc);
    plan.boundary_alpha.assign(faces, bc == "dirichlet" ? 1.0 : 0.0);
    plan.boundary_beta.assign(faces, bc == "neumann" ? 1.0 : 0.0);
    plan.boundary_value.assign(faces, 0.0);
  }
  const std::string slot = "pops.amr.default-field";
  p_->field_plans.emplace(slot, std::move(plan));
  p_->default_field_slot = slot;
}

template <int Dim>
void AmrSystem<Dim>::set_field_solver_plan(
    const std::string& provider_slot, const std::string& plan_identity,
    const std::string& provider_identity, const std::string& output_owner_identity,
    const std::string& output_block, const std::string& output_key,
    const std::vector<std::string>& provider_identities,
    const std::vector<std::string>& provider_blocks, const std::vector<std::string>& provider_keys,
    const std::vector<double>& provider_coefficients, const std::string& solver,
    const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
    const AmrFieldSolverOptions& solver_options) {
  require_amr_assembling(p_->lifecycle, "set_field_solver_plan");
  const std::size_t count = provider_identities.size();
  if (provider_slot.empty() || plan_identity.empty() || provider_identity.empty() ||
      output_owner_identity.empty() || output_block.empty() || output_key.empty() ||
      solver.empty() || count == 0 || provider_blocks.size() != count ||
      provider_keys.size() != count || provider_coefficients.size() != count)
    throw std::invalid_argument("AMR exact field solver plan is incomplete");
  if (p_->field_plans.contains(provider_slot))
    throw std::runtime_error("AMR exact field provider slot is already installed: " +
                             provider_slot);
  for (const auto& [slot, existing] : p_->field_plans) {
    (void)slot;
    if (existing.output_block == output_block && existing.output_key == output_key)
      throw std::runtime_error("AMR exact field output block/key is already owned by another slot");
  }
  hierarchy_policy.validate();
  (void)solver_options.exact_contract();
  typename Impl::FieldPlan plan;
  plan.plan_identity = plan_identity;
  plan.provider_identity = provider_identity;
  plan.output_owner_identity = output_owner_identity;
  plan.output_block = output_block;
  plan.output_key = output_key;
  plan.solver_route = solver;
  plan.hierarchy_policy = hierarchy_policy;
  plan.solver_options = solver_options;
  plan.providers.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (provider_identities[index].empty() || provider_blocks[index].empty() ||
        provider_keys[index].empty() || !std::isfinite(provider_coefficients[index]))
      throw std::invalid_argument("AMR exact field provider binding is incomplete");
    plan.providers.push_back({provider_identities[index], provider_blocks[index],
                              provider_keys[index], provider_coefficients[index]});
  }
  plan.rhs_by_block.resize(p_->blocks.size());
  p_->field_plans.emplace(provider_slot, std::move(plan));
}

template <int Dim>
AmrFieldSolverConfiguration AmrSystem<Dim>::field_solver_configuration(
    const std::string& provider_slot) const {
  const auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("unknown exact AMR field provider slot");
  const auto& plan = found->second;
  return {plan.plan_identity, plan.provider_identity, plan.solver_route, plan.hierarchy_policy,
          plan.solver_options};
}

template <int Dim>
void AmrSystem<Dim>::set_field_reaction(const std::string& provider_slot, double reaction) {
  require_amr_assembling(p_->lifecycle, "set_field_reaction");
  if (!std::isfinite(reaction) || reaction <= 0.0)
    throw std::invalid_argument("AMR screened field reaction must be finite and strictly positive");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field reaction names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field reaction cannot change after solver materialization");
  found->second.reaction = reaction;
  found->second.has_reaction = true;
}

template <int Dim>
void AmrSystem<Dim>::set_field_topology_authority(const std::string& provider_slot,
                                                  const std::string& provider_kind,
                                                  const std::string& provenance,
                                                  const std::string& topology_digest) {
  require_amr_assembling(p_->lifecycle, "set_field_topology_authority");
  if (provider_kind.empty() || provenance.empty() || topology_digest.empty())
    throw std::invalid_argument("AMR field topology authority is incomplete");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field topology authority names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field topology cannot change after solver materialization");
  found->second.topology_provider_kind = provider_kind;
  found->second.topology_provenance = provenance;
  found->second.topology_digest = topology_digest;
}

template <int Dim>
std::vector<runtime::field::FieldTopologyReportRow> AmrSystem<Dim>::field_topology_report(
    const std::string& provider_slot) const {
  if (!p_->field_plans.contains(provider_slot))
    throw std::out_of_range("unknown exact AMR field provider slot");
  return {};
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_plan(const std::string& provider_slot,
                                             const std::vector<std::string>& kind,
                                             const std::vector<double>& alpha,
                                             const std::vector<double>& beta,
                                             const std::vector<double>& value) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_plan");
  const std::size_t faces = static_cast<std::size_t>(2 * Dim);
  if (kind.size() != faces || alpha.size() != faces || beta.size() != faces ||
      value.size() != faces)
    throw std::invalid_argument(
        "AMR field boundary plan must cover both faces of every exact axis");
  for (std::size_t face = 0; face < faces; ++face) {
    if (kind[face] != "periodic" && kind[face] != "dirichlet" && kind[face] != "neumann" &&
        kind[face] != "mixed")
      throw std::invalid_argument("AMR field boundary kind is unknown");
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) || !std::isfinite(value[face]))
      throw std::invalid_argument("AMR field boundary coefficients must be finite");
  }
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field boundary cannot change after solver materialization");
  found->second.boundary_kind = kind;
  found->second.boundary_alpha = alpha;
  found->second.boundary_beta = beta;
  found->second.boundary_value = value;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_dependencies(const std::string& provider_slot,
                                                     const std::vector<std::string>& state_blocks,
                                                     const std::vector<int>& state_components,
                                                     const std::vector<std::string>& field_blocks,
                                                     const std::vector<std::string>& field_keys,
                                                     const std::vector<int>& field_components) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() || field_blocks.size() != field_keys.size() ||
      field_blocks.size() != field_components.size())
    throw std::invalid_argument("AMR field boundary dependency vectors differ in length");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary dependencies name an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error(
        "AMR field boundary dependencies cannot change after solver materialization");
  found->second.boundary_state_blocks = state_blocks;
  found->second.boundary_state_components = state_components;
  found->second.boundary_field_blocks = field_blocks;
  found->second.boundary_field_keys = field_keys;
  found->second.boundary_field_components = field_components;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_kernel(const std::string& provider_slot,
                                               const CompiledFieldBoundaryKernel<Dim>& kernel) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_kernel");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary kernel names an unknown provider slot");
  if (found->second.prepared_solver || found->second.boundary_kernel)
    throw std::logic_error("AMR field boundary kernel is already fixed");
  kernel.validate();
  found->second.boundary_kernel = kernel;
}

template <int Dim>
void AmrSystem<Dim>::set_field_logical_timepoint(const std::string& provider_slot,
                                                 const FieldLogicalTimePoint& point) {
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field logical timepoint names an unknown provider slot");
  const bool invalid = !std::isfinite(static_cast<double>(point.time)) ||
                       !std::isfinite(static_cast<double>(point.dt)) || point.dt <= Real(0) ||
                       point.clock_slot < 0 || point.partition_slot < 0 || point.stage_slot < 0 ||
                       point.level < 0 || point.step < 0 || point.substep < 0 ||
                       point.iteration < 0;
  if (all_reduce_max(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument("AMR field logical timepoint is incomplete");
  found->second.boundary_point = point;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_parameters(const std::string& provider_slot,
                                                   const std::vector<double>& parameters) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_parameters");
  if (!std::all_of(parameters.begin(), parameters.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("AMR field boundary parameters must be finite");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary parameters name an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error(
        "AMR field boundary parameters cannot change after solver materialization");
  found->second.boundary_parameters.assign(parameters.begin(), parameters.end());
}

template <int Dim>
void AmrSystem<Dim>::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                           int max_iterations, double linear_tolerance,
                                           int linear_max_iterations, int restart, double armijo,
                                           double minimum_step) {
  require_amr_assembling(p_->lifecycle, "set_field_newton_plan");
  FieldNewtonOptions options{
      static_cast<Real>(tolerance),   max_iterations, static_cast<Real>(linear_tolerance),
      linear_max_iterations,          restart,        static_cast<Real>(armijo),
      static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field Newton plan names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field Newton plan cannot change after solver materialization");
  found->second.newton = options;
}

template <int Dim>
void AmrSystem<Dim>::set_field_nullspace(const std::string& provider_slot,
                                         const std::string& nullspace_provider_identity,
                                         const PreparedProviderOptions& options) {
  require_amr_assembling(p_->lifecycle, "set_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("AMR field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field nullspace names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field nullspace cannot change after solver materialization");
  found->second.nullspace_provider_identity = nullspace_provider_identity;
  found->second.nullspace_options = options;
}

template <int Dim>
void AmrSystem<Dim>::register_elliptic_field(
    const std::string& block_name, const std::string& provider_key,
    const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign) {
  require_amr_assembling(p_->lifecycle, "register_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot register a named elliptic field after hierarchy materialization");
  if (provider_key.empty())
    throw std::invalid_argument("AmrSystem named elliptic field identity must be non-empty");
  (void)p_->block(block_name);
  if (!p_->auxiliary_registry.sealed())
    throw std::logic_error(
        "AMR named elliptic outputs require a sealed auxiliary provider registry");
  const runtime::field::NamedFieldOutput<Dim> output(output_keys.size(), gradient_sign);
  std::vector<std::string> exact_output_keys;
  exact_output_keys.reserve(output_keys.size());
  for (const auto& key : output_keys) {
    key.validate();
    const std::string exact_key = key.exact_key();
    if (std::find(exact_output_keys.begin(), exact_output_keys.end(), exact_key) !=
        exact_output_keys.end())
      throw std::invalid_argument("AMR named elliptic output keys must be unique");
    exact_output_keys.push_back(exact_key);
    if (p_->auxiliary_registry.provider_for_key(key).kind() !=
        runtime::system::AuxiliaryProviderKind::field_output)
      throw std::invalid_argument(
          "AMR named elliptic output key is not owned by a field-output provider");
  }
  const std::string slot = p_->resolve_field_slot(provider_key);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (plan.output)
    throw std::logic_error("AMR exact field output is already registered");
  if (plan.output_block != block_name || plan.output_key != provider_key)
    throw std::invalid_argument(
        "AMR exact field output registration differs from its resolved plan");
  plan.output = output;
  plan.output_keys = output_keys;
  plan.rhs_by_block.resize(p_->blocks.size());
}

template <int Dim>
void AmrSystem<Dim>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs) {
  require_amr_assembling(p_->lifecycle, "set_block_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot install a named elliptic RHS after hierarchy materialization");
  if (field.empty() || !rhs)
    throw std::invalid_argument(
        "AmrSystem named elliptic RHS requires a field identity and prepared closure");
  (void)p_->block(block_name);
  const std::string slot = field == "fields_from_state" && !p_->default_field_slot.empty()
                               ? p_->default_field_slot
                               : p_->resolve_field_slot(field);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (plan.prepared_solver)
    throw std::logic_error("AMR exact field RHS cannot change after solver materialization");
  Real coefficient = Real(0);
  bool contributes = false;
  if (slot == p_->default_field_slot) {
    coefficient = Real(1);
    contributes = true;
    plan.use_prepared_level_rhs = false;
  } else {
    for (const typename Impl::FieldProviderBinding& binding : plan.providers)
      if (binding.block == block_name) {
        coefficient += static_cast<Real>(binding.coefficient);
        contributes = true;
      }
  }
  if (!contributes)
    throw std::invalid_argument("AMR exact field RHS has no resolved provider binding");
  plan.rhs_by_block.resize(p_->blocks.size());
  const std::size_t block_index = static_cast<std::size_t>(std::distance(
      p_->blocks.begin(), std::find_if(p_->blocks.begin(), p_->blocks.end(),
                                       [&](const typename Impl::BlockSpec& candidate) {
                                         return candidate.name == block_name;
                                       })));
  if (block_index >= plan.rhs_by_block.size())
    throw std::logic_error("AMR exact field RHS lost its runtime block identity");
  plan.rhs_by_block[block_index].push_back({std::move(rhs), coefficient});
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_field_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int active_level, const MultiFab<Dim>* stage_override) {
  if (point.clock.empty() || point.level != active_level || point.stage < 0 ||
      point.stage_fraction.denominator <= 0 || !std::isfinite(point.dt) || point.dt <= 0.0 ||
      !std::isfinite(point.physical_time))
    throw std::invalid_argument("AMR exact field solve has an invalid evaluation point");
  if (provider_slot.empty())
    throw std::invalid_argument("AMR exact field solve requires a provider slot");
  ExactContractBuilder request;
  request.text("pops.amr.program-field-solve")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(provider_slot)
      .text(point.clock)
      .scalar(point.tick)
      .scalar(point.level)
      .scalar(point.substep)
      .scalar(point.stage)
      .scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator)
      .scalar(point.dt)
      .scalar(point.physical_time)
      .presence(stage_override != nullptr);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-program-field-solve", std::move(request).release()}}))
    throw std::invalid_argument("AMR exact field solve request differs between MPI ranks");

  const std::string slot = p_->resolve_field_slot(provider_slot);
  std::vector<const MultiFab<Dim>*> stages(p_->blocks.size(), nullptr);
  if (stage_override != nullptr) {
    if (stages.size() != 1)
      throw std::logic_error(
          "AMR exact field stage override requires the prepared multi-block hierarchy provider");
    stages.front() = stage_override;
  }
  SolveReport report;
  std::exception_ptr local_error;
  try {
    report = p_->solve_field_candidate(slot, active_level, stages, &point);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    p_->reject_field_candidate();
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR exact field solve failed on at least one MPI rank");
  }
  return p_->make_field_outcome(std::move(report));
}

template <int Dim>
void AmrSystem<Dim>::with_program_field_candidate_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int active_level, const MultiFab<Dim>& stage_override, const std::function<void()>& evaluate) {
  if (!evaluate)
    throw std::invalid_argument("AMR perturbed field evaluation requires a callback");
  p_->ensure_engine();
  typename Impl::AcceptedSnapshot accepted(*p_);
  std::exception_ptr local_error;
  try {
    SolveOutcome outcome =
        solve_program_field_at(point, provider_slot, active_level, &stage_override);
    if (!outcome.report().solved_value_available())
      throw std::runtime_error("AMR perturbed field solve produced no publishable candidate");
    (void)outcome.consume(SolveConsumption::kAccept);
    evaluate();
  } catch (...) {
    local_error = std::current_exception();
    if (!p_->active_field_slot.empty())
      p_->reject_field_candidate();
  }
  accepted.restore(*p_);
  const ExecutionLane& lane = p_->multiblock_hierarchy->lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR perturbed field evaluation failed collectively and rolled back");
  }
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_field_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int active_level, const std::vector<const MultiFab<Dim>*>& stage_overrides) {
  if (stage_overrides.size() != p_->blocks.size())
    throw std::invalid_argument(
        "AMR exact simultaneous field solve must cover the runtime block registry");
  if (stage_overrides.size() != 1)
    throw std::logic_error(
        "AMR exact simultaneous field solve requires the prepared multi-block hierarchy provider");
  return solve_program_field_at(point, provider_slot, active_level, stage_overrides.front());
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_default_field(int active_level) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  p_->ensure_engine();
  if (active_level < 0 ||
      static_cast<std::size_t>(active_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AMR default field active level lies outside the hierarchy");
  SolveReport report;
  std::exception_ptr local_error;
  try {
    report =
        p_->solve_field_candidate(p_->default_field_slot, active_level,
                                  std::vector<const MultiFab<Dim>*>(p_->blocks.size(), nullptr));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    p_->reject_field_candidate();
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR default exact field solve failed on at least one MPI rank");
  }
  return p_->make_field_outcome(std::move(report));
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::field_provider_slots() const {
  std::vector<std::string> result;
  result.reserve(p_->field_plans.size());
  for (const auto& [slot, plan] : p_->field_plans) {
    (void)plan;
    result.push_back(slot);
  }
  return result;
}

template <int Dim>
int AmrSystem<Dim>::field_provider_levels(const std::string& provider_slot) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  return p_->field_plans.at(slot).prepared_solver->level_count();
}

template <int Dim>
void AmrSystem<Dim>::set_field_potential_level(const std::string& provider_slot, int level,
                                               const std::vector<double>& phi) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (!p_->active_field_slot.empty())
    throw std::logic_error("AMR field potential cannot be restored during an active solve");
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field potential restore level lies outside the hierarchy");
  const Box<Dim>& domain = p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
  write_field(*plan.accepted_potential[static_cast<std::size_t>(level)], domain, phi, 1);
  copy_full_field_in_place(*plan.accepted_potential[static_cast<std::size_t>(level)],
                           plan.prepared_solver->candidate_level(level));
}

template <int Dim>
void AmrSystem<Dim>::set_field_potential(const std::string& provider_slot,
                                         const std::vector<double>& phi) {
  set_field_potential_level(provider_slot, 0, phi);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::field_potential_level_global(const std::string& provider_slot,
                                                                 int level) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field potential level lies outside the hierarchy");
  return gather_field(*plan.accepted_potential[static_cast<std::size_t>(level)],
                      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), 1);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::field_potential_global(const std::string& provider_slot) {
  return field_potential_level_global(provider_slot, 0);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::named_field_values(const std::string& field) {
  const std::string slot = p_->resolve_field_slot(field);
  runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = "pops.amr.direct-field-read";
  point.level = 0;
  point.stage_fraction = {0, 1};
  point.dt = 1.0;
  point.physical_time = p_->accepted_time;
  (void)consume_solve_outcome(solve_program_field_at(point, slot, 0, nullptr));
  return field_potential_level_global(slot, 0);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_field_local_pieces(
    const std::string& provider_slot, int level) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field output level lies outside the hierarchy");
  const MultiFab<Dim>& field = *plan.accepted_potential[static_cast<std::size_t>(level)];
  return output_local_pieces(field, level, field.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_field_root_pieces(
    const ObserverMpiLane& lane, const std::string& provider_slot, int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "field", provider_slot, level),
      [&] { return output_field_local_pieces(provider_slot, level); });
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_potential(int level) {
  return level_potential_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_potential_global(int level) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  return field_potential_level_global(p_->default_field_slot, level);
}

template <int Dim>
void AmrSystem<Dim>::set_level_potential(int level, const std::vector<double>& phi) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  set_field_potential_level(p_->default_field_slot, level, phi);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::potential() {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  (void)consume_solve_outcome(solve_program_default_field(0));
  return field_potential_level_global(p_->default_field_slot, 0);
}

template <int Dim>
void AmrSystem<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_amr_assembling(p_->lifecycle, "install_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_amr_assembling(p_->lifecycle, "install_prepared_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(name, identity, required_depth, state_identity,
                                         std::move(boundary));
}

template <int Dim>
void AmrSystem<Dim>::discard_hyperbolic_boundaries() {
  require_amr_assembling(p_->lifecycle, "discard_hyperbolic_boundaries");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem cannot discard state routes after block publication");
  p_->boundary_registry.discard_transaction();
  p_->prepared_boundary_components.clear();
  p_->boundary_execution_context.reset();
}

template <int Dim>
void AmrSystem<Dim>::set_density(const std::string& name, const std::vector<double>& density) {
  typename Impl::BlockSpec& block = p_->block(name);
  if (density.size() != checked_cells(p_->cfg.index_domain()))
    throw std::invalid_argument("AmrSystem density differs from the exact coarse shape");
  if (p_->engine)
    write_component(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), density, 0);
  p_->discard_level_evaluations();
  block.density = density;
  block.has_density = true;
}

template <int Dim>
void AmrSystem<Dim>::set_conservative_state(const std::string& name,
                                            const std::vector<double>& state) {
  typename Impl::BlockSpec& block = p_->block(name);
  const std::size_t cells = checked_cells(p_->cfg.index_domain());
  if (state.size() != static_cast<std::size_t>(block.ncomp) * cells)
    throw std::invalid_argument("AmrSystem state differs from the exact coarse shape");
  if (p_->engine)
    write_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), state, block.ncomp);
  p_->discard_level_evaluations();
  block.state = state;
  block.has_state = true;
}

template <int Dim>
void AmrSystem<Dim>::bind_bootstrap_subject(const std::string& subject_id,
                                            const std::string& runtime_block,
                                            const std::string& source_route) {
  const auto prepare = [&] {
    require_amr_assembling(p_->lifecycle, "bind_bootstrap_subject");
    if (p_->engine || subject_id.empty() || runtime_block.empty() || source_route.empty())
      throw std::invalid_argument(
          "AMR bootstrap subject binding requires one pre-materialization subject, block and "
          "source route");
    (void)p_->block(runtime_block);
    if (p_->boundary_registry.state_route(runtime_block) != subject_id)
      throw std::invalid_argument(
          "AMR bootstrap subject does not match the block's authenticated state route");
    if (p_->bootstrap_sources.contains(subject_id))
      throw std::invalid_argument("AMR bootstrap subject is already bound to a runtime block");
    auto candidate = p_->bootstrap_sources;
    candidate.emplace(subject_id,
                      typename Impl::BootstrapSourceAuthority{runtime_block, source_route});
    return candidate;
  };
  const auto canonicalize = [&] {
    ExactContractBuilder contract;
    contract.text("pops.amr.bootstrap-subject-binding")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(subject_id)
        .text(runtime_block)
        .text(source_route);
    return std::move(contract).release();
  };
  std::map<std::string, typename Impl::BootstrapSourceAuthority> candidate;
  if (p_->prepared_hierarchy && p_->prepared_hierarchy->lane) {
    candidate = analytic::collectively_prepare_exact_analytic_request(
        "AmrSystem::bind_bootstrap_subject", prepare, canonicalize,
        p_->prepared_hierarchy->lane->communicator());
  } else {
    candidate = prepare();
  }
  p_->bootstrap_sources = std::move(candidate);
}

template <int Dim>
void AmrSystem<Dim>::stage_bootstrap_analytic_state(
    const std::string& subject_id, const std::string& runtime_block, const std::string& space,
    const std::string& centering, const std::string& projection,
    const analytic::AnalyticOpcodeRows& opcodes, const analytic::AnalyticLiteralRows& literals) {
  const auto prepare = [&] {
    require_amr_assembling(p_->lifecycle, "stage_bootstrap_analytic_state");
    if (p_->engine || space != "cell" || centering != "cell" ||
        projection != "conservative_cell_average")
      throw std::invalid_argument(
          "AMR analytic bootstrap requires a pre-materialization cell conservative-cell-average "
          "state");
    const auto binding = p_->bootstrap_sources.find(subject_id);
    if (binding == p_->bootstrap_sources.end() || binding->second.runtime_block != runtime_block)
      throw std::invalid_argument(
          "AMR analytic bootstrap subject is not authenticated for its runtime block");
    if (binding->second.kind != Impl::BootstrapSourceKind::unstaged)
      throw std::invalid_argument("AMR bootstrap subject already owns a staged source");
    const typename Impl::BlockSpec& block = p_->block(runtime_block);
    std::vector<analytic::AnalyticProgram> programs =
        analytic::compile_component_programs(opcodes, literals);
    if (programs.size() != static_cast<std::size_t>(block.ncomp))
      throw std::invalid_argument(
          "AMR analytic bootstrap component count differs from its runtime block");
    return programs;
  };
  std::vector<analytic::AnalyticProgram> programs;
  if (p_->prepared_hierarchy && p_->prepared_hierarchy->lane) {
    programs = analytic::collectively_prepare_analytic_request(
        "AmrSystem::stage_bootstrap_analytic_state",
        {{"centering", centering},
         {"projection", projection},
         {"runtime_block", runtime_block},
         {"space", space},
         {"subject_id", subject_id}},
        {}, opcodes, literals, prepare, p_->prepared_hierarchy->lane->communicator());
  } else {
    programs = prepare();
  }
  typename Impl::BlockSpec& block = p_->block(runtime_block);
  block.density.clear();
  block.state.clear();
  block.has_density = false;
  block.has_state = false;
  block.has_analytic_state = true;
  block.analytic_state = std::move(programs);
  p_->bootstrap_sources.at(subject_id).kind = Impl::BootstrapSourceKind::analytic;
}

template <int Dim>
void AmrSystem<Dim>::stage_bootstrap_array(const std::string& subject_id,
                                           const std::string& runtime_block,
                                           const std::string& space, const std::string& centering,
                                           int components, const Extent<Dim>& spatial_shape,
                                           const std::vector<double>& values) {
  const auto prepare = [&] {
    require_amr_assembling(p_->lifecycle, "stage_bootstrap_array");
    if (p_->engine || space != "cell" || centering != "cell")
      throw std::invalid_argument(
          "AMR array bootstrap requires one pre-materialization cell-centred state");
    const auto binding = p_->bootstrap_sources.find(subject_id);
    if (binding == p_->bootstrap_sources.end() || binding->second.runtime_block != runtime_block)
      throw std::invalid_argument(
          "AMR array bootstrap subject is not authenticated for its runtime block");
    if (binding->second.kind != Impl::BootstrapSourceKind::unstaged)
      throw std::invalid_argument("AMR bootstrap subject already owns a staged source");
    const typename Impl::BlockSpec& block = p_->block(runtime_block);
    const std::size_t cells = checked_cells(p_->cfg.index_domain());
    if (spatial_shape != p_->cfg.shape || components != block.ncomp ||
        values.size() != static_cast<std::size_t>(components) * cells)
      throw std::invalid_argument(
          "AMR bootstrap array differs from the exact-rank runtime block shape");
    return values;
  };
  std::vector<double> state;
  if (p_->prepared_hierarchy && p_->prepared_hierarchy->lane) {
    const auto canonicalize = [&] {
      ExactContractBuilder contract;
      contract.text("pops.amr.bootstrap-array")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .text(subject_id)
          .text(runtime_block)
          .text(space)
          .text(centering)
          .scalar(components);
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(spatial_shape[axis]);
      contract.scalar(static_cast<std::uint64_t>(values.size()));
      for (const double value : values)
        contract.scalar(value);
      return std::move(contract).release();
    };
    state = analytic::collectively_prepare_exact_analytic_request(
        "AmrSystem::stage_bootstrap_array", prepare, canonicalize,
        p_->prepared_hierarchy->lane->communicator());
  } else {
    state = prepare();
  }
  typename Impl::BlockSpec& block = p_->block(runtime_block);
  block.density.clear();
  block.analytic_state.clear();
  block.has_density = false;
  block.has_analytic_state = false;
  block.has_state = true;
  block.state = std::move(state);
  p_->bootstrap_sources.at(subject_id).kind = Impl::BootstrapSourceKind::array;
}

template <int Dim>
std::size_t AmrSystem<Dim>::materialize_bootstrap_action(const std::string& subject_id,
                                                         const std::string& action,
                                                         const std::string& action_route,
                                                         int level) {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AMR bootstrap materialization requires an active transaction");
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR bootstrap materialization lacks its prepared execution lane");
  const CommunicatorView communicator = p_->prepared_hierarchy->lane->communicator();
  const auto validate = [&] {
    if (subject_id.empty() || action.empty() || action_route.empty())
      throw std::invalid_argument(
          "AMR bootstrap materialization requires one subject, action and native route");
    const auto source = p_->bootstrap_sources.find(subject_id);
    if (source == p_->bootstrap_sources.end() ||
        source->second.kind == Impl::BootstrapSourceKind::unstaged)
      throw std::invalid_argument(
          "AMR bootstrap materialization has no authenticated staged source");
    if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
      throw std::out_of_range("AMR bootstrap materialization level is not live");
    if (p_->bootstrap_materialized_actions.contains(std::make_pair(subject_id, level)))
      throw std::logic_error("AMR bootstrap source is already materialized at this level");
    if (level == 0) {
      if (action != "initialize_level_zero" || action_route != source->second.source_route)
        throw std::invalid_argument(
            "AMR level-zero bootstrap action does not authenticate its staged source");
      return;
    }
    if (static_cast<std::size_t>(level + 1) != p_->engine->hierarchy().num_levels())
      throw std::invalid_argument(
          "AMR fine bootstrap action must target the newest live hierarchy level");
    if (source->second.kind == Impl::BootstrapSourceKind::analytic) {
      if (action != "analytic_reprojection" || action_route != source->second.source_route)
        throw std::invalid_argument(
            "AMR analytic bootstrap action does not authenticate its staged source");
      return;
    }
    if (action != "prolong_from_parent")
      throw std::invalid_argument(
          "AMR array bootstrap requires the resolved parent prolongation action");
    const auto subject_route =
        p_->bootstrap_subject_routes.find(std::make_pair(subject_id, std::string("prolongation")));
    if (subject_route == p_->bootstrap_subject_routes.end())
      throw std::invalid_argument("AMR array bootstrap has no authenticated prolongation provider");
    const auto route = p_->bootstrap_transfer_routes.find(subject_route->second);
    if (route == p_->bootstrap_transfer_routes.end() || route->second.kernel != action_route)
      throw std::invalid_argument(
          "AMR array bootstrap action differs from its registered native provider");
    const std::size_t transition = static_cast<std::size_t>(level - 1);
    if (transition >= p_->cfg.transition_ratios.size() ||
        route->second.refinement_ratio != p_->cfg.transition_ratios[transition])
      throw std::invalid_argument(
          "AMR array bootstrap provider does not authenticate this ranked transition");
  };
  (void)analytic::collectively_prepare_exact_analytic_request(
      "AmrSystem::materialize_bootstrap_action",
      [&] {
        validate();
        return true;
      },
      [&] {
        ExactContractBuilder contract;
        contract.text("pops.amr.bootstrap-materialization")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .text(subject_id)
            .text(action)
            .text(action_route)
            .scalar(level);
        return std::move(contract).release();
      },
      communicator);

  const auto source = p_->bootstrap_sources.find(subject_id);
  const auto block = std::find_if(p_->blocks.begin(), p_->blocks.end(),
                                  [&](const typename Impl::BlockSpec& candidate) {
                                    return candidate.name == source->second.runtime_block;
                                  });
  if (block == p_->blocks.end())
    throw std::logic_error("AMR bootstrap source lost its authenticated runtime block");
  const std::size_t block_index =
      static_cast<std::size_t>(std::distance(p_->blocks.begin(), block));
  typename Impl::field_type& target =
      p_->multiblock_hierarchy->state(block_index, static_cast<std::size_t>(level));
  const auto require_collective_success = [&](const std::exception_ptr& local_error,
                                              std::string_view phase) {
    const long failures = all_reduce_sum(local_error ? 1L : 0L, communicator);
    if (failures == 0)
      return;
    if (communicator.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR bootstrap " + std::string(phase) + " failed collectively on " +
                             std::to_string(failures) + " rank(s)");
  };

  using field_type = typename Impl::field_type;
  using analytic_materialization_type =
      analytic::PreparedAnalyticMaterialization<Dim, typename field_type::memory_space>;

  std::optional<field_type> candidate_storage;
  field_type* candidate = nullptr;
  std::optional<analytic_materialization_type> analytic_materialization;
  std::unique_ptr<PreparedRegriddedStateTransfer<Dim>> transfer_materialization;
  std::set<std::pair<std::string, int>> candidate_materialized_actions;
  std::size_t materialized = 0;
  std::exception_ptr preparation_error;
  try {
    candidate_materialized_actions = p_->bootstrap_materialized_actions;
    const auto [marker, inserted] = candidate_materialized_actions.emplace(subject_id, level);
    (void)marker;
    if (!inserted)
      throw std::logic_error("AMR bootstrap materialization marker already exists");
    if (source->second.kind == Impl::BootstrapSourceKind::analytic) {
      candidate_storage.emplace(target.layout(), target.distribution(), target.local_rank(),
                                target.ncomp(), target.ghosts());
      candidate = &*candidate_storage;
      candidate->set_val(Real(0));
      const Box<Dim>& domain =
          p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
      const Geometry<Dim> geometry =
          Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
      analytic_materialization.emplace(analytic::prepare_cell_average_materialization(
          *candidate, geometry, block->analytic_state));
      materialized = static_cast<std::size_t>(analytic_materialization->materialized_values);
    } else if (level == 0) {
      candidate_storage.emplace(target.layout(), target.distribution(), target.local_rank(),
                                target.ncomp(), target.ghosts());
      candidate = &*candidate_storage;
      candidate->set_val(Real(0));
      write_field(*candidate, p_->cfg.index_domain(), block->state, block->ncomp);
      materialized = checked_size_product(
          static_cast<std::size_t>(block->ncomp), checked_cells(p_->cfg.index_domain()),
          "AMR level-zero bootstrap materialization exceeds size_t");
    } else {
      const std::size_t parent_level = static_cast<std::size_t>(level - 1);
      const auto subject_route = p_->bootstrap_subject_routes.find(
          std::make_pair(subject_id, std::string("prolongation")));
      const auto route = p_->bootstrap_transfer_routes.find(subject_route->second);
      const amr::transfer::TransferKind kind = route->second.kernel == "conservative_linear"
                                                   ? amr::transfer::TransferKind::LinearProlongation
                                                   : amr::transfer::TransferKind::ConstantInjection;
      transfer_materialization = prepare_regridded_state_transfer(
          p_->multiblock_hierarchy->state(block_index, parent_level),
          p_->engine->hierarchy().layout(parent_level),
          p_->engine->hierarchy().layout(static_cast<std::size_t>(level)),
          std::optional<SparseFieldImage<Dim>>{}, kind, communicator.rank());
      candidate = &transfer_materialization->child;
      materialized =
          checked_size_product(static_cast<std::size_t>(block->ncomp),
                               static_cast<std::size_t>(checked_layout_cells(target.layout())),
                               "AMR fine bootstrap materialization exceeds size_t");
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }
  require_collective_success(preparation_error, "materialization preparation");

  std::exception_ptr execution_error;
  try {
    if (analytic_materialization)
      (void)analytic::materialize_cell_average(*analytic_materialization, communicator);
    else if (transfer_materialization)
      execute_regridded_state_transfer(*transfer_materialization, communicator);
  } catch (...) {
    execution_error = std::current_exception();
  }
  require_collective_success(execution_error, "materialization execution");
  if (candidate == nullptr)
    throw std::logic_error("AMR bootstrap materialization produced no collective candidate");

  std::exception_ptr publication_error;
  try {
    copy_full_field_in_place(*candidate, target);
  } catch (...) {
    publication_error = std::current_exception();
  }
  require_collective_success(publication_error, "materialization publication");
  static_assert(noexcept(p_->bootstrap_materialized_actions.swap(candidate_materialized_actions)));
  p_->bootstrap_materialized_actions.swap(candidate_materialized_actions);
  p_->discard_level_evaluations();
  return materialized;
}

template <int Dim>
runtime::amr::PreparedTaggerCandidates<Dim> AmrSystem<Dim>::execute_prepared_tagging(
    int parent_level) {
  p_->ensure_engine();
  return p_->execute_tagging(parent_level);
}

template <int Dim>
bool AmrSystem<Dim>::regrid_from_prepared_tagging(int parent_level) {
  return p_->execute_transaction([&] { return p_->regrid_parent(parent_level); });
}

template <int Dim>
void AmrSystem<Dim>::begin_bootstrap_plan() {
  if (!p_->cfg.explicit_bootstrap)
    throw std::logic_error("AmrSystem explicit bootstrap is disabled by the resolved layout");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap requires the accepted t=0/step=0 state");
  if (!p_->tagging_spec)
    throw std::logic_error("AmrSystem bootstrap requires a prepared tagging authority");
  if (p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap transaction is already active");
  if (!p_->bootstrap_materialized_actions.empty())
    throw std::logic_error("AmrSystem bootstrap retained stale materialization actions");
  if (p_->bootstrap_sources.size() != p_->blocks.size())
    throw std::logic_error(
        "AmrSystem explicit bootstrap requires one authenticated initial source per block");
  std::set<std::string> source_blocks;
  for (const auto& [subject, source] : p_->bootstrap_sources) {
    (void)subject;
    if (source.kind == Impl::BootstrapSourceKind::unstaged ||
        !source_blocks.insert(source.runtime_block).second)
      throw std::logic_error(
          "AmrSystem explicit bootstrap sources do not uniquely cover the runtime blocks");
  }
  p_->ensure_engine();
  auto transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  runtime::amr::PersistentTaggingState<Dim> staged_state = p_->tagging_state;
  staged_state.begin_cycle(p_->tagging_spec->min_cycles);
  p_->bootstrap_transaction = std::move(transaction);
  p_->tagging_state = std::move(staged_state);
}

template <int Dim>
bool AmrSystem<Dim>::bootstrap_next_level() {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap level requires an active transaction");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap cannot advance the authoritative clock");
  const int parent_level = static_cast<int>(p_->engine->hierarchy().num_levels()) - 1;
  if (parent_level >= p_->cfg.level_count - 1)
    throw std::out_of_range("AmrSystem bootstrap would exceed the resolved hierarchy depth");
  for (const auto& [subject, source] : p_->bootstrap_sources) {
    (void)source;
    if (!p_->bootstrap_materialized_actions.contains(std::make_pair(subject, parent_level)))
      throw std::logic_error("AmrSystem bootstrap cannot tag an unmaterialized parent source");
  }
  return p_->regrid_parent(parent_level, std::nullopt, &p_->tagging_state);
}

template <int Dim>
void AmrSystem<Dim>::commit_bootstrap_level() {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap commit has no active transaction");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap commit requires the accepted t=0/step=0 state");
  for (const auto& [subject, source] : p_->bootstrap_sources) {
    (void)source;
    for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
      if (!p_->bootstrap_materialized_actions.contains(
              std::make_pair(subject, static_cast<int>(level))))
        throw std::logic_error(
            "AmrSystem bootstrap commit requires every live source level to be materialized");
  }
  p_->program.refresh_hierarchy_state("AmrSystem::commit_bootstrap_level");
  p_->publish_tagging_checkpoint();
  p_->bootstrap_transaction.reset();
  p_->bootstrap_materialized_actions.clear();
  p_->automatic_bootstrap_complete = true;
}

template <int Dim>
void AmrSystem<Dim>::rollback_bootstrap_level() {
  if (!p_->multiblock_hierarchy)
    throw std::logic_error("AmrSystem bootstrap rollback has no prepared execution lane");
  const CommunicatorView communicator = p_->multiblock_hierarchy->lane().communicator();
  const long missing = all_reduce_sum(p_->bootstrap_transaction ? 0L : 1L, communicator);
  if (missing != 0) {
    p_->bootstrap_transaction.reset();
    p_->bootstrap_materialized_actions.clear();
    (void)all_reduce_sum(0L, communicator);
    throw std::logic_error("AmrSystem bootstrap rollback transaction differs between ranks");
  }
  std::unique_ptr<typename Impl::AcceptedSnapshot> transaction =
      std::move(p_->bootstrap_transaction);
  std::exception_ptr local_error;
  try {
    transaction->restore(*p_);
  } catch (...) {
    local_error = std::current_exception();
  }
  transaction.reset();
  p_->bootstrap_materialized_actions.clear();
  const long failures = all_reduce_sum(local_error ? 1L : 0L, communicator);
  if (failures == 0)
    return;
  if (communicator.size() == 1 && local_error)
    std::rethrow_exception(local_error);
  throw std::runtime_error("AmrSystem bootstrap rollback failed collectively on " +
                           std::to_string(failures) + " rank(s)");
}

template <int Dim>
void AmrSystem<Dim>::register_bootstrap_transfer_route(
    const std::string& identity, const std::vector<std::string>& subjects,
    const std::string& provider_identity, const std::string& space, const std::string& centering,
    const std::string& representation, const std::string& storage, const std::string& operation,
    const std::string& kernel, int order, const Extent<Dim>& ghost_depth,
    const Extent<Dim>& refinement_ratio) {
  std::map<std::string, typename Impl::BootstrapTransferRoute> staged_routes;
  std::map<std::pair<std::string, std::string>, std::string> staged_subjects;
  std::string exact;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "register_bootstrap_transfer_route");
    if (p_->engine || identity.empty() || subjects.empty() || provider_identity.empty() ||
        space.empty() || centering.empty() || representation.empty() || storage.empty() ||
        operation.empty() || kernel.empty() || order < 1)
      throw std::invalid_argument(
          "AMR bootstrap transfer requires one complete pre-materialization provider route");
    std::set<std::string> unique_subjects;
    for (const std::string& subject : subjects)
      if (subject.empty() || !unique_subjects.insert(subject).second)
        throw std::invalid_argument("AMR bootstrap transfer subjects must be non-empty and unique");
    bool ratio_matches_transition = p_->cfg.transition_ratios.empty();
    for (int axis = 0; axis < Dim; ++axis)
      if (ghost_depth[axis] < 0 || refinement_ratio[axis] < 1)
        throw std::invalid_argument(
            "AMR bootstrap transfer ghosts and ranked refinement ratio are invalid");
    for (const Extent<Dim>& configured : p_->cfg.transition_ratios)
      ratio_matches_transition = ratio_matches_transition || configured == refinement_ratio;
    if (!ratio_matches_transition)
      throw std::invalid_argument(
          "AMR bootstrap transfer route names no configured ranked hierarchy transition");
    if (operation == "prolongation" && space == "cell") {
      if (centering != "cell" || representation != "conservative" || storage != "dense")
        throw std::invalid_argument(
            "AMR state prolongation requires the exact cell-centered conservative dense route");
      if (kernel == "conservative_linear") {
        bool lacks_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          lacks_stencil = lacks_stencil || ghost_depth[axis] < 1;
        if (order != 2 || lacks_stencil)
          throw std::invalid_argument(
              "AMR conservative-linear prolongation requires order two and one ranked ghost");
      } else if (kernel == "conservative_injection") {
        bool has_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          has_stencil = has_stencil || ghost_depth[axis] != 0;
        if (order != 1 || has_stencil)
          throw std::invalid_argument(
              "AMR conservative injection requires explicit order one and zero ghosts");
      } else {
        throw std::invalid_argument("AMR prolongation kernel has no exact native provider");
      }
    } else if (operation == "prolongation" && space == "face") {
      bool exact_ghost = true;
      for (int axis = 0; axis < Dim; ++axis)
        exact_ghost = exact_ghost && ghost_depth[axis] == 1;
      if (subjects.size() != 1 || centering.size() <= 5 || centering.rfind("face_", 0) != 0 ||
          representation != "conservative" || storage != "dense" ||
          kernel != "divergence_preserving_face" || order != 2 || !exact_ghost)
        throw std::invalid_argument(
            "AMR face prolongation requires one oriented conservative dense subject, the "
            "divergence-preserving order-two provider, and one exact-ranked ghost");
    } else if (operation == "prolongation" && space == "node") {
      bool zero_ghost = true;
      for (int axis = 0; axis < Dim; ++axis)
        zero_ghost = zero_ghost && ghost_depth[axis] == 0;
      if (subjects.size() != 1 || centering != "node" || representation != "primitive" ||
          storage != "dense" || kernel != "node_multilinear" || order != 2 || !zero_ghost)
        throw std::invalid_argument(
            "AMR node prolongation requires one node-centered primitive dense subject, the "
            "multilinear order-two provider, and zero ghosts");
    } else if (operation == "prolongation") {
      throw std::invalid_argument("AMR prolongation space has no exact native provider");
    }
    if (operation == "restriction") {
      bool zero_ghost = true;
      for (int axis = 0; axis < Dim; ++axis)
        zero_ghost = zero_ghost && ghost_depth[axis] == 0;
      if (space != "cell" || centering != "cell" || representation != "conservative" ||
          storage != "dense" || kernel != "volume_average" || order != 1 || !zero_ghost)
        throw std::invalid_argument(
            "AMR restriction requires the exact conservative cell-volume average provider");
    }
    if (operation == "coarse_fine_fill") {
      if (space != "cell" || centering != "cell" || representation != "conservative" ||
          storage != "dense")
        throw std::invalid_argument(
            "AMR state coarse/fine fill requires the exact cell-centered conservative dense "
            "route");
      if (kernel == "conservative_coarse_fine") {
        bool lacks_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          lacks_stencil = lacks_stencil || ghost_depth[axis] < 1;
        if (order != 2 || lacks_stencil)
          throw std::invalid_argument(
              "AMR limited-linear coarse/fine fill requires order two and one ranked ghost");
      } else if (kernel == "conservative_polynomial5_coarse_fine") {
        bool lacks_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          lacks_stencil = lacks_stencil || ghost_depth[axis] < 3;
        if (order != 5 || lacks_stencil)
          throw std::invalid_argument(
              "AMR polynomial coarse/fine fill requires order five and three ranked ghosts");
      } else {
        throw std::invalid_argument("AMR coarse/fine kernel has no exact native provider");
      }
    }
    if (operation == "temporal_interpolation") {
      bool zero_ghost = true;
      for (int axis = 0; axis < Dim; ++axis)
        zero_ghost = zero_ghost && ghost_depth[axis] == 0;
      if (space != "cell" || centering != "cell" || representation != "conservative" ||
          storage != "dense" || kernel != "linear_time_interpolation" || order != 2 || !zero_ghost)
        throw std::invalid_argument(
            "AMR temporal interpolation requires the exact conservative linear-time provider");
    }
    if (operation != "prolongation" && operation != "restriction" &&
        operation != "coarse_fine_fill" && operation != "temporal_interpolation")
      throw std::invalid_argument("AMR bootstrap transfer operation has no exact native provider");

    typename Impl::BootstrapTransferRoute candidate{identity, subjects,    provider_identity,
                                                    space,    centering,   representation,
                                                    storage,  operation,   kernel,
                                                    order,    ghost_depth, refinement_ratio};
    ExactContractBuilder contract;
    contract.text("pops.amr-system.bootstrap-transfer-route")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(identity)
        .text(provider_identity)
        .text(space)
        .text(centering)
        .text(representation)
        .text(storage)
        .text(operation)
        .text(kernel)
        .scalar(order)
        .sequence(subjects, [](ExactContractBuilder& item, const std::string& subject) {
          item.text(subject);
        });
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(ghost_depth[axis]).scalar(refinement_ratio[axis]);
    exact = std::move(contract).release();

    staged_routes = p_->bootstrap_transfer_routes;
    staged_subjects = p_->bootstrap_subject_routes;
    if (!staged_routes.emplace(identity, std::move(candidate)).second)
      throw std::invalid_argument("AMR bootstrap transfer route identity is not unique");
    for (const std::string& subject : subjects)
      if (!staged_subjects.emplace(std::make_pair(subject, operation), identity).second)
        throw std::invalid_argument(
            "AMR bootstrap transfer subject already owns this operation route");
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR bootstrap transfer route failed validation on another MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("bootstrap-transfer-route"), std::string_view(exact)}}))
    throw std::invalid_argument("AMR bootstrap transfer route differs between MPI ranks");
  p_->bootstrap_transfer_routes = std::move(staged_routes);
  p_->bootstrap_subject_routes = std::move(staged_subjects);
}

template <int Dim>
void AmrSystem<Dim>::register_bootstrap_oriented_face_subjects(
    const std::vector<std::string>& oriented_subjects) {
  std::map<std::string, std::array<std::string, Dim>> staged_groups;
  std::string exact;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "register_bootstrap_oriented_face_subjects");
    if (p_->engine || oriented_subjects.size() != static_cast<std::size_t>(Dim))
      throw std::invalid_argument(
          "AMR oriented face provider requires exactly one subject per native axis");
    std::set<std::string> unique_subjects;
    std::set<std::string> unique_centerings;
    std::array<std::string, Dim> ranked_subjects{};
    std::string provider_identity;
    Extent<Dim> refinement_ratio{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::string& subject = oriented_subjects[static_cast<std::size_t>(axis)];
      if (subject.empty() || !unique_subjects.insert(subject).second)
        throw std::invalid_argument(
            "AMR oriented face subjects must be non-empty and unique by native axis");
      const auto subject_route =
          p_->bootstrap_subject_routes.find(std::make_pair(subject, std::string("prolongation")));
      if (subject_route == p_->bootstrap_subject_routes.end())
        throw std::invalid_argument(
            "AMR oriented face subject has no registered prolongation provider route");
      const auto route = p_->bootstrap_transfer_routes.find(subject_route->second);
      if (route == p_->bootstrap_transfer_routes.end())
        throw std::logic_error("AMR oriented face subject lost its provider route");
      const typename Impl::BootstrapTransferRoute& descriptor = route->second;
      if (descriptor.space != "face" || descriptor.centering.size() <= 5 ||
          descriptor.centering.rfind("face_", 0) != 0 ||
          descriptor.representation != "conservative" || descriptor.storage != "dense" ||
          descriptor.operation != "prolongation" ||
          descriptor.kernel != "divergence_preserving_face" || descriptor.order != 2 ||
          descriptor.subjects != std::vector<std::string>{subject})
        throw std::invalid_argument(
            "AMR oriented face group references a non-exact divergence-preserving route");
      if (!unique_centerings.insert(descriptor.centering).second)
        throw std::invalid_argument(
            "AMR oriented face group must contain one distinct centering per native axis");
      if (axis == 0) {
        provider_identity = descriptor.provider_identity;
        refinement_ratio = descriptor.refinement_ratio;
      } else if (descriptor.provider_identity != provider_identity ||
                 descriptor.refinement_ratio != refinement_ratio) {
        throw std::invalid_argument(
            "AMR oriented face subjects must share one provider and ranked transition ratio");
      }
      ranked_subjects[static_cast<std::size_t>(axis)] = subject;
    }

    ExactContractBuilder contract;
    contract.text("pops.amr-system.bootstrap-oriented-face-subjects")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_identity)
        .sequence(oriented_subjects, [](ExactContractBuilder& item, const std::string& subject) {
          item.text(subject);
        });
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(refinement_ratio[axis]);
    exact = std::move(contract).release();

    staged_groups = p_->bootstrap_oriented_face_groups;
    if (!staged_groups.emplace(provider_identity, std::move(ranked_subjects)).second)
      throw std::invalid_argument("AMR oriented face provider identity is not unique");
    for (const auto& [existing_provider, existing_subjects] : staged_groups) {
      if (existing_provider == provider_identity)
        continue;
      for (const std::string& existing : existing_subjects)
        if (unique_subjects.contains(existing))
          throw std::invalid_argument(
              "AMR oriented face subject cannot belong to multiple provider groups");
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR oriented face provider failed validation on another MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("bootstrap-oriented-face-subjects"), std::string_view(exact)}}))
    throw std::invalid_argument("AMR oriented face subjects differ between MPI ranks");
  p_->bootstrap_oriented_face_groups = std::move(staged_groups);
}

template <int Dim>
void AmrSystem<Dim>::add_dt_bound(const std::string& label, std::function<double()> evaluate) {
  require_amr_assembling(p_->lifecycle, "add_dt_bound");
  if (label.empty() || !evaluate)
    throw std::invalid_argument("AmrSystem dt bound requires a label and provider");
  p_->dt_bounds.push_back(typename Impl::GlobalDtBound{label, std::move(evaluate)});
}

template <int Dim>
std::string AmrSystem<Dim>::last_dt_bound() const {
  return p_->last_dt_reason;
}

template <int Dim>
void AmrSystem<Dim>::step(double dt) {
  p_->program.require_step_installed("AmrSystem::step");
  runtime::program::ProfileScope scope(p_->program.profiler_, "step");
  p_->program.profiler_.count("steps");
  if (p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem cannot step during an active bootstrap transaction");
  p_->execute_transaction([&] {
    p_->program.dispatch_cadence_step(p_->accepted_time, p_->macro_step, dt, "AmrSystem");
    p_->program.refresh_hierarchy_state("AmrSystem::step");
    if (!p_->tagging_spec || p_->cfg.regrid_every == 0 ||
        p_->macro_step % p_->cfg.regrid_every != 0)
      return;
    std::vector<std::optional<SparseFieldImage<Dim>>> previous(
        p_->engine->hierarchy().num_levels());
    for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
      previous[level] = gather_sparse_field(p_->engine->hierarchy().state(level),
                                            p_->engine->hierarchy().layout(level).domain(),
                                            p_->prepared_hierarchy->lane->communicator());
    runtime::amr::PersistentTaggingState<Dim> staged_state = p_->tagging_state;
    staged_state.begin_cycle(p_->tagging_spec->min_cycles);
    for (int parent_level = 0; parent_level < p_->cfg.level_count - 1; ++parent_level) {
      const std::size_t child = static_cast<std::size_t>(parent_level + 1);
      const std::optional<SparseFieldImage<Dim>> old_child =
          child < previous.size() ? previous[child] : std::nullopt;
      if (!p_->regrid_parent(parent_level, old_child, &staged_state))
        break;
    }
    p_->tagging_state = std::move(staged_state);
    p_->publish_tagging_checkpoint();
  });
  p_->discard_level_evaluations();
}

template <int Dim>
void AmrSystem<Dim>::advance(double dt, int nsteps) {
  p_->program.require_step_installed("AmrSystem::advance");
  if (nsteps < 0)
    throw std::invalid_argument("AmrSystem::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
double AmrSystem<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  std::string request_contract;
  std::exception_ptr request_error;
  long request_failure = 0;
  try {
    p_->program.require_step_installed("AmrSystem::step_cfl");
    if (!std::isfinite(cfl) || cfl <= 0.0 || !std::isfinite(speed_floor) || speed_floor <= 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl requires positive finite CFL inputs");
    if (std::isnan(max_dt) || max_dt <= 0.0 || !std::isfinite(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl received invalid strategy bounds");
    if (p_->blocks.empty() || p_->prepared_blocks.size() != p_->blocks.size())
      throw std::logic_error("AmrSystem::step_cfl requires one retained generated block");

    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-request")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(cfl)
        .scalar(speed_floor)
        .scalar(max_dt)
        .scalar(min_dt)
        .bytes(p_->prepared_blocks.front().collective_contract)
        .scalar(static_cast<std::uint64_t>(p_->dt_bounds.size()));
    for (const typename Impl::GlobalDtBound& bound : p_->dt_bounds)
      contract.text(bound.label);
    contract.scalar(static_cast<bool>(p_->program.dt_bound_));
    request_contract = std::move(contract).release();
  } catch (...) {
    request_failure = 1;
    request_error = std::current_exception();
  }
  if (all_reduce_max(request_failure) != 0) {
    if (request_error)
      std::rethrow_exception(request_error);
    throw std::runtime_error("AmrSystem::step_cfl request validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-request"), std::string_view(request_contract)}}))
    throw std::invalid_argument(
        "AmrSystem::step_cfl inputs or prepared bound authorities differ between MPI ranks");

  p_->ensure_engine();
  const long invalid_hierarchy =
      !p_->prepared_hierarchy || p_->prepared_hierarchy->block_levels.size() != p_->blocks.size()
          ? 1L
          : 0L;
  if (all_reduce_max(invalid_hierarchy) != 0)
    throw std::logic_error("AmrSystem::step_cfl requires one live prepared hierarchy graph");

  enum class BoundKind : std::int32_t {
    degenerate,
    transport,
    source_frequency,
    stability_dt,
    global,
    program,
    maximum_dt,
  };
  double selected = std::numeric_limits<double>::infinity();
  BoundKind reason_kind = BoundKind::degenerate;
  std::size_t reason_block_index = 0;
  std::size_t global_reason_index = std::numeric_limits<std::size_t>::max();
  for (std::size_t block_index = 0; block_index < p_->blocks.size(); ++block_index) {
    const typename Impl::BlockSpec& block = p_->blocks[block_index];
    for (std::size_t level = 0; level < p_->prepared_hierarchy->block_levels[block_index].size();
         ++level) {
      const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
      const Geometry<Dim> geometry =
          Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
      Real spacing = geometry.spacing(0);
      for (int axis = 1; axis < Dim; ++axis)
        spacing = std::min(spacing, geometry.spacing(axis));

      const typename Impl::level_block_type& prepared_level =
          p_->prepared_hierarchy->block_levels[block_index][level];
      const Real speed = std::max(prepared_level.maximum_speed(), static_cast<Real>(speed_floor));
      double level_dt = cfl * static_cast<double>(spacing) * block.substeps /
                        (static_cast<double>(block.stride) * static_cast<double>(speed));
      const char* level_reason = "transport";
      if (const std::optional<Real> frequency = prepared_level.source_frequency();
          frequency && *frequency > Real(0)) {
        const double source_dt =
            cfl * block.substeps /
            (static_cast<double>(block.stride) * static_cast<double>(*frequency));
        if (source_dt < level_dt) {
          level_dt = source_dt;
          level_reason = "source_frequency";
        }
      }
      if (const std::optional<Real> admissible = prepared_level.stability_dt();
          admissible && *admissible > Real(0)) {
        const double stability_dt =
            static_cast<double>(*admissible) * block.substeps / static_cast<double>(block.stride);
        if (stability_dt < level_dt) {
          level_dt = stability_dt;
          level_reason = "stability_dt";
        }
      }
      if (level_dt < selected) {
        selected = level_dt;
        if (std::string_view(level_reason) == "source_frequency")
          reason_kind = BoundKind::source_frequency;
        else if (std::string_view(level_reason) == "stability_dt")
          reason_kind = BoundKind::stability_dt;
        else
          reason_kind = BoundKind::transport;
        reason_block_index = block_index;
      }
    }
  }
  if (!std::isfinite(selected))
    throw std::runtime_error("AmrSystem::step_cfl found no finite generated stability bound");

  for (std::size_t bound_index = 0; bound_index < p_->dt_bounds.size(); ++bound_index) {
    const typename Impl::GlobalDtBound& bound = p_->dt_bounds[bound_index];
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = bound.evaluate();
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem global dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem global dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::global;
      global_reason_index = bound_index;
    }
  }
  const bool has_program_bound = static_cast<bool>(p_->program.dt_bound_);
  if (has_program_bound) {
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = static_cast<double>(p_->program.dt_bound_(static_cast<Real>(cfl)));
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem Program dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem Program dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (std::isfinite(candidate) && candidate > 0.0 && candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::program;
    }
  }
  if (max_dt < selected) {
    selected = max_dt;
    reason_kind = BoundKind::maximum_dt;
  }
  if (selected < min_dt)
    throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");

  std::string reason;
  std::string decision_contract;
  std::exception_ptr decision_error;
  long decision_failure = 0;
  try {
    switch (reason_kind) {
      case BoundKind::transport:
        reason = "transport:" + p_->blocks[reason_block_index].name;
        break;
      case BoundKind::source_frequency:
        reason = "source_frequency:" + p_->blocks[reason_block_index].name;
        break;
      case BoundKind::stability_dt:
        reason = "stability_dt:" + p_->blocks[reason_block_index].name;
        break;
      case BoundKind::global:
        if (global_reason_index >= p_->dt_bounds.size())
          throw std::logic_error("AmrSystem::step_cfl lost its selected global bound identity");
        reason = "global:" + p_->dt_bounds[global_reason_index].label;
        break;
      case BoundKind::program:
        reason = "program:dt_bound";
        break;
      case BoundKind::maximum_dt:
        reason = "strategy:max_dt";
        break;
      case BoundKind::degenerate:
        reason = "degenerate";
        break;
    }
    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-decision")
        .scalar(std::uint32_t{1})
        .scalar(selected)
        .scalar(reason_kind)
        .scalar(static_cast<std::uint64_t>(global_reason_index))
        .text(reason);
    decision_contract = std::move(contract).release();
  } catch (...) {
    decision_failure = 1;
    decision_error = std::current_exception();
  }
  if (all_reduce_max(decision_failure) != 0) {
    if (decision_error)
      std::rethrow_exception(decision_error);
    throw std::runtime_error("AmrSystem::step_cfl decision preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-decision"), std::string_view(decision_contract)}}))
    throw std::runtime_error("AmrSystem::step_cfl selected different bounds across MPI ranks");
  p_->last_dt_reason = std::move(reason);
  step(selected);
  return selected;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_transaction() {
  p_->ensure_engine();
  if (p_->external_step_transaction)
    throw std::runtime_error("AmrSystem step transaction is already active");
  p_->external_step_transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::commit_step_transaction() {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no active uncommitted step transaction");
  p_->external_step_committed = true;
}

template <int Dim>
void AmrSystem<Dim>::finalize_step_transaction() {
  if (!p_->external_step_transaction || !p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no committed step transaction");
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::rollback_step_transaction() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no active step transaction");
  p_->external_step_transaction->restore(*p_);
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
bool AmrSystem<Dim>::has_active_step_transaction() const noexcept {
  return static_cast<bool>(p_->external_step_transaction);
}

template <int Dim>
void AmrSystem<Dim>::restore_active_step_transaction_for_program() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no outer Program rollback image");
  p_->external_step_transaction->restore(*p_);
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction || !p_->engine || !p_->external_step_transaction->engine)
    throw std::runtime_error("AmrSystem::step_change_l2 requires an active transaction");
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error(
        "AmrSystem::step_change_l2 requires a prepared composite coverage provider");
  const MultiFab<Dim>& current = p_->engine->hierarchy().state(0);
  const MultiFab<Dim>& previous = p_->external_step_transaction->engine->hierarchy.state(0);
  const double value = std::sqrt(cell_measure(p_->cfg, p_->cfg.index_domain()) *
                                 static_cast<double>(difference_sum_sq_all(current, previous)));
  return {{p_->blocks.front().name, value}};
}

template <int Dim>
void AmrSystem<Dim>::begin_restart_transaction() {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AmrSystem restart requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  const long invalid = p_->restart_transaction || p_->external_step_transaction ? 1L : 0L;
  if (all_reduce_max(invalid, lane) != 0)
    throw std::logic_error(
        "AmrSystem restart transaction cannot overlap another restart or step transaction");

  std::unique_ptr<typename Impl::AcceptedSnapshot> candidate;
  std::exception_ptr snapshot_error;
  try {
    candidate = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("AMR restart snapshot failed on at least one MPI rank");
  }
  p_->restart_transaction = std::move(candidate);
}

template <int Dim>
void AmrSystem<Dim>::commit_restart_transaction() {
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AmrSystem restart commit requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  const long inactive = p_->restart_transaction ? 0L : 1L;
  if (all_reduce_max(inactive, lane) != 0)
    throw std::logic_error("AmrSystem has no collective restart transaction to commit");
  p_->restart_transaction.reset();
}

template <int Dim>
void AmrSystem<Dim>::rollback_restart_transaction() {
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AmrSystem restart rollback requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  const long inactive = p_->restart_transaction ? 0L : 1L;
  if (all_reduce_max(inactive, lane) != 0)
    throw std::logic_error("AmrSystem has no collective restart transaction to roll back");

  std::unique_ptr<typename Impl::AcceptedSnapshot> snapshot = std::move(p_->restart_transaction);
  std::exception_ptr restore_error;
  try {
    snapshot->restore(*p_);
    p_->program.resync_after_restart_rollback("AmrSystem::rollback_restart_transaction:");
  } catch (...) {
    restore_error = std::current_exception();
  }
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::runtime_error("AMR restart rollback lost its prepared hierarchy lane");
  const ExecutionLane& restored_lane = *p_->prepared_hierarchy->lane;
  if (all_reduce_max(restore_error ? 1L : 0L, restored_lane) != 0) {
    if (restored_lane.size() == 1 && restore_error)
      std::rethrow_exception(restore_error);
    throw std::runtime_error("AMR restart rollback failed on at least one MPI rank");
  }
}

template <int Dim>
void AmrSystem<Dim>::preflight_regrid_on_restart() {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR restart regrid preflight requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  const long invalid = !p_->restart_transaction || p_->external_step_transaction ? 1L : 0L;
  if (all_reduce_max(invalid, lane) != 0)
    throw std::logic_error(
        "AmrSystem restart regrid preflight requires one active restart transaction");
  std::exception_ptr preflight_error;
  try {
    p_->program.preflight_regrid_on_restart("AmrSystem::preflight_regrid_on_restart:");
  } catch (...) {
    preflight_error = std::current_exception();
  }
  if (all_reduce_max(preflight_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && preflight_error)
      std::rethrow_exception(preflight_error);
    throw std::runtime_error("AMR restart regrid preflight failed on at least one MPI rank");
  }
}

template <int Dim>
void AmrSystem<Dim>::regrid_on_restart() {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR restart regrid requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  const long invalid = !p_->restart_transaction || p_->external_step_transaction ? 1L : 0L;
  if (all_reduce_max(invalid, lane) != 0)
    throw std::logic_error("AmrSystem restart regrid requires one active restart transaction");
  const std::uint64_t prior_topology_epoch = p_->engine->topology_epoch();
  std::exception_ptr regrid_error;
  try {
    p_->program.regrid_on_restart("AmrSystem::regrid_on_restart:");
    p_->refresh_prepared_hierarchy();
    if (p_->engine->topology_epoch() == prior_topology_epoch)
      throw std::runtime_error("AmrSystem restart regrid authority did not publish a new topology");
  } catch (...) {
    regrid_error = std::current_exception();
  }
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::runtime_error("AMR restart regrid lost its prepared hierarchy lane");
  const ExecutionLane& published_lane = *p_->prepared_hierarchy->lane;
  if (all_reduce_max(regrid_error ? 1L : 0L, published_lane) != 0) {
    if (published_lane.size() == 1 && regrid_error)
      std::rethrow_exception(regrid_error);
    throw std::runtime_error("AMR restart regrid failed on at least one MPI rank");
  }
}

template <int Dim>
int AmrSystem<Dim>::checkpoint_regrid_count() const {
  return p_->checkpoint_regrid_count_value;
}

template <int Dim>
std::uint64_t AmrSystem<Dim>::checkpoint_topology_epoch() const {
  return p_->engine ? p_->engine->topology_epoch() : 0;
}

template <int Dim>
void AmrSystem<Dim>::restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch) {
  p_->ensure_engine();
  if (!p_->restart_transaction)
    throw std::logic_error("AmrSystem checkpoint counters require one active restart transaction");
  if (regrid_count < 0)
    throw std::invalid_argument("AmrSystem checkpoint regrid count must be non-negative");

  ExactContractBuilder request;
  request.text("pops.amr-system.restore-checkpoint-counters")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{regrid_count})
      .scalar(topology_epoch);
  const std::string request_contract = std::move(request).release();
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-checkpoint-counters"), std::string_view(request_contract)}}))
    throw std::invalid_argument("AMR checkpoint counters differ between MPI ranks");

  using publication_type = typename Impl::engine_type::PreparedRestorePublication;
  std::optional<publication_type> publication;
  std::exception_ptr preparation_error;
  try {
    auto snapshot = p_->engine->snapshot();
    snapshot.topology_epoch = topology_epoch;
    snapshot.exact_spatial_contract = runtime::amr::detail::exact_runtime_spatial_contract(
        p_->engine->spatial_identity(), snapshot.hierarchy, snapshot.topology_epoch,
        snapshot.materialization_generation);
    publication.emplace(p_->engine->prepare_restore_publication(snapshot));
  } catch (...) {
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AMR checkpoint counter publication failed to prepare collectively");
  }
  p_->engine->publish_prepared_restore(std::move(*publication));
  p_->checkpoint_regrid_count_value = regrid_count;
  p_->refresh_prepared_hierarchy();
}

template <int Dim>
void AmrSystem<Dim>::install_program_step(std::function<void(double)> step) {
  p_->program.install_unverified_step(std::move(step));
  p_->program_flux_expression_budget.reset();
}

template <int Dim>
void AmrSystem<Dim>::install_program_hierarchy_refresh(std::function<void()> refresh) {
  p_->program.install_hierarchy_refresh(std::move(refresh), "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::install_program_restart_hooks(std::function<void()> preflight,
                                                   std::function<void()> regrid,
                                                   std::function<void()> resync) {
  p_->program.install_restart_hooks(std::move(preflight), std::move(regrid), std::move(resync),
                                    "AmrSystem");
}

template <int Dim>
POPS_EXPORT void AmrSystem<Dim>::install_program(const std::string& so_path) {
  require_amr_assembling(p_->lifecycle, "install_program");
  if (p_->engine)
    throw std::logic_error(
        "AmrSystem::install_program must run before the AMR runtime is materialized");

  using install_type = void (*)(AmrSystem<Dim>*);
  using dt_bound_type = Real (*)(AmrSystem<Dim>*, Real);
  using boundary_install_type = void (*)(AmrSystem<Dim>*);
  using flux_expression_flag_type = bool (*)();
  using flux_expression_count_type = int (*)();
  using flux_expression_bound_type = std::uint64_t (*)(int);
  pops::dynlib::handle handle{};
  install_type install = nullptr;
  dt_bound_type dt_bound = nullptr;
  boundary_install_type install_boundaries = nullptr;
  bool program_has_dt_bound = false;
  bool program_has_flux_expression = false;
  std::string installed_hash;
  std::vector<PreparedAmrProgramFluxExpressionBlockBudget> flux_expression_blocks;
  std::vector<runtime::program::ProgramOperatorAuthority> operator_authorities;
  std::vector<runtime::program::ProgramHistoryReplayAuthority> history_replay_authorities;
  std::vector<int> program_block_map;
  std::map<int, std::vector<double>> program_param_defaults;
  std::exception_ptr preparation_error;

  try {
#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&detail::abi_key_string), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
#endif
    handle = pops::dynlib::open(so_path);
    if (!pops::dynlib::valid(handle))
      throw std::runtime_error("AmrSystem::install_program: cannot load '" + so_path +
                               "': " + pops::dynlib::last_error());

    auto key =
        reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_program_abi_key"));
    if (!key)
      throw std::runtime_error(
          "AmrSystem::install_program: pops_program_abi_key is missing; regenerate the artifact");
    const std::string module_key = detail::abi_key_string();
    if (std::string(key()) != module_key)
      throw std::runtime_error(
          "AmrSystem::install_program: compiled Program ABI differs from the native module");

    auto route_manifest = reinterpret_cast<const char* (*)()>(
        pops::dynlib::sym(handle, "pops_program_route_manifest"));
    if (!route_manifest)
      throw std::runtime_error(
          "AmrSystem::install_program: pops_program_route_manifest is missing; regenerate the "
          "artifact");
    const char* route_data = route_manifest();
    if (route_data == nullptr || route_data[0] == '\0')
      throw std::runtime_error(
          "AmrSystem::install_program: pops_program_route_manifest returned empty data");
    pops::verify_route_manifest(std::string(route_data), "install_program");

    operator_authorities = runtime::program::read_program_operator_authorities(handle);
    history_replay_authorities = runtime::program::read_program_history_replay_authorities(handle);
    install = reinterpret_cast<install_type>(pops::dynlib::sym(handle, "pops_install_program_amr"));
    if (!install)
      throw std::runtime_error(
          "AmrSystem::install_program: pops_install_program_amr is missing; compile the Program "
          "with target='amr_system'");

    const auto metadata = runtime::program::read_module_metadata(handle);
    const std::vector<std::string> runtime_blocks = block_names();
    std::string configured_solver;
    if (!p_->default_field_slot.empty()) {
      const auto field = p_->field_plans.find(p_->default_field_slot);
      if (field == p_->field_plans.end())
        throw std::logic_error("AmrSystem default field route is not materialized");
      configured_solver = field->second.solver_route;
    }
    for (const auto& operation : metadata.operators) {
      for (const std::string& required : runtime::program::required_blocks(operation.requirements))
        if (std::find(runtime_blocks.begin(), runtime_blocks.end(), required) ==
            runtime_blocks.end())
          throw std::runtime_error("operator '" + operation.name + "' requires block instance '" +
                                   required + "'");
      const std::string required_solver = runtime::program::required_solver(operation.requirements);
      if (!required_solver.empty() && required_solver != configured_solver)
        throw std::runtime_error("field operator '" + operation.name + "' requires solver '" +
                                 required_solver + "'");
    }

    auto has_dt =
        reinterpret_cast<bool (*)()>(pops::dynlib::sym(handle, "pops_program_has_dt_bound"));
    dt_bound =
        reinterpret_cast<dt_bound_type>(pops::dynlib::sym(handle, "pops_program_dt_bound_amr"));
    program_has_dt_bound = has_dt && has_dt();
    if (program_has_dt_bound && !dt_bound)
      throw std::runtime_error(
          "AmrSystem::install_program: Program declares a dt bound but "
          "pops_program_dt_bound_amr is missing");
    auto hash = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(handle, "pops_program_hash"));
    installed_hash = hash ? std::string(hash()) : std::string();
    install_boundaries = reinterpret_cast<boundary_install_type>(
        pops::dynlib::sym(handle, "pops_install_field_boundaries_amr"));

    using count_type = int (*)();
    using integer_type = int (*)(int);
    auto block_count =
        reinterpret_cast<count_type>(pops::dynlib::sym(handle, "pops_program_block_count"));
    auto block_name = reinterpret_cast<const char* (*)(int)>(
        pops::dynlib::sym(handle, "pops_program_block_name"));
    if (!block_count || !block_name)
      throw std::runtime_error(
          "AmrSystem::install_program: the exact Program block identity table is missing");
    const int count = block_count();
    if (count < 0)
      throw std::runtime_error("AmrSystem::install_program: Program block count is negative");
    program_block_map.assign(static_cast<std::size_t>(count), -1);
    for (int program = 0; program < count; ++program) {
      const char* raw_name = block_name(program);
      if (raw_name == nullptr || raw_name[0] == '\0')
        throw std::runtime_error("AmrSystem::install_program: Program block identity is empty");
      const auto found = std::find(runtime_blocks.begin(), runtime_blocks.end(), raw_name);
      if (found == runtime_blocks.end())
        throw std::runtime_error("Program requires block instance '" + std::string(raw_name) +
                                 "', but simulation did not instantiate it");
      program_block_map[static_cast<std::size_t>(program)] =
          static_cast<int>(std::distance(runtime_blocks.begin(), found));
    }

    const auto has_flux_expression = reinterpret_cast<flux_expression_flag_type>(
        pops::dynlib::sym(handle, "pops_program_has_flux_expression"));
    const auto flux_budget_count = reinterpret_cast<flux_expression_count_type>(
        pops::dynlib::sym(handle, "pops_program_flux_expression_budget_count"));
    const auto rhs_basis_bound = reinterpret_cast<flux_expression_bound_type>(
        pops::dynlib::sym(handle, "pops_program_flux_rhs_basis_bound"));
    const auto coefficient_term_bound = reinterpret_cast<flux_expression_bound_type>(
        pops::dynlib::sym(handle, "pops_program_flux_coefficient_term_bound"));
    if (!has_flux_expression || !flux_budget_count || !rhs_basis_bound || !coefficient_term_bound)
      throw std::runtime_error(
          "AmrSystem::install_program: exact flux-expression budget metadata is missing; "
          "regenerate the artifact");
    program_has_flux_expression = has_flux_expression();
    const int flux_blocks = flux_budget_count();
    if (flux_blocks < 0)
      throw std::runtime_error(
          "AmrSystem::install_program: flux-expression budget count is negative");
    if (flux_blocks != count)
      throw std::runtime_error(
          "AmrSystem::install_program: flux-expression budget count differs from the Program "
          "block count");
    flux_expression_blocks.reserve(static_cast<std::size_t>(flux_blocks));
    bool any_flux_expression_block = false;
    for (int program = 0; program < flux_blocks; ++program) {
      const std::uint64_t rhs = rhs_basis_bound(program);
      const std::uint64_t coefficients = coefficient_term_bound(program);
      if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (rhs > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            coefficients > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
          throw std::overflow_error(
              "AmrSystem::install_program: flux-expression bound exceeds size_t");
      }
      const std::size_t prepared_rhs = static_cast<std::size_t>(rhs);
      const std::size_t prepared_coefficients = static_cast<std::size_t>(coefficients);
      const bool active = prepared_rhs != 0 || prepared_coefficients != 0;
      if (active && (prepared_rhs == 0 || prepared_coefficients == 0))
        throw std::runtime_error(
            "AmrSystem::install_program: each active flux-expression block requires both bounds");
      any_flux_expression_block = any_flux_expression_block || active;
      if (prepared_rhs > std::numeric_limits<std::size_t>::max() - prepared_coefficients)
        throw std::overflow_error(
            "AmrSystem::install_program: flux-expression budget sum overflows size_t");
      flux_expression_blocks.push_back({prepared_rhs, prepared_coefficients});
    }
    if (program_has_flux_expression != any_flux_expression_block)
      throw std::runtime_error(
          "AmrSystem::install_program: flux-expression flag differs from its per-block budgets");

    auto parameter_count =
        reinterpret_cast<count_type>(pops::dynlib::sym(handle, "pops_program_param_count"));
    auto parameter_block =
        reinterpret_cast<integer_type>(pops::dynlib::sym(handle, "pops_program_param_block"));
    auto parameter_index =
        reinterpret_cast<integer_type>(pops::dynlib::sym(handle, "pops_program_param_index"));
    auto parameter_default =
        reinterpret_cast<double (*)(int)>(pops::dynlib::sym(handle, "pops_program_param_default"));
    if (parameter_count || parameter_block || parameter_index || parameter_default) {
      if (!parameter_count || !parameter_block || !parameter_index || !parameter_default)
        throw std::runtime_error(
            "AmrSystem::install_program: Program parameter table is incomplete");
      const int parameters = parameter_count();
      if (parameters < 0)
        throw std::runtime_error("AmrSystem::install_program: Program parameter count is negative");
      for (int ordinal = 0; ordinal < parameters; ++ordinal) {
        const int block = parameter_block(ordinal);
        const int index = parameter_index(ordinal);
        if (block < 0 || block >= count || index < 0)
          throw std::runtime_error(
              "AmrSystem::install_program: Program parameter route is out of range");
        std::vector<double>& values = program_param_defaults[block];
        if (static_cast<int>(values.size()) <= index)
          values.resize(static_cast<std::size_t>(index) + 1, 0.0);
        values[static_cast<std::size_t>(index)] = parameter_default(ordinal);
      }
    }
  } catch (...) {
    preparation_error = std::current_exception();
  }

  if (all_reduce_max(preparation_error ? 1L : 0L) != 0) {
    pops::dynlib::close(handle);
    if (n_ranks() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("AmrSystem::install_program preparation failed collectively");
  }

  using artifact_snapshot_type = decltype(p_->program.capture_artifact_step_install());
  std::optional<artifact_snapshot_type> previous_install;
  std::optional<typename Impl::AcceptedSnapshot> previous_runtime;
  std::map<std::string, std::optional<CompiledFieldBoundaryKernel<Dim>>> previous_boundaries;
  std::exception_ptr snapshot_error;
  try {
    previous_install.emplace(p_->program.capture_artifact_step_install());
    previous_runtime.emplace(*p_);
    for (const auto& [slot, plan] : p_->field_plans)
      previous_boundaries.emplace(slot, plan.boundary_kernel);
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L) != 0) {
    pops::dynlib::close(handle);
    if (n_ranks() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("AmrSystem::install_program rollback snapshot failed collectively");
  }

  std::optional<PreparedAmrProgramFluxExpressionBudget> prepared_flux_expression_budget;
  std::exception_ptr installation_error;
  try {
    p_->program.reset_artifact_candidate_state();
    p_->program.block_map_ = program_block_map;
    p_->program.block_params_.clear();
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program.operator_authorities_ = operator_authorities;

    p_->ensure_engine();
    install(this);
    p_->program.require_exact_artifact_step_install(*previous_install,
                                                    "AmrSystem::install_program:");
    if (!p_->program.hierarchy_refresh_)
      throw std::runtime_error(
          "AmrSystem::install_program: artifact lacks its hierarchy refresh hook");
    if (!p_->program.restart_regrid_preflight_ || !p_->program.restart_regrid_ ||
        !p_->program.restart_resync_)
      throw std::runtime_error(
          "AmrSystem::install_program: artifact lacks its restart preflight/regrid/resync hooks");
    if (!p_->prepared_program_block_map)
      throw std::logic_error(
          "AmrSystem::install_program: exact Program block map was not materialized");
    prepared_flux_expression_budget.emplace(p_->prepare_program_flux_expression_budget(
        installed_hash, std::move(flux_expression_blocks), *p_->prepared_program_block_map,
        program_has_flux_expression, *p_->engine, *p_->multiblock_hierarchy));

    p_->program.block_map_ = std::move(program_block_map);
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program.operator_authorities_ = std::move(operator_authorities);
    p_->program.history_replay_authorities_ = std::move(history_replay_authorities);
    p_->program.installed_hash_ = installed_hash;
    if (program_has_dt_bound) {
      AmrSystem<Dim>* self = this;
      p_->program.dt_bound_ = [self, dt_bound](Real cfl) { return dt_bound(self, cfl); };
    }
    p_->program.artifact_backed_ = true;
    static_assert(std::is_nothrow_move_assignable_v<decltype(p_->program_flux_expression_budget)>);
    p_->program_flux_expression_budget = std::move(prepared_flux_expression_budget);
    if (install_boundaries)
      install_boundaries(this);
    p_->program.refresh_hierarchy_state("AmrSystem::install_program");
  } catch (...) {
    installation_error = std::current_exception();
  }

  if (all_reduce_max(installation_error ? 1L : 0L) != 0) {
    p_->program.rollback_artifact_step_install(std::move(*previous_install));
    for (auto& [slot, plan] : p_->field_plans) {
      plan.discard_materialization();
      const auto boundary = previous_boundaries.find(slot);
      if (boundary != previous_boundaries.end())
        plan.boundary_kernel = std::move(boundary->second);
    }
    p_->prepared_hierarchy.reset();
    p_->prepared_program_block_map.reset();
    p_->multiblock_hierarchy.reset();
    p_->engine.reset();
    p_->pending_provider_restore.reset();
    p_->pending_provider_registry_restore.reset();
    p_->resolved_tagging.reset();
    p_->tagging_plan.reset();
    p_->component_tagging_plan.reset();
    p_->bootstrap_transaction.reset();
    previous_runtime->restore(*p_);
    pops::dynlib::close(handle);
    if (n_ranks() == 1 && installation_error)
      std::rethrow_exception(installation_error);
    throw std::runtime_error("AmrSystem::install_program installation failed collectively");
  }
  // The installed closures point into the authenticated DSO, which therefore remains loaded.
}

template <int Dim>
void AmrSystem<Dim>::set_program_cadence(int substeps, int stride) {
  require_amr_assembling(p_->lifecycle, "set_program_cadence");
  p_->program.set_cadence(substeps, stride, "AmrSystem");
}

template <int Dim>
int AmrSystem<Dim>::program_substeps() const {
  return p_->program.substeps_;
}

template <int Dim>
int AmrSystem<Dim>::program_stride() const {
  return p_->program.stride_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_dt() const {
  return p_->program.cadence_window_dt_;
}

template <int Dim>
int AmrSystem<Dim>::program_cadence_window_steps() const {
  return p_->program.cadence_window_steps_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_start_time() const {
  return p_->program.cadence_window_start_time_;
}

template <int Dim>
double AmrSystem<Dim>::program_last_dt() const {
  return static_cast<double>(p_->program.last_dt_);
}

template <int Dim>
void AmrSystem<Dim>::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                    double window_start_time,
                                                    double accepted_last_dt, double accepted_time,
                                                    int macro_step) {
  p_->program.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                     accepted_last_dt, accepted_time, macro_step, "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_block_map(const std::vector<int>& program_to_runtime) {
  if (program_to_runtime.size() != p_->blocks.size())
    throw std::invalid_argument(
        "AmrSystem Program block map must cover every prepared block exactly once");
  for (std::size_t program = 0; program < program_to_runtime.size(); ++program) {
    const int block = program_to_runtime[program];
    if (block < 0 || block >= static_cast<int>(p_->blocks.size()))
      throw std::out_of_range("AmrSystem Program block map is out of range");
    for (std::size_t previous = 0; previous < program; ++previous)
      if (program_to_runtime[previous] == block)
        throw std::invalid_argument("AmrSystem Program block map contains duplicate routes");
  }
  p_->program.block_map_ = program_to_runtime;
  p_->program_flux_expression_budget.reset();
  if (p_->multiblock_hierarchy)
    p_->prepare_program_block_map();
}

template <int Dim>
const std::vector<int>& AmrSystem<Dim>::program_block_map() const {
  return p_->program.block_map_;
}

template <int Dim>
std::string AmrSystem<Dim>::installed_program_hash() const {
  return p_->program.installed_hash_;
}

template <int Dim>
void AmrSystem<Dim>::seed_program_params(int block, const std::vector<double>& defaults) {
  p_->program.seed_params(block, defaults);
}

template <int Dim>
void AmrSystem<Dim>::set_program_params(int block, const std::vector<double>& values) {
  p_->program.set_params(block, values, "AmrSystem");
}

template <int Dim>
RuntimeParams AmrSystem<Dim>::program_params(int block) const {
  return p_->program.params(block);
}

template <int Dim>
runtime::amr::AmrRuntime<Dim>* AmrSystem<Dim>::engine() const {
  p_->ensure_engine();
  return p_->engine.get();
}

template <int Dim>
bool AmrSystem<Dim>::uses_runtime_engine() const {
  return static_cast<bool>(p_->engine);
}

template <int Dim>
runtime::program::Profiler& AmrSystem<Dim>::profiler_handle() {
  return p_->program.profiler_;
}

template <int Dim>
runtime::program::ProgramRuntimeState<Dim>& AmrSystem<Dim>::program_runtime_state_() {
  return p_->program;
}

template <int Dim>
void AmrSystem<Dim>::record_program_diagnostic(const std::string& name, double value) {
  p_->program.record_diagnostic(name, static_cast<Real>(value));
}

template <int Dim>
void AmrSystem<Dim>::record_program_balance_term(const std::string& route, const std::string& term,
                                                 double value) {
  p_->program.record_balance_term(route, term, static_cast<Real>(value), "AmrSystem");
}

template <int Dim>
bool AmrSystem<Dim>::program_balance_consumer_is_due(const std::string& contract,
                                                     const std::string& route, int every_n) const {
  return p_->program.balance_consumer_is_due(contract, route, every_n, "AmrSystem");
}

template <int Dim>
double AmrSystem<Dim>::program_diagnostic(const std::string& name) const {
  return static_cast<double>(p_->program.diagnostic(name, "AmrSystem"));
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::program_diagnostics() const {
  std::map<std::string, double> result;
  for (const auto& [name, value] : p_->program.diagnostics())
    result.emplace(name, static_cast<double>(value));
  return result;
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::accepted_balance_terms(
    const std::string& route) const {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error(
        "AmrSystem::_accepted_balance_terms requires an active uncommitted external step "
        "transaction");
  std::map<std::string, double> result;
  for (const auto& [term, value] : p_->program.accepted_balance_terms(route, "AmrSystem"))
    result.emplace(term, static_cast<double>(value));
  return result;
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::selected_accepted_balance_terms(
    const std::string& route, const std::string& block, int component,
    const std::vector<int>& levels, const std::vector<std::string>& automatic_terms) const {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error(
        "AmrSystem::_selected_accepted_balance_terms requires an active uncommitted external step "
        "transaction");
  p_->ensure_engine();
  const typename Impl::BlockSpec& selected = p_->block(block);
  if (component < 0 || component >= selected.ncomp)
    throw std::out_of_range(
        "AmrSystem::_selected_accepted_balance_terms component is out of range");
  const std::size_t level_count = p_->engine->hierarchy().num_levels();
  if (levels.empty() || std::any_of(levels.begin(), levels.end(), [&](int level) {
        return level < 0 || static_cast<std::size_t>(level) >= level_count;
      }))
    throw std::out_of_range(
        "AmrSystem::_selected_accepted_balance_terms level is out of active hierarchy range");
  const int runtime_block = static_cast<int>(&selected - p_->blocks.data());
  std::map<std::string, double> result;
  for (const auto& [term, value] : p_->program.selected_accepted_balance_terms(
           route, runtime_block, component, levels, automatic_terms, "AmrSystem"))
    result.emplace(term, static_cast<double>(value));
  return result;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_projection_report() {
  p_->program.begin_step_projection_report();
}

template <int Dim>
void AmrSystem<Dim>::note_step_projection(const std::string& name) {
  p_->program.note_step_projection(name);
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::consume_step_projections() {
  return p_->program.consume_step_projections();
}

template <int Dim>
void AmrSystem<Dim>::mark_bound() {
  seal_auxiliary_providers();
  if (p_->lifecycle.frozen())
    p_->lifecycle.to_bound();
  p_->ensure_engine();
  const auto& routes = p_->boundary_registry.state_routes();
  if (!routes.empty() && routes.size() != p_->blocks.size())
    throw std::runtime_error("AmrSystem state routes do not exactly cover its blocks");
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    if (!routes.empty() && !routes.contains(block.name))
      throw std::runtime_error("AmrSystem block lacks its exact state route");
    const auto* boundary = p_->boundary_registry.find_boundary(block.name);
    if (boundary == nullptr)
      continue;
    if (boundary->authority->ncomp() != block.ncomp ||
        boundary->authority->periodic_axes() != p_->cfg.periodicity)
      throw std::runtime_error("AmrSystem boundary differs from its block/domain contract");
    for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
      for (int axis = 0; axis < Dim; ++axis)
        if (p_->engine->hierarchy().state(level).ghosts()[axis] < boundary->required_depth)
          throw std::runtime_error("AmrSystem boundary depth exceeds level storage");
  }
  p_->lifecycle.to_bound();
}

template <int Dim>
std::string AmrSystem<Dim>::lifecycle_state() const {
  return p_->lifecycle.state(p_->macro_step);
}

template <int Dim>
Extent<Dim> AmrSystem<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double AmrSystem<Dim>::time() const {
  return p_->accepted_time;
}

template <int Dim>
int AmrSystem<Dim>::macro_step() const {
  return p_->macro_step;
}

template <int Dim>
void AmrSystem<Dim>::set_clock(double accepted_time, int macro_step) {
  if (!std::isfinite(accepted_time) || macro_step < 0)
    throw std::invalid_argument("AmrSystem clock requires finite time and non-negative step");
  p_->program.consume_cadence_clock_restore(accepted_time, macro_step, "AmrSystem");
  p_->accepted_time = accepted_time;
  p_->macro_step = macro_step;
}

template <int Dim>
void AmrSystem<Dim>::enable_profiling() {
  p_->program.profiler_.enable();
}

template <int Dim>
void AmrSystem<Dim>::disable_profiling() {
  p_->program.profiler_.disable();
}

template <int Dim>
bool AmrSystem<Dim>::is_profiling() const {
  return p_->program.profiler_.enabled();
}

template <int Dim>
void AmrSystem<Dim>::reset_profiling() {
  p_->program.profiler_.reset();
}

template <int Dim>
std::string AmrSystem<Dim>::profile_report() const {
  return p_->program.profiler_.report();
}

template <int Dim>
int AmrSystem<Dim>::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::block_names() const {
  std::vector<std::string> names;
  names.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks)
    names.push_back(block.name);
  return names;
}

template <int Dim>
EffectiveOptionsReport AmrSystem<Dim>::effective_options_report() const {
  if (!p_->embedded_boundary_configuration_contract.empty())
    p_->ensure_engine();
  EffectiveOptionsReport report;
  report.runtime = "amr_system";
  report.has_amr = true;
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->cfg.periodicity[axis]);
  report.poisson.solver = "geometric_mg";
  report.poisson.solver_option_schema = "pops.amr.field-solver-options.geometric-mg@1";
  if (!p_->embedded_boundary_configuration_contract.empty()) {
    report.eb.enabled = true;
    report.eb.geometry_mode = std::string(
        runtime::system::prepared_embedded_boundary_mode_name(p_->embedded_boundary_mode));
    report.eb.kappa_min = static_cast<double>(p_->embedded_boundary_thresholds.kappa_min);
    report.eb.face_open_eps = static_cast<double>(p_->embedded_boundary_thresholds.face_open_eps);
    report.eb.cut_theta_min = static_cast<double>(p_->embedded_boundary_thresholds.cut_theta_min);
    report.eb.semantic_digest = p_->embedded_boundary_semantic_digest;
    report.eb.materialization_digest =
        p_->prepared_hierarchy->embedded_boundary_materialization_digest;
    report.eb.generation = p_->embedded_boundary_generation;
  }
  report.blocks.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.n_ghost = block.required_ghost_depth;
    row.gamma = block.gamma;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.time = block.time;
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template <int Dim>
int AmrSystem<Dim>::n_levels() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().num_levels());
}

template <int Dim>
int AmrSystem<Dim>::max_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::configured_n_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::n_vars() {
  if (p_->blocks.size() != 1)
    throw std::runtime_error("AmrSystem::n_vars requires exactly one block");
  return p_->blocks.front().ncomp;
}

template <int Dim>
int AmrSystem<Dim>::block_n_vars(const std::string& name) {
  return p_->block(name).ncomp;
}

template <int Dim>
int AmrSystem<Dim>::n_patches() {
  p_->ensure_engine();
  const auto& hierarchy = p_->engine->hierarchy();
  return static_cast<int>(hierarchy.layout(hierarchy.num_levels() - 1).patches().size());
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::patch_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::output_geometry_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
int AmrSystem<Dim>::coarse_local_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().state(0).local_size());
}

template <int Dim>
int AmrSystem<Dim>::coarse_total_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().layout(0).patches().size());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state(int level) {
  return level_state_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state_global(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  const auto& hierarchy = p_->engine->hierarchy();
  return gather_field(hierarchy.state(static_cast<std::size_t>(level)),
                      hierarchy.layout(static_cast<std::size_t>(level)).domain(),
                      hierarchy.state(static_cast<std::size_t>(level)).ncomp());
}

template <int Dim>
void AmrSystem<Dim>::set_level_state(int level, const std::vector<double>& state) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  auto& hierarchy = p_->engine->hierarchy();
  write_field(hierarchy.state(static_cast<std::size_t>(level)),
              hierarchy.layout(static_cast<std::size_t>(level)).domain(), state,
              hierarchy.state(static_cast<std::size_t>(level)).ncomp());
  p_->discard_level_evaluations();
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state_global(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state_global(level);
}

template <int Dim>
void AmrSystem<Dim>::set_block_level_state(const std::string& name, int level,
                                           const std::vector<double>& state) {
  (void)p_->block(name);
  set_level_state(level, state);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_local_pieces(const std::string& name,
                                                                        int level) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem output level is out of range");
  const MultiFab<Dim>& state = p_->engine->hierarchy().state(static_cast<std::size_t>(level));
  return output_local_pieces(state, level, state.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_embedded_boundary_local_pieces(
    const std::string& name, int level) {
  p_->ensure_engine();
  if (level < 0 ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->embedded_boundary.size())
    throw std::out_of_range("AmrSystem embedded-boundary output level is out of range");
  const auto& embedded = p_->prepared_hierarchy->embedded_boundary[static_cast<std::size_t>(level)];
  if (!embedded)
    throw std::logic_error("AmrSystem has no prepared embedded-boundary geometry");
  const MultiFab<Dim>* field = nullptr;
  if (name == "pops_active")
    field = &embedded->active_mask();
  else if (name == "pops_phi")
    field = &embedded->phi();
  else if (name == "pops_kappa")
    field = &embedded->volume_fraction();
  else
    throw std::invalid_argument("unknown AMR embedded-boundary sidecar: " + name);
  return output_local_pieces(*field, level, field->distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_root_pieces(const ObserverMpiLane& lane,
                                                                       const std::string& name,
                                                                       int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "state", name, level),
      [&] { return output_state_local_pieces(name, level); });
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_embedded_boundary_root_pieces(
    const ObserverMpiLane& lane, const std::string& name, int level) {
  if (lane.size() == 1)
    return output_embedded_boundary_local_pieces(name, level);
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "embedded-boundary", name, level),
      [&] { return output_embedded_boundary_local_pieces(name, level); });
}

template <int Dim>
double AmrSystem<Dim>::composite_reduce(const std::string& name, const std::string& kind,
                                        int component,
                                        const std::vector<int>& requested_levels) const {
  const typename Impl::BlockSpec& block = p_->block(name);
  p_->ensure_engine();
  if (kind == "sum_all" || kind == "abs_sum_all" || kind == "sum_sq_all" || kind == "abs_max_all") {
    const std::string base = kind == "sum_all"       ? "sum"
                             : kind == "abs_sum_all" ? "abs_sum"
                             : kind == "sum_sq_all"  ? "sum_sq"
                                                     : "abs_max";
    double result = 0.0;
    for (int current = 0; current < block.ncomp; ++current) {
      const double value = composite_reduce(name, base, current, requested_levels);
      result = kind == "abs_max_all" ? std::max(result, value) : result + value;
    }
    return result;
  }
  if (component < 0 || component >= block.ncomp)
    throw std::out_of_range("AMR composite reduction component is outside the block state");
  const auto levels =
      resolved_composite_levels(p_->engine->hierarchy().num_levels(), requested_levels);
  std::vector<std::shared_ptr<const MultiFab<Dim>>> selected_coverage;
  const std::vector<std::shared_ptr<const MultiFab<Dim>>>* coverage =
      &p_->prepared_hierarchy->active_coverage;
  if (!requested_levels.empty()) {
    selected_coverage = Impl::prepare_active_coverage(*p_->engine, levels);
    coverage = &selected_coverage;
  }
  std::vector<runtime::amr::CompositeLevelView<Dim, memory_space>> views;
  views.reserve(levels.size());
  for (std::size_t position = 0; position < levels.size(); ++position) {
    const int level = levels[position];
    const std::size_t index = static_cast<std::size_t>(level);
    const Box<Dim>& domain = p_->engine->hierarchy().layout(index).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    std::array<Real, Dim> extent{};
    for (int axis = 0; axis < Dim; ++axis)
      extent[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    const MultiFab<Dim>* relative = nullptr;
    const auto& embedded = p_->prepared_hierarchy->embedded_boundary[index];
    if (embedded && embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::staircase)
      relative = &embedded->active_mask();
    else if (embedded &&
             embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell)
      relative = &embedded->volume_fraction();
    const std::size_t coverage_index = requested_levels.empty() ? index : position;
    views.push_back({&p_->engine->hierarchy().state(index), (*coverage)[coverage_index].get(),
                     extent, relative});
  }
  return static_cast<double>(
      runtime::amr::composite_reduce<Dim, memory_space>(
          views, component, composite_reduction_kind(kind), *p_->prepared_hierarchy->lane)
          .value);
}

template <int Dim>
double AmrSystem<Dim>::composite_reduce_field(const std::string& provider_slot,
                                              const std::string& kind, int component,
                                              const std::vector<int>& requested_levels) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  const auto levels = resolved_composite_levels(plan.accepted_potential.size(), requested_levels);
  std::vector<std::shared_ptr<const MultiFab<Dim>>> selected_coverage;
  const std::vector<std::shared_ptr<const MultiFab<Dim>>>* coverage = &plan.active_coverage;
  if (!requested_levels.empty()) {
    selected_coverage = Impl::prepare_active_coverage(*p_->engine, levels);
    coverage = &selected_coverage;
  }
  const int ncomp = plan.accepted_potential.empty() ? 0 : plan.accepted_potential.front()->ncomp();
  if (component < 0 || component >= ncomp)
    throw std::out_of_range("AMR composite field component is outside the prepared field");
  if (kind == "sum_all" || kind == "abs_sum_all" || kind == "sum_sq_all" || kind == "abs_max_all") {
    const std::string base = kind == "sum_all"       ? "sum"
                             : kind == "abs_sum_all" ? "abs_sum"
                             : kind == "sum_sq_all"  ? "sum_sq"
                                                     : "abs_max";
    double result = 0.0;
    for (int current = 0; current < ncomp; ++current) {
      const double value = composite_reduce_field(slot, base, current, requested_levels);
      result = kind == "abs_max_all" ? std::max(result, value) : result + value;
    }
    return result;
  }
  std::vector<runtime::amr::CompositeLevelView<Dim, memory_space>> views;
  views.reserve(levels.size());
  for (std::size_t position = 0; position < levels.size(); ++position) {
    const int level = levels[position];
    const std::size_t index = static_cast<std::size_t>(level);
    const Box<Dim>& domain = p_->engine->hierarchy().layout(index).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    std::array<Real, Dim> extent{};
    for (int axis = 0; axis < Dim; ++axis)
      extent[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    const std::size_t coverage_index = requested_levels.empty() ? index : position;
    views.push_back(
        {plan.accepted_potential[index].get(), (*coverage)[coverage_index].get(), extent, nullptr});
  }
  return static_cast<double>(
      runtime::amr::composite_reduce<Dim, memory_space>(
          views, component, composite_reduction_kind(kind), *p_->prepared_hierarchy->lane)
          .value);
}

template <int Dim>
void AmrSystem<Dim>::set_hierarchy(const std::vector<AmrPatch<Dim>>& boxes) {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AmrSystem::set_hierarchy requires its prepared hierarchy lane");
  if (p_->prepared_hierarchy->lane->size() != 1)
    throw std::runtime_error(
        "AmrSystem::set_hierarchy under MPI requires rebuild_hierarchy with explicit owner ranks");
  rebuild_hierarchy(boxes, std::vector<int>(boxes.size(), 0));
}

template <int Dim>
void AmrSystem<Dim>::rebuild_hierarchy(const std::vector<AmrPatch<Dim>>& boxes,
                                       const std::vector<int>& owner_ranks) {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR hierarchy rebuild requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  std::vector<std::vector<Box<Dim>>> level_boxes;
  std::vector<std::vector<int>> level_owners;
  std::string request_contract;
  std::exception_ptr validation_error;
  try {
    if (boxes.size() != owner_ranks.size())
      throw std::invalid_argument(
          "AmrSystem::rebuild_hierarchy boxes and owner ranks must align exactly");
    int active_levels = 1;
    for (std::size_t index = 0; index < boxes.size(); ++index) {
      const AmrPatch<Dim>& patch = boxes[index];
      if (patch.level < 1 || patch.level >= p_->cfg.level_count || patch.box.empty())
        throw std::invalid_argument(
            "AmrSystem::rebuild_hierarchy requires non-empty configured fine-level patches");
      if (owner_ranks[index] < 0 || owner_ranks[index] >= lane.size())
        throw std::out_of_range(
            "AmrSystem::rebuild_hierarchy owner rank lies outside the execution lane");
      active_levels = std::max(active_levels, patch.level + 1);
    }
    level_boxes.resize(static_cast<std::size_t>(active_levels));
    level_owners.resize(static_cast<std::size_t>(active_levels));
    ExactContractBuilder request;
    request.text("pops.amr-system.rebuild-hierarchy")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(boxes.size()));
    for (std::size_t index = 0; index < boxes.size(); ++index) {
      const AmrPatch<Dim>& patch = boxes[index];
      const std::size_t level = static_cast<std::size_t>(patch.level);
      level_boxes[level].push_back(patch.box);
      level_owners[level].push_back(owner_ranks[index]);
      request.scalar(std::int32_t{patch.level});
      for (int axis = 0; axis < Dim; ++axis)
        request.scalar(std::int32_t{patch.box.lo[axis]});
      for (int axis = 0; axis < Dim; ++axis)
        request.scalar(std::int32_t{patch.box.hi[axis]});
      request.scalar(std::int32_t{owner_ranks[index]});
    }
    for (int level = 1; level < active_levels; ++level)
      if (level_boxes[static_cast<std::size_t>(level)].empty())
        throw std::invalid_argument(
            "AmrSystem::rebuild_hierarchy requires contiguous active fine levels");
    request_contract = std::move(request).release();
  } catch (...) {
    validation_error = std::current_exception();
  }
  if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && validation_error)
      std::rethrow_exception(validation_error);
    throw std::runtime_error("AMR hierarchy rebuild validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-rebuild-hierarchy"), std::string_view(request_contract)}}, lane))
    throw std::invalid_argument("AMR hierarchy rebuild request differs between MPI ranks");

  p_->execute_transaction([&] {
    using engine_type = typename Impl::engine_type;
    using hierarchy_type = typename engine_type::hierarchy_type;
    using level_type = typename engine_type::level_type;
    using field_type = typename engine_type::field_type;

    const auto& live_hierarchy = p_->engine->hierarchy();
    hierarchy_type candidate_hierarchy = live_hierarchy.truncated(0);
    const mesh::RankSpace<Dim>& rank_space = live_hierarchy.layout(0).distribution().rank_space();
    const Index<Dim> local_rank = live_hierarchy.state(0).local_rank();
    Box<Dim> level_domain = live_hierarchy.layout(0).domain();
    for (std::size_t level = 1; level < level_boxes.size(); ++level) {
      const auto ratio = refinement_ratio(p_->cfg.transition_ratios[level - 1]);
      level_domain = amr::hierarchy::refine_box(level_domain, ratio);
      mesh::BoxArray<Dim> patches(level_boxes[level]);
      std::vector<Index<Dim>> owners;
      owners.reserve(level_owners[level].size());
      for (int rank : level_owners[level])
        owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
      mesh::Distribution<Dim> distribution =
          mesh::Distribution<Dim>::partitioned(patches, rank_space, std::move(owners));
      const mesh::BoxArrayValidationBudget layout_budget{patches.size(),
                                                         checked_pair_count(patches.size())};
      amr::hierarchy::LevelLayout<Dim> layout(static_cast<int>(level), level_domain, patches,
                                              distribution, ratio, layout_budget);
      field_type state(patches, distribution, local_rank, live_hierarchy.state(0).ncomp(),
                       live_hierarchy.state(0).ghosts());
      state.set_val(Real(0));
      candidate_hierarchy =
          candidate_hierarchy.with_level(level_type(std::move(layout), std::move(state)));
    }

    auto candidate_engine = std::make_shared<engine_type>(
        std::move(candidate_hierarchy), p_->load_balance, p_->engine->spatial_identity());
    auto qualified = candidate_engine->snapshot();
    qualified.topology_epoch = runtime::amr::detail::next_generation(
        p_->engine->topology_epoch(), "AMR hierarchy topology epoch");
    qualified.materialization_generation = runtime::amr::detail::next_generation(
        p_->engine->materialization_generation(), "AMR hierarchy materialization generation");
    qualified.exact_spatial_contract = runtime::amr::detail::exact_runtime_spatial_contract(
        candidate_engine->spatial_identity(), qualified.hierarchy, qualified.topology_epoch,
        qualified.materialization_generation);
    candidate_engine->restore(qualified);
    std::vector<typename Impl::multiblock_type::AdditionalBlock> additional;
    additional.reserve(p_->blocks.size() - 1);
    for (std::size_t block = 1; block < p_->blocks.size(); ++block) {
      std::vector<field_type> levels;
      levels.reserve(candidate_engine->hierarchy().num_levels());
      for (std::size_t level = 0; level < candidate_engine->hierarchy().num_levels(); ++level) {
        const field_type& primary = candidate_engine->hierarchy().state(level);
        levels.emplace_back(primary.layout(), primary.distribution(), primary.local_rank(),
                            p_->blocks[block].ncomp, p_->blocks[block].ghosts);
        levels.back().set_val(Real(0));
      }
      additional.push_back({p_->blocks[block].name, std::move(levels)});
    }
    auto candidate_multiblock = std::make_unique<typename Impl::multiblock_type>(
        Impl::multiblock_type::prepare_collectively(candidate_engine, p_->blocks.front().name,
                                                    std::move(additional),
                                                    "pops.amr-system.multiblock/rebuild"));
    for (const auto& coupling : p_->prepared_couplings)
      candidate_multiblock->install_prepared_coupling_operator(coupling.provider_contract,
                                                               coupling.view, coupling.operation);
    candidate_multiblock->seal_couplings();
    std::unique_ptr<typename Impl::PreparedHierarchy> candidate_graph =
        p_->prepare_hierarchy_graph(*candidate_engine, *candidate_multiblock, nullptr);
    std::optional<typename Impl::multiblock_type::ProgramBlockMap> block_map_candidate =
        p_->prepare_program_block_map_candidate(*candidate_multiblock);
    std::optional<typename Impl::flux_expression_budget_type> flux_budget_candidate;
    if (p_->program_flux_expression_budget) {
      if (!block_map_candidate)
        throw std::logic_error(
            "AMR Program flux-expression budget lost its exact Program block map");
      const bool has_flux_expression =
          Impl::flux_expression_budget_is_active(p_->program_flux_expression_budget->blocks);
      flux_budget_candidate.emplace(p_->prepare_program_flux_expression_budget(
          p_->program_flux_expression_budget->program_hash,
          p_->program_flux_expression_budget->blocks, *block_map_candidate, has_flux_expression,
          *candidate_engine, *candidate_multiblock));
    }

    // Engine, carrier, graph, exact map and flux budget are fully qualified candidates.  The
    // following ownership moves are the publication boundary, so a hierarchy refresh cannot
    // observe the prior budget generation on the rebuilt carrier.
    static_assert(std::is_nothrow_move_assignable_v<decltype(p_->prepared_program_block_map)>);
    static_assert(std::is_nothrow_move_assignable_v<decltype(p_->program_flux_expression_budget)>);
    p_->engine.swap(candidate_engine);
    p_->multiblock_hierarchy.swap(candidate_multiblock);
    p_->prepared_hierarchy.swap(candidate_graph);
    p_->prepared_program_block_map = std::move(block_map_candidate);
    p_->program_flux_expression_budget = std::move(flux_budget_candidate);
    p_->pending_provider_restore.reset();
    p_->pending_provider_registry_restore.reset();
    for (auto& [slot, plan] : p_->field_plans) {
      (void)slot;
      plan.discard_materialization();
    }
    p_->active_field_slot.clear();
    p_->tagging_plan.reset();
    p_->component_tagging_plan.reset();
    p_->automatic_bootstrap_complete = true;
    p_->program.refresh_hierarchy_state("AmrSystem::rebuild_hierarchy");
  });
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::rematerialize_hierarchy_ownership(
    const std::vector<AmrPatch<Dim>>& boxes) {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR ownership rematerialization requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  std::vector<std::vector<Box<Dim>>> level_boxes;
  std::vector<std::vector<std::size_t>> source_indices;
  std::string request_contract;
  std::exception_ptr validation_error;
  try {
    int active_levels = 1;
    for (const AmrPatch<Dim>& patch : boxes) {
      if (patch.level < 1 || patch.level >= p_->cfg.level_count || patch.box.empty())
        throw std::invalid_argument(
            "AMR ownership rematerialization requires non-empty configured fine-level patches");
      active_levels = std::max(active_levels, patch.level + 1);
    }
    level_boxes.resize(static_cast<std::size_t>(active_levels));
    source_indices.resize(static_cast<std::size_t>(active_levels));
    ExactContractBuilder request;
    request.text("pops.amr-system.rematerialize-ownership")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(boxes.size()));
    for (std::size_t index = 0; index < boxes.size(); ++index) {
      const AmrPatch<Dim>& patch = boxes[index];
      const std::size_t level = static_cast<std::size_t>(patch.level);
      level_boxes[level].push_back(patch.box);
      source_indices[level].push_back(index);
      request.scalar(std::int32_t{patch.level});
      for (int axis = 0; axis < Dim; ++axis)
        request.scalar(std::int32_t{patch.box.lo[axis]});
      for (int axis = 0; axis < Dim; ++axis)
        request.scalar(std::int32_t{patch.box.hi[axis]});
    }
    for (int level = 1; level < active_levels; ++level)
      if (level_boxes[static_cast<std::size_t>(level)].empty())
        throw std::invalid_argument(
            "AMR ownership rematerialization requires contiguous active fine levels");
    request_contract = std::move(request).release();
  } catch (...) {
    validation_error = std::current_exception();
  }
  if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && validation_error)
      std::rethrow_exception(validation_error);
    throw std::runtime_error("AMR ownership rematerialization validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-rematerialize-ownership"), std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument("AMR ownership rematerialization request differs between ranks");
  if (boxes.empty())
    return {};

  const mesh::RankSpace<Dim> rank_space = process_rank_space<Dim>(lane);
  std::vector<int> result(boxes.size(), -1);
  for (std::size_t level = 1; level < level_boxes.size(); ++level) {
    const mesh::BoxArray<Dim> patches(level_boxes[level]);
    const parallel::LoadBalancePreparationBudget budget{patches.size(), rank_space.size(),
                                                        checked_layout_cells(patches)};
    const auto prepared = p_->load_balance->prepare(patches, rank_space, budget, {}, lane);
    const auto& distribution = prepared.plan().distribution();
    for (std::size_t patch = 0; patch < patches.size(); ++patch)
      result[source_indices[level][patch]] =
          static_cast<int>(rank_space.linear_rank(distribution.owner(patch)));
  }
  return result;
}

template <int Dim>
std::vector<std::uint8_t> AmrSystem<Dim>::rematerialize_program_accepted_state(
    const std::vector<std::vector<std::uint8_t>>& source_states,
    const std::vector<std::vector<int>>& source_level_owners,
    const std::vector<std::vector<int>>& target_level_owners) {
  p_->ensure_engine();
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR Program rematerialization requires its prepared hierarchy lane");
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  std::optional<runtime::program::AmrProgramAcceptedState<Dim>> accepted;
  std::string request_contract;
  std::exception_ptr validation_error;
  try {
    if (source_states.empty() ||
        source_states.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument(
          "AMR Program rematerialization requires one source image per recorded rank");
    const std::size_t levels = p_->engine->hierarchy().num_levels();
    if (source_level_owners.size() != levels || target_level_owners.size() != levels)
      throw std::invalid_argument(
          "AMR Program rematerialization ownership must cover every active level");
    const int source_rank_count = static_cast<int>(source_states.size());
    for (std::size_t level = 0; level < levels; ++level) {
      if (source_level_owners[level].size() != target_level_owners[level].size())
        throw std::invalid_argument(
            "AMR Program source and target ownership name different patch counts");
      const auto& live = p_->engine->hierarchy().layout(level).distribution();
      const std::size_t live_owners = live.replicated() ? 0 : live.owners().size();
      if (target_level_owners[level].size() != live_owners)
        throw std::invalid_argument(
            "AMR Program target ownership differs from the live exact hierarchy");
      for (int owner : source_level_owners[level])
        if (owner < 0 || owner >= source_rank_count)
          throw std::out_of_range("AMR Program source owner rank is out of range");
      for (std::size_t patch = 0; patch < target_level_owners[level].size(); ++patch) {
        const int owner = target_level_owners[level][patch];
        if (owner < 0 || owner >= lane.size() ||
            owner != static_cast<int>(live.rank_space().linear_rank(live.owner(patch))))
          throw std::invalid_argument(
              "AMR Program target ownership differs from the published hierarchy");
      }
    }

    accepted.emplace(
        runtime::program::deserialize_amr_program_accepted_state<Dim>(source_states.front()));
    for (std::size_t rank = 1; rank < source_states.size(); ++rank) {
      const auto candidate =
          runtime::program::deserialize_amr_program_accepted_state<Dim>(source_states[rank]);
      if (runtime::program::serialize_amr_program_accepted_state(candidate) !=
          runtime::program::serialize_amr_program_accepted_state(*accepted))
        throw std::invalid_argument(
            "AMR Program rank-independent accepted images differ across source ranks");
    }
    if (accepted->level_clocks.size() != levels)
      throw std::invalid_argument(
          "AMR Program accepted clocks differ from the rematerialized hierarchy depth");
    accepted->spatial_contract = p_->engine->spatial_contract();
    accepted->topology_epoch = p_->engine->topology_epoch();
    accepted->materialization_generation = p_->engine->materialization_generation();
    if (accepted->temporal_partition.kind == runtime::program::TemporalPartitionKind::CellLocal)
      accepted->temporal_partition.topology_epoch = accepted->topology_epoch;
    const std::vector<std::uint8_t> bytes =
        runtime::program::serialize_amr_program_accepted_state(*accepted);
    ExactContractBuilder request;
    request.text("pops.amr-system.rematerialize-program-state")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(source_states.size()))
        .scalar(static_cast<std::uint64_t>(levels))
        .bytes(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    for (const auto& owners : source_level_owners) {
      request.scalar(static_cast<std::uint64_t>(owners.size()));
      for (int owner : owners)
        request.scalar(std::int32_t{owner});
    }
    for (const auto& owners : target_level_owners) {
      request.scalar(static_cast<std::uint64_t>(owners.size()));
      for (int owner : owners)
        request.scalar(std::int32_t{owner});
    }
    request_contract = std::move(request).release();
  } catch (...) {
    validation_error = std::current_exception();
  }
  if (all_reduce_max(validation_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && validation_error)
      std::rethrow_exception(validation_error);
    throw std::runtime_error("AMR Program rematerialization validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-rematerialize-program-state"),
            std::string_view(request_contract)}},
          lane))
    throw std::invalid_argument("AMR Program rematerialization request differs between ranks");
  runtime::program::require_collective_amr_program_checkpoint_consensus(*accepted, lane);
  return runtime::program::serialize_amr_program_accepted_state(*accepted);
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::level_owner_ranks(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem owner level is out of range");
  const auto& distribution =
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).distribution();
  if (distribution.replicated())
    throw std::runtime_error("replicated AMR levels have no unique owner ranks");
  std::vector<int> result;
  result.reserve(distribution.owners().size());
  for (const Index<Dim>& owner : distribution.owners())
    result.push_back(static_cast<int>(distribution.rank_space().linear_rank(owner)));
  return result;
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::history_names() const {
  if (!p_->engine)
    return {};
  std::vector<std::string> result;
  for (const auto& descriptor : p_->history_descriptors())
    result.push_back(descriptor.name);
  return result;
}

template <int Dim>
int AmrSystem<Dim>::history_depth(const std::string& name) const {
  p_->ensure_engine();
  return p_->history_descriptor(name).depth;
}

template <int Dim>
int AmrSystem<Dim>::history_ncomp(const std::string& name) const {
  p_->ensure_engine();
  return p_->history_descriptor(name).components;
}

template <int Dim>
bool AmrSystem<Dim>::history_initialized(const std::string& name) const {
  p_->ensure_engine();
  const auto keys = p_->history_level_keys(name);
  return std::all_of(keys.begin(), keys.end(), [&](const std::string& key) {
    const auto found = p_->program.hist_.initialized.find(key);
    return found != p_->program.hist_.initialized.end() && found->second;
  });
}

template <int Dim>
int AmrSystem<Dim>::history_fill_count(const std::string& name) const {
  p_->ensure_engine();
  const auto keys = p_->history_level_keys(name);
  int result = p_->history_descriptor(name).depth;
  for (const std::string& key : keys) {
    const auto found = p_->program.hist_.fill_count.find(key);
    if (found == p_->program.hist_.fill_count.end())
      throw std::logic_error("AMR history ring lacks its accepted fill count");
    result = std::min(result, found->second);
  }
  return result;
}

template <int Dim>
void AmrSystem<Dim>::set_history_initialized(const std::string& name, bool initialized) {
  p_->ensure_engine();
  const int fill = initialized ? p_->history_descriptor(name).depth : 0;
  for (const std::string& key : p_->history_level_keys(name)) {
    p_->program.hist_.initialized.at(key) = initialized;
    p_->program.hist_.fill_count.at(key) = fill;
    p_->program.hist_.store_pending.at(key) = false;
  }
}

template <int Dim>
void AmrSystem<Dim>::restore_history_fill_count(const std::string& name, int fill_count) {
  p_->ensure_engine();
  const int depth = p_->history_descriptor(name).depth;
  if (fill_count < 0 || fill_count > depth)
    throw std::invalid_argument("AMR history fill count lies outside its exact ring depth");
  for (const std::string& key : p_->history_level_keys(name)) {
    p_->program.hist_.fill_count.at(key) = fill_count;
    p_->program.hist_.initialized.at(key) = fill_count > 0;
    p_->program.hist_.store_pending.at(key) = false;
  }
}

template <int Dim>
void AmrSystem<Dim>::restore_history_metadata(const std::string& name, bool initialized,
                                              int fill_count) {
  p_->ensure_engine();
  const int depth = p_->history_descriptor(name).depth;
  if (fill_count < 0 || fill_count > depth)
    throw std::invalid_argument("AMR history fill count lies outside its exact ring depth");
  const auto keys = p_->history_level_keys(name);
  for (const std::string& key : keys) {
    if (!p_->program.hist_.initialized.contains(key) ||
        !p_->program.hist_.fill_count.contains(key) ||
        !p_->program.hist_.store_pending.contains(key))
      throw std::logic_error("AMR history publication metadata is incomplete");
  }
  for (const std::string& key : keys) {
    p_->program.hist_.initialized.at(key) = initialized;
    p_->program.hist_.fill_count.at(key) = fill_count;
    p_->program.hist_.store_pending.at(key) = false;
  }
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::history_global(const std::string& name, int slot) const {
  p_->ensure_engine();
  const auto descriptor = p_->history_descriptor(name);
  if (slot < 0 || slot >= descriptor.depth)
    throw std::out_of_range("AMR history slot lies outside its exact ring depth");
  if (!p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("AMR history gather requires its prepared hierarchy lane");
  const auto keys = p_->history_level_keys(name);
  std::vector<double> result;
  for (std::size_t level = 0; level < keys.size(); ++level) {
    const auto& ring = p_->program.hist_.histories.at(keys[level]);
    const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
    std::vector<double> values =
        gather_field(ring[static_cast<std::size_t>(slot)], domain, descriptor.components,
                     &*p_->prepared_hierarchy->lane);
    result.insert(result.end(), values.begin(), values.end());
  }
  return result;
}

template <int Dim>
void AmrSystem<Dim>::restore_history(const std::string& name, int slot,
                                     const std::vector<double>& values) {
  p_->ensure_engine();
  const auto descriptor = p_->history_descriptor(name);
  if (slot < 0 || slot >= descriptor.depth)
    throw std::out_of_range("AMR history slot lies outside its exact ring depth");
  const auto keys = p_->history_level_keys(name);
  std::vector<std::size_t> level_sizes;
  level_sizes.reserve(keys.size());
  std::size_t expected = 0;
  for (std::size_t level = 0; level < keys.size(); ++level) {
    const std::size_t cells = checked_cells(p_->engine->hierarchy().layout(level).domain());
    if (cells >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(descriptor.components))
      throw std::overflow_error("AMR history restore level size overflow");
    const std::size_t count = cells * static_cast<std::size_t>(descriptor.components);
    if (count > std::numeric_limits<std::size_t>::max() - expected)
      throw std::overflow_error("AMR history restore total size overflow");
    level_sizes.push_back(count);
    expected += count;
  }
  if (values.size() != expected)
    throw std::invalid_argument("AMR history restore payload has the wrong exact-ranked size");
  std::size_t offset = 0;
  for (std::size_t level = 0; level < keys.size(); ++level) {
    auto& field = p_->program.hist_.histories.at(keys[level])[static_cast<std::size_t>(slot)];
    const std::vector<double> level_values(
        values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + level_sizes[level]));
    write_field(field, p_->engine->hierarchy().layout(level).domain(), level_values,
                descriptor.components);
    offset += level_sizes[level];
  }
}

template <int Dim>
double AmrSystem<Dim>::history_slot_dt(const std::string& name, int slot) const {
  p_->ensure_engine();
  const int depth = p_->history_descriptor(name).depth;
  if (slot < 0 || slot >= depth)
    throw std::out_of_range("AMR history dt slot lies outside its exact ring depth");
  std::optional<Real> retained;
  for (const std::string& key : p_->history_level_keys(name)) {
    const auto& values = p_->program.hist_.slot_dt.at(key);
    if (values.size() != static_cast<std::size_t>(depth))
      throw std::logic_error("AMR history dt ledger differs from its exact ring depth");
    const Real candidate = values[static_cast<std::size_t>(slot)];
    if (retained && *retained != candidate)
      throw std::logic_error("AMR history dt differs between active hierarchy levels");
    retained = candidate;
  }
  return static_cast<double>(*retained);
}

template <int Dim>
void AmrSystem<Dim>::restore_history_slot_dt(const std::string& name, int slot, double dt) {
  p_->ensure_engine();
  const int depth = p_->history_descriptor(name).depth;
  if (slot < 0 || slot >= depth || !std::isfinite(dt) || dt < 0.0)
    throw std::invalid_argument("AMR history dt restore requires a valid slot and finite dt >= 0");
  const Real native_dt = static_cast<Real>(dt);
  if (!std::isfinite(static_cast<double>(native_dt)))
    throw std::overflow_error("AMR history dt exceeds native Real");
  for (const std::string& key : p_->history_level_keys(name))
    p_->program.hist_.slot_dt.at(key)[static_cast<std::size_t>(slot)] = native_dt;
}

template <int Dim>
int AmrSystem<Dim>::rebuild_history_slots(const std::string& name,
                                          const std::vector<int>& stored_slots) {
  p_->ensure_engine();
  const auto descriptor = p_->history_descriptor(name);
  if (!p_->program.step_)
    throw std::logic_error("AMR history replay requires an installed compiled Program");
  if (!p_->program.authorizes_history_replay(name, descriptor.depth))
    throw std::logic_error("AMR history selective replay lacks its exact artifact-owned authority");
  if (p_->program.substeps_ != 1 || p_->program.stride_ != 1)
    throw std::logic_error(
        "AMR history selective replay requires Program cadence (substeps=1, stride=1)");

  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  if (anchors.empty() || anchors.front() != 0 || anchors.back() != descriptor.depth - 1)
    throw std::invalid_argument(
        "AMR history selective replay requires exact newest and oldest anchors");
  for (const int anchor : anchors)
    if (anchor < 0 || anchor >= descriptor.depth)
      throw std::out_of_range("AMR history selective replay anchor is outside the ring");
  if (static_cast<int>(anchors.size()) == descriptor.depth) {
    p_->last_replay_regrid_steps.clear();
    return 0;
  }
  if (p_->cfg.regrid_every > 0)
    for (int slot = descriptor.depth - 2; slot >= 0; --slot) {
      const int cursor = p_->macro_step - 1 - slot;
      if (cursor > 0 && cursor % p_->cfg.regrid_every == 0)
        throw std::logic_error(
            "AMR history selective replay window contains a scheduled regrid; use dense safety "
            "storage");
    }

  const auto keys = p_->history_level_keys(name);
  std::vector<Real> dts = p_->program.hist_.slot_dt.at(keys.front());
  for (std::size_t anchor = 0; anchor + 1 < anchors.size(); ++anchor)
    for (int slot = anchors[anchor + 1] - 1; slot > anchors[anchor]; --slot)
      if (slot + 1 >= static_cast<int>(dts.size()) ||
          !std::isfinite(static_cast<double>(dts[static_cast<std::size_t>(slot + 1)])) ||
          !(dts[static_cast<std::size_t>(slot + 1)] > Real(0)))
        throw std::invalid_argument(
            "AMR history selective replay requires every omitted outgoing dt");

  typename Impl::AcceptedSnapshot accepted(*p_);
  std::vector<std::vector<MultiFab<Dim>>> reconstructed(static_cast<std::size_t>(descriptor.depth));
  std::vector<int> fired;
  try {
    for (std::size_t anchor = 0; anchor + 1 < anchors.size(); ++anchor) {
      const int newer = anchors[anchor];
      const int older = anchors[anchor + 1];
      accepted.restore(*p_);
      for (std::size_t level = 0; level < keys.size(); ++level)
        copy_full_field_in_place(
            p_->program.hist_.histories.at(keys[level])[static_cast<std::size_t>(older)],
            p_->engine->hierarchy().state(level));
      for (int slot = older - 1; slot > newer; --slot) {
        const int cursor = p_->macro_step - 1 - slot;
        p_->macro_step = cursor;
        p_->program.last_dt_ = dts[static_cast<std::size_t>(slot + 1)];
        const std::uint64_t epoch = p_->engine->topology_epoch();
        p_->program.run_balance_replay("AmrSystem::rebuild_history_slots", [&] {
          p_->program.step_(static_cast<double>(dts[static_cast<std::size_t>(slot + 1)]));
        });
        if (p_->engine->topology_epoch() != epoch)
          fired.push_back(cursor);
        auto& image = reconstructed[static_cast<std::size_t>(slot)];
        image.reserve(keys.size());
        for (std::size_t level = 0; level < keys.size(); ++level)
          image.push_back(p_->engine->hierarchy().state(level));
      }
    }
  } catch (...) {
    accepted.restore(*p_);
    throw;
  }
  accepted.restore(*p_);
  if (!fired.empty())
    throw std::logic_error(
        "AMR history selective replay changed topology; use dense safety storage");

  int recomputed = 0;
  for (int slot = 0; slot < descriptor.depth; ++slot) {
    if (std::binary_search(anchors.begin(), anchors.end(), slot))
      continue;
    const auto& image = reconstructed[static_cast<std::size_t>(slot)];
    if (image.size() != keys.size())
      throw std::logic_error("AMR history anchors do not bracket every omitted slot");
    for (std::size_t level = 0; level < keys.size(); ++level)
      copy_full_field_in_place(image[level], p_->program.hist_.histories.at(
                                                 keys[level])[static_cast<std::size_t>(slot)]);
    ++recomputed;
  }
  p_->last_replay_regrid_steps.clear();
  return recomputed;
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::last_replay_regrid_steps() const {
  return p_->last_replay_regrid_steps;
}

template <int Dim>
double AmrSystem<Dim>::mass() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return mass(p_->blocks.front().name);
}

template <int Dim>
double AmrSystem<Dim>::mass(const std::string& name) {
  return composite_reduce(name, "sum", 0);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return density(p_->blocks.front().name);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  return gather_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), 1);
}

template <int Dim>
std::vector<std::uint8_t> AmrSystem<Dim>::program_accepted_state() const {
  if (p_->program_accepted_bytes.empty() && !p_->program.artifact_backed_ && !p_->tagging_spec)
    return {};
  p_->ensure_engine();
  runtime::program::AmrProgramAcceptedState<Dim> state =
      p_->program_accepted_bytes.empty()
          ? p_->minimal_program_accepted_state()
          : runtime::program::deserialize_amr_program_accepted_state<Dim>(
                p_->program_accepted_bytes);
  if (p_->program_accepted_bytes.empty() || p_->program_accepted_bytes_runtime_owned)
    p_->requalify_runtime_owned_program_state(state);
  else
    runtime::program::require_live_amr_program_checkpoint(state, *p_->engine);
  return runtime::program::serialize_amr_program_accepted_state(state);
}

template <int Dim>
void AmrSystem<Dim>::copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const {
  state = program_accepted_state();
}

template <int Dim>
void AmrSystem<Dim>::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  if (p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
  p_->program_accepted_bytes = state;
  p_->program_accepted_bytes_runtime_owned = false;
  ++p_->program_accepted_revision;
}

template <int Dim>
void AmrSystem<Dim>::restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state) {
  p_->ensure_engine();
  std::optional<runtime::program::AmrProgramAcceptedState<Dim>> decoded;
  std::optional<runtime::amr::PersistentTaggingState<Dim>> decoded_tagging;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    decoded.emplace(runtime::program::deserialize_amr_program_accepted_state<Dim>(state));
    runtime::program::require_live_amr_program_checkpoint(*decoded, *p_->engine);
    if (p_->tagging_spec) {
      decoded_tagging.emplace(runtime::amr::PersistentTaggingState<Dim>::decode(
          decoded->tagging_hysteresis_state, p_->tagging_spec->min_cycles,
          p_->tagging_spec->provider_identity, p_->live_tagging_parent_domains()));
    } else {
      if (!decoded->tagging_hysteresis_state.empty())
        throw std::invalid_argument(
            "AMR checkpoint carries tagging hysteresis without a bound tagging authority");
      decoded_tagging.emplace();
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  if (all_reduce_max(local_failure, lane.communicator()) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR checkpoint tagging state failed validation on another MPI rank");
  }
  runtime::program::require_collective_amr_program_checkpoint_consensus(*decoded, lane);
  const long exhausted =
      p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max() ? 1L : 0L;
  if (all_reduce_max(exhausted, lane.communicator()) != 0)
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");

  std::vector<std::uint8_t> candidate_bytes;
  std::vector<std::uint8_t> previous_bytes;
  std::optional<runtime::amr::PersistentTaggingState<Dim>> previous_tagging;
  bool previous_runtime_owned = false;
  const std::uint64_t previous_revision = p_->program_accepted_revision;
  std::exception_ptr snapshot_error;
  try {
    candidate_bytes = state;
    previous_bytes = p_->program_accepted_bytes;
    previous_tagging.emplace(p_->tagging_state);
    previous_runtime_owned = p_->program_accepted_bytes_runtime_owned;
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L, lane.communicator()) != 0) {
    if (lane.size() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("AMR checkpoint accepted-state snapshot failed collectively");
  }

  static_assert(std::is_nothrow_swappable_v<std::vector<std::uint8_t>>);
  p_->program_accepted_bytes.swap(candidate_bytes);
  p_->program_accepted_bytes_runtime_owned = false;
  p_->tagging_state = std::move(*decoded_tagging);
  ++p_->program_accepted_revision;
  std::exception_ptr resync_error;
  try {
    p_->program.resync_after_restart_rollback("AmrSystem::restore_checkpoint_accepted_state");
  } catch (...) {
    resync_error = std::current_exception();
  }
  if (all_reduce_max(resync_error ? 1L : 0L, lane.communicator()) == 0)
    return;

  p_->program_accepted_bytes.swap(previous_bytes);
  p_->program_accepted_bytes_runtime_owned = previous_runtime_owned;
  p_->tagging_state = std::move(*previous_tagging);
  p_->program_accepted_revision = previous_revision;
  std::exception_ptr rollback_resync_error;
  try {
    p_->program.resync_after_restart_rollback(
        "AmrSystem::restore_checkpoint_accepted_state rollback");
  } catch (...) {
    rollback_resync_error = std::current_exception();
  }
  if (all_reduce_max(rollback_resync_error ? 1L : 0L, lane.communicator()) != 0)
    throw std::runtime_error("AMR checkpoint accepted-state rollback resynchronization failed");
  if (lane.size() == 1 && resync_error)
    std::rethrow_exception(resync_error);
  throw std::runtime_error("AMR checkpoint accepted-state resynchronization failed collectively");
}

template <int Dim>
void AmrSystem<Dim>::materialize_program_restart_histories(const std::vector<std::uint8_t>& state,
                                                           const std::vector<std::string>& names,
                                                           const std::vector<int>& depths,
                                                           const std::vector<int>& ncomps) {
  p_->ensure_engine();
  if (!p_->restart_transaction)
    throw std::logic_error("AMR Program restart histories require one active restart transaction");
  if (names.size() != depths.size() || names.size() != ncomps.size())
    throw std::invalid_argument(
        "AMR Program restart history names, depths and component counts must align");

  runtime::program::HistoryManager<Dim> candidate;
  std::string contract;
  std::exception_ptr preparation_error;
  try {
    const auto accepted = runtime::program::deserialize_amr_program_accepted_state<Dim>(state);
    runtime::program::require_live_amr_program_checkpoint(accepted, *p_->engine);
    if (accepted.histories.size() != names.size())
      throw std::invalid_argument(
          "AMR Program restart history registry differs from its accepted checkpoint");
    const auto& hierarchy = p_->engine->hierarchy();
    ExactContractBuilder exact;
    exact.text("pops.amr-system.restart-history-materialization")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(p_->engine->spatial_contract())
        .scalar(static_cast<std::uint64_t>(accepted.histories.size()));
    for (std::size_t index = 0; index < names.size(); ++index) {
      const auto& descriptor = accepted.histories[index];
      if (descriptor.name != names[index] || descriptor.depth != depths[index] ||
          descriptor.components != ncomps[index])
        throw std::invalid_argument(
            "AMR Program restart history arguments differ from the accepted descriptor");
      if (index != 0 && names[index - 1] >= names[index])
        throw std::invalid_argument("AMR Program restart history names must be uniquely ordered");
      if (descriptor.program_owner < 0 ||
          static_cast<std::size_t>(descriptor.program_owner) >= p_->program.block_map_.size())
        throw std::invalid_argument("AMR Program restart history has an invalid program owner");
      const int runtime_owner =
          p_->program.block_map_[static_cast<std::size_t>(descriptor.program_owner)];
      if (runtime_owner < 0 || static_cast<std::size_t>(runtime_owner) >= p_->blocks.size())
        throw std::logic_error("AMR Program restart history maps to an invalid runtime block");
      exact.text(descriptor.name)
          .scalar(std::int32_t{descriptor.program_owner})
          .text(descriptor.state_identity)
          .text(descriptor.space_identity)
          .text(descriptor.clock_identity)
          .text(descriptor.interpolation_identity)
          .scalar(std::int32_t{descriptor.depth})
          .scalar(std::int32_t{descriptor.components});
      for (std::size_t level = 0; level < hierarchy.num_levels(); ++level) {
        const std::string key = exact_amr_history_key(descriptor.name, static_cast<int>(level));
        const MultiFab<Dim>& prototype =
            p_->block_state(static_cast<std::size_t>(runtime_owner), level);
        std::vector<MultiFab<Dim>> ring;
        ring.reserve(static_cast<std::size_t>(descriptor.depth));
        for (int slot = 0; slot < descriptor.depth; ++slot) {
          MultiFab<Dim> field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                              descriptor.components, prototype.ghosts());
          field.set_val(Real(0));
          ring.push_back(std::move(field));
        }
        candidate.histories.emplace(key, std::move(ring));
        candidate.depth.emplace(key, descriptor.depth);
        candidate.initialized.emplace(key, false);
        candidate.fill_count.emplace(key, 0);
        candidate.store_pending.emplace(key, false);
        candidate.owner.emplace(key, runtime_owner);
        candidate.state_identity.emplace(key, descriptor.state_identity);
        candidate.space_identity.emplace(key, descriptor.space_identity);
        candidate.clock_identity.emplace(key, descriptor.clock_identity);
        candidate.interpolation_identity.emplace(key, descriptor.interpolation_identity);
        candidate.slot_dt.emplace(
            key, std::vector<Real>(static_cast<std::size_t>(descriptor.depth), Real(0)));
      }
    }
    contract = std::move(exact).release();
  } catch (...) {
    preparation_error = std::current_exception();
  }
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  if (all_reduce_max(preparation_error ? 1L : 0L, lane.communicator()) != 0) {
    if (lane.size() == 1 && preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error(
        "AMR Program restart history materialization failed on at least one MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-restart-histories"), std::string_view(contract)}},
          lane.communicator()))
    throw std::invalid_argument("AMR Program restart history contracts differ between MPI ranks");
  p_->program.hist_ = std::move(candidate);
}

template <int Dim>
std::uint64_t AmrSystem<Dim>::program_accepted_state_revision() const {
  return p_->program_accepted_revision;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_accepted_state_manifest() const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  rows.reserve(state.histories.size());
  for (const auto& history : state.histories)
    rows.push_back({history.name, "program.block." + std::to_string(history.program_owner),
                    history.state_identity, history.space_identity, history.clock_identity,
                    history.interpolation_identity, std::to_string(history.depth),
                    std::to_string(state.level_clocks.size())});
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_clock_manifest() const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  rows.reserve(state.level_clocks.size() + state.logical_clock_ticks.size());
  for (const auto& clock : state.level_clocks)
    rows.push_back({"level", std::to_string(clock.level), std::to_string(clock.macro_step),
                    std::to_string(clock.phase.numerator), std::to_string(clock.phase.denominator),
                    std::to_string(clock.physical_time)});
  for (const auto& [identity, tick] : state.logical_clock_ticks)
    rows.push_back({"logical", identity, std::to_string(tick)});
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_temporal_partition_manifest() const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  const auto& temporal = state.temporal_partition;
  rows.push_back(
      {"summary",
       temporal.kind == runtime::program::TemporalPartitionKind::Global ? "global" : "cell_local",
       temporal.provider_identity, std::to_string(temporal.topology_epoch),
       std::to_string(temporal.synchronization_tick), std::to_string(temporal.tick_denominator),
       std::to_string(temporal.cells.size())});
  std::map<int, std::size_t> rungs;
  for (const auto& cell : temporal.cells)
    ++rungs[cell.rung];
  for (const auto& [rung, count] : rungs)
    rows.push_back({"rung", std::to_string(rung), std::to_string(count)});
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_flux_ledger_manifest() const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  for (int axis = 0; axis < Dim; ++axis)
    for (const auto& fragment : state.accepted_face_flux[static_cast<std::size_t>(axis)]) {
      const auto& key = fragment.key;
      const auto& measure = fragment.measure;
      const int level = key.role == ::pops::amr::reflux::FaceLedgerRole::Coarse ? key.levels.coarse
                                                                                : key.levels.fine;
      const std::string orientation =
          std::string(1, static_cast<char>('x' + axis)) +
          (key.role == ::pops::amr::reflux::FaceLedgerRole::Coarse ? "_coarse" : "_fine");
      rows.push_back(
          {key.owner, key.state, key.stage, "numerical_flux", std::to_string(level),
           std::to_string(key.clock.macro_step), std::to_string(key.clock.phase.numerator),
           std::to_string(key.clock.phase.denominator),
           std::to_string(measure.stage_weight.numerator),
           std::to_string(measure.stage_weight.denominator), orientation,
           std::to_string(measure.face_measure), std::to_string(measure.substep_duration)});
    }
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_interface_flux_ledger_manifest()
    const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  for (const auto& fragment : state.accepted_interface_flux) {
    const auto& key = fragment.key;
    const auto& measure = fragment.measure;
    rows.push_back({key.interface_identity,
                    std::to_string(key.topology_epoch),
                    std::to_string(key.coarse_level),
                    std::to_string(key.fine_level),
                    std::to_string(key.clock.level),
                    std::to_string(key.clock.macro_step),
                    std::to_string(key.clock.phase.numerator),
                    std::to_string(key.clock.phase.denominator),
                    std::to_string(key.clock.physical_time),
                    key.stage_identity,
                    std::to_string(key.interval.begin.level),
                    std::to_string(key.interval.begin.macro_step),
                    std::to_string(key.interval.begin.phase.numerator),
                    std::to_string(key.interval.begin.phase.denominator),
                    std::to_string(key.interval.begin.physical_time),
                    std::to_string(key.interval.end.level),
                    std::to_string(key.interval.end.macro_step),
                    std::to_string(key.interval.end.phase.numerator),
                    std::to_string(key.interval.end.phase.denominator),
                    std::to_string(key.interval.end.physical_time),
                    key.orientation == ::pops::amr::InterfaceFluxOrientation::CoarseOutward
                        ? "coarse_outward"
                        : "fine_outward",
                    std::to_string(key.left_block),
                    std::to_string(key.right_block),
                    std::to_string(measure.stage_weight.numerator),
                    std::to_string(measure.stage_weight.denominator),
                    std::to_string(measure.face_measure),
                    std::to_string(measure.substep_duration),
                    measure.stage_weight_resolved ? "resolved" : "unresolved"});
  }
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_sync_manifest() const {
  std::vector<std::vector<std::string>> rows;
  const auto bytes = program_accepted_state();
  if (bytes.empty())
    return rows;
  const auto state = runtime::program::deserialize_amr_program_accepted_state<Dim>(bytes);
  rows.reserve(state.synchronization_events.size());
  for (const auto& event : state.synchronization_events)
    rows.push_back({std::to_string(event.parent_level), std::to_string(event.child_level),
                    std::to_string(event.runtime_block), event.phase,
                    std::to_string(event.clock.macro_step),
                    std::to_string(event.clock.phase.numerator),
                    std::to_string(event.clock.phase.denominator)});
  return rows;
}

template <int Dim>
std::vector<::pops::amr::ParentChildClockRelation>
AmrSystem<Dim>::prepared_program_temporal_relations() const {
  p_->ensure_engine();
  if (p_->temporal_relations.size() + 1 != p_->engine->hierarchy().num_levels())
    throw std::runtime_error(
        "AMR Program temporal hierarchy provider lacks one relation per live transition");
  for (std::size_t transition = 0; transition < p_->temporal_relations.size(); ++transition) {
    const auto& relation = p_->temporal_relations[transition];
    const auto ratio = relation.temporal_ratio();
    if (relation.parent_level() != static_cast<int>(transition) ||
        relation.child_level() != static_cast<int>(transition + 1) || ratio.numerator <= 0 ||
        ratio.denominator <= 0)
      throw std::runtime_error(
          "AMR Program temporal hierarchy provider has a non-canonical level/ratio chain");
  }
  return p_->temporal_relations;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::checkpoint_temporal_relations() const {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(p_->temporal_relations.size());
  for (const auto& relation : p_->temporal_relations) {
    const auto ratio = relation.temporal_ratio();
    rows.push_back({std::to_string(relation.parent_level()), std::to_string(relation.child_level()),
                    std::to_string(ratio.numerator), std::to_string(ratio.denominator),
                    relation.remainder_policy() == ::pops::amr::RemainderPolicy::IntegralOnly
                        ? "integral_only"
                        : "explicit_final_substep"});
  }
  return rows;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::checkpoint_transfer_routes() const {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(p_->bootstrap_subject_routes.size());
  for (const auto& [key, route_identity] : p_->bootstrap_subject_routes) {
    const auto found = p_->bootstrap_transfer_routes.find(route_identity);
    if (found == p_->bootstrap_transfer_routes.end())
      throw std::logic_error("AMR bootstrap transfer manifest lost a registered provider");
    const typename Impl::BootstrapTransferRoute& route = found->second;
    std::string ghosts;
    std::string ratio;
    for (int axis = 0; axis < Dim; ++axis) {
      if (axis != 0) {
        ghosts.push_back(',');
        ratio.push_back(',');
      }
      ghosts += std::to_string(route.ghost_depth[axis]);
      ratio += std::to_string(route.refinement_ratio[axis]);
    }
    rows.push_back({key.first, key.second, route.identity, route.provider_identity, route.kernel,
                    route.space, route.centering, route.representation, route.storage,
                    route.operation, std::to_string(route.order), std::move(ghosts),
                    std::to_string(Dim), std::move(ratio)});
  }
  return rows;
}

template AmrSystem<kNativeDimension>::AmrSystem(const AmrSystemConfig<kNativeDimension>&);
template AmrSystem<kNativeDimension>::~AmrSystem();
template AmrSystem<kNativeDimension>::AmrSystem(AmrSystem&&) noexcept;
template AmrSystem<kNativeDimension>& AmrSystem<kNativeDimension>::operator=(AmrSystem&&) noexcept;
template void AmrSystem<kNativeDimension>::add_block(
    const std::string&, const ModelSpec&, const std::string&, const std::string&,
    const std::string&, const std::string&, int, int, const std::vector<std::string>&,
    const std::vector<std::string>&, const NewtonOptions&, bool, double, double, bool);
template void AmrSystem<kNativeDimension>::set_compiled_block(
    int, double, int, AmrCompiledBlockBuilder<kNativeDimension>, const std::string&, bool,
    const std::string&, int, const std::vector<std::string>&, const std::vector<std::string>&,
    double, double, bool);
template void AmrSystem<kNativeDimension>::add_native_block(const std::string&, const std::string&,
                                                            const std::string&, const std::string&,
                                                            const std::string&, const std::string&,
                                                            double, int, const std::vector<double>&,
                                                            double, double, bool);
template void AmrSystem<kNativeDimension>::register_external_riemann_package(
    const std::string&, const std::string&, const std::string&, const std::string&, int, int,
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, double, int, int, double, double);
template void AmrSystem<kNativeDimension>::install_prepared_amr_block(
    PreparedAmrSystemBlock<kNativeDimension>);
template const MultiFab<kNativeDimension>& AmrSystem<kNativeDimension>::prepared_amr_block_state(
    int, int) const;
template MultiFab<kNativeDimension>& AmrSystem<kNativeDimension>::prepared_amr_block_state(int,
                                                                                           int);
template void AmrSystem<kNativeDimension>::install_prepared_amr_coupling_operator(
    std::string, CouplingOperatorView, AmrSystem<kNativeDimension>::PreparedCouplingOperator);
template void AmrSystem<kNativeDimension>::install_prepared_amr_interface_flux_provider(
    std::string,
    std::function<void(runtime::multiblock::InterfaceFluxScheduler<kNativeDimension>&)>);
template const AmrSystem<kNativeDimension>::ProgramBlockMap&
AmrSystem<kNativeDimension>::prepared_amr_program_block_map() const;
template void AmrSystem<kNativeDimension>::install_prepared_amr_program_flux_expression_budget(
    std::string,
    std::vector<AmrSystem<kNativeDimension>::PreparedAmrProgramFluxExpressionBlockBudget>);
template const AmrSystem<kNativeDimension>::PreparedAmrProgramFluxExpressionBudget&
AmrSystem<kNativeDimension>::prepared_amr_program_flux_expression_budget() const;
template std::size_t AmrSystem<kNativeDimension>::apply_prepared_amr_program_candidates(
    int, Real, std::span<MultiFab<kNativeDimension>* const>,
    const runtime::multiblock::BoundaryEvaluationPoint&,
    runtime::multiblock::InterfaceFluxFragmentPublication*);
template void AmrSystem<kNativeDimension>::publish_prepared_amr_program_candidates(
    int, std::span<MultiFab<kNativeDimension>* const>);
template void AmrSystem<kNativeDimension>::install_prepared_boundary_execution_context(
    std::shared_ptr<const component::PreparedExecutionContextV1>);
template void AmrSystem<kNativeDimension>::stage_prepared_ghost_boundary_component(
    const std::string&, std::shared_ptr<PreparedGhostBoundaryComponent>);
template void AmrSystem<kNativeDimension>::stage_prepared_boundary_flux_component(
    const std::string&, std::shared_ptr<PreparedBoundaryFluxComponent>);
template void AmrSystem<kNativeDimension>::stage_prepared_field_boundary_component_pair(
    const std::string&, std::shared_ptr<PreparedFieldBoundaryResidualComponent>,
    std::shared_ptr<PreparedFieldBoundaryJvpComponent>);
template void AmrSystem<kNativeDimension>::set_bootstrap_tagging(
    const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<int>&, const std::vector<double>&, const std::vector<int>&,
    const std::vector<runtime::amr::PreparedTaggingProgram<kNativeDimension>::Stencil>&,
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&,
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&, int, const std::string&,
    const std::string&, const std::string&, const std::string&);
template void AmrSystem<kNativeDimension>::install_tagger_component(
    std::shared_ptr<component::LoadedComponent>, const std::string&, const std::string&,
    std::uint32_t, const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::string&,
    std::shared_ptr<const component::PreparedExecutionContextV1>);
template void AmrSystem<kNativeDimension>::set_temporal_relations(const std::vector<std::int64_t>&,
                                                                  const std::vector<std::int64_t>&,
                                                                  const std::vector<std::string>&);
template runtime::amr::PreparedTaggerCandidates<kNativeDimension>
AmrSystem<kNativeDimension>::execute_prepared_tagging(int);
template bool AmrSystem<kNativeDimension>::regrid_from_prepared_tagging(int);
template void AmrSystem<kNativeDimension>::begin_bootstrap_plan();
template bool AmrSystem<kNativeDimension>::bootstrap_next_level();
template void AmrSystem<kNativeDimension>::commit_bootstrap_level();
template void AmrSystem<kNativeDimension>::rollback_bootstrap_level();
template void AmrSystem<kNativeDimension>::register_bootstrap_transfer_route(
    const std::string&, const std::vector<std::string>&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, int, const Extent<kNativeDimension>&, const Extent<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::register_bootstrap_oriented_face_subjects(
    const std::vector<std::string>&);
template void AmrSystem<kNativeDimension>::set_analytic_level_set(const std::vector<std::string>&,
                                                                  const std::vector<double>&,
                                                                  const std::string&, double,
                                                                  double, double);
template void AmrSystem<kNativeDimension>::set_geometry_mode(const std::string&);
template void AmrSystem<kNativeDimension>::refresh_prepared_amr_levels();
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint&);
template void AmrSystem<kNativeDimension>::prepare_generated_amr_level_state(
    const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_level_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::prepare_generated_amr_block_level_state(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_block_level_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_block_level_flux_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template bool AmrSystem<kNativeDimension>::requires_prepared_amr_block_boundary_session(int) const;
template bool AmrSystem<kNativeDimension>::has_prepared_amr_block_boundary_linearization(int) const;
template void AmrSystem<kNativeDimension>::prepared_amr_block_level_rhs_core_into_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&, bool);
template void AmrSystem<kNativeDimension>::prepared_amr_block_level_boundary_residual_into_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::prepared_amr_block_level_boundary_jvp_into_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&,
    const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::prepared_amr_block_level_source_into_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&);
template SolveOutcome AmrSystem<kNativeDimension>::solve_prepared_amr_block_level_source_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&, Real,
    const NewtonOptions&);
template void AmrSystem<kNativeDimension>::prepare_generated_amr_block_level_state(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&, int,
    const MultiFab<kNativeDimension>*);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_block_level_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&, int,
    const MultiFab<kNativeDimension>*);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_block_level_flux_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&, int,
    const MultiFab<kNativeDimension>*);
template void AmrSystem<kNativeDimension>::prepared_amr_block_level_source_into_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&,
    MultiFab<kNativeDimension>&, int, const MultiFab<kNativeDimension>*);
template SolveOutcome AmrSystem<kNativeDimension>::solve_prepared_amr_block_level_source_at(
    int, const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&, Real,
    const NewtonOptions&, int, const MultiFab<kNativeDimension>*);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_level_evaluation(int) const;
template const PreparedAmrLevelEvaluation<kNativeDimension>*
AmrSystem<kNativeDimension>::prepared_amr_level_evaluation_if_present(int) const noexcept;
template void AmrSystem<kNativeDimension>::clear_prepared_amr_level_evaluations() const noexcept;
template void AmrSystem<kNativeDimension>::bind_program_hierarchy_candidates(
    const std::vector<MultiFab<kNativeDimension>>*) const;
template void AmrSystem<kNativeDimension>::unbind_program_hierarchy_candidates(
    const std::vector<MultiFab<kNativeDimension>>*) const noexcept;
template void AmrSystem<kNativeDimension>::bind_program_block_hierarchy_candidates(
    int, const std::vector<MultiFab<kNativeDimension>>*) const;
template void AmrSystem<kNativeDimension>::unbind_program_block_hierarchy_candidates(
    int, const std::vector<MultiFab<kNativeDimension>>*) const noexcept;
template Geometry<kNativeDimension> AmrSystem<kNativeDimension>::prepared_amr_level_geometry(
    int) const;
template AmrSystem<kNativeDimension>::PreparedMultiBlockHierarchy&
AmrSystem<kNativeDimension>::prepared_amr_multiblock_hierarchy_();
template const AmrSystem<kNativeDimension>::PreparedMultiBlockHierarchy&
AmrSystem<kNativeDimension>::prepared_amr_multiblock_hierarchy_() const;
template BoundaryTopology<kNativeDimension>
AmrSystem<kNativeDimension>::prepared_amr_boundary_topology() const;
template Real AmrSystem<kNativeDimension>::prepared_amr_level_maximum_speed(
    int, const MultiFab<kNativeDimension>&) const;
template Real AmrSystem<kNativeDimension>::prepared_amr_block_level_maximum_speed(
    int, int, const MultiFab<kNativeDimension>&) const;
template void AmrSystem<kNativeDimension>::validate_prepared_amr_state_publication_candidate(
    int, int, const MultiFab<kNativeDimension>&) const;
template void AmrSystem<kNativeDimension>::add_prepared_amr_block_poisson_rhs(
    int, int, const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::install_prepared_auxiliary_provider(
    runtime::system::PreparedAuxiliaryProvider<kNativeDimension>);
template void AmrSystem<kNativeDimension>::install_auxiliary_consumer_plan(
    runtime::system::AuxiliaryConsumerProviderPlan<kNativeDimension>);
template void AmrSystem<kNativeDimension>::seal_auxiliary_providers();
template void AmrSystem<kNativeDimension>::stage_auxiliary_input(
    const runtime::system::AuxiliaryComponentKey&, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::refresh_auxiliary(
    const runtime::system::AuxiliaryEvaluationPoint&);
template runtime::system::AuxiliaryStorageAddress<kNativeDimension>
AmrSystem<kNativeDimension>::auxiliary_address(const runtime::system::AuxiliaryComponentKey&) const;
template std::vector<double> AmrSystem<kNativeDimension>::auxiliary_component(
    const runtime::system::AuxiliaryComponentKey&, int) const;
template std::string AmrSystem<kNativeDimension>::auxiliary_registry_contract() const;
template const runtime::system::ResolvedAuxiliaryConsumerPlan<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_auxiliary_consumer_plan(const std::string&) const;
template const runtime::system::AuxiliaryStorageGroups<kNativeDimension>*
AmrSystem<kNativeDimension>::prepared_amr_provider_storage_groups(int) const;
template const runtime::system::ResolvedAuxiliaryConsumerPlan<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_auxiliary_consumer_plan(const std::string&, int) const;
template std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<kNativeDimension>>
AmrSystem<kNativeDimension>::capture_auxiliary_checkpoint_accepted_state() const;
template void AmrSystem<kNativeDimension>::restore_auxiliary_checkpoint_accepted_state(
    const std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<kNativeDimension>>&);
template void AmrSystem<kNativeDimension>::add_prepared_amr_poisson_rhs(
    int, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::install_block_state_route(const std::string&,
                                                                     const std::string&);
template void AmrSystem<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                       const std::string&);
template void AmrSystem<kNativeDimension>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<kNativeDimension>>);
template void AmrSystem<kNativeDimension>::register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<kNativeDimension>>);
template void AmrSystem<kNativeDimension>::register_program_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<kNativeDimension>>);
template std::shared_ptr<
    const runtime::program::HierarchyTensorSolverProviderRegistry<kNativeDimension>>
AmrSystem<kNativeDimension>::hierarchy_tensor_solver_provider_registry() const;
template void AmrSystem<kNativeDimension>::register_field_solver_provider(
    std::shared_ptr<const runtime::amr::ExactAmrFieldSolverProvider<kNativeDimension>>);
template std::string AmrSystem<kNativeDimension>::register_field_solver_provider(
    const std::string&, runtime::field::PreparedFieldSolverSpec,
    std::shared_ptr<component::LoadedComponent>, std::shared_ptr<component::LoadedComponent>);
template void AmrSystem<kNativeDimension>::set_default_field_nullspace(
    const std::string&, const PreparedProviderOptions&);
template void AmrSystem<kNativeDimension>::set_poisson(const std::string&, const std::string&,
                                                       const std::string&,
                                                       const AmrFieldSolverOptions&);
template void AmrSystem<kNativeDimension>::set_field_solver_plan(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<double>&,
    const std::string&, const AmrFieldHierarchyPolicyAuthority&, const AmrFieldSolverOptions&);
template AmrFieldSolverConfiguration AmrSystem<kNativeDimension>::field_solver_configuration(
    const std::string&) const;
template void AmrSystem<kNativeDimension>::set_field_reaction(const std::string&, double);
template void AmrSystem<kNativeDimension>::set_field_topology_authority(const std::string&,
                                                                        const std::string&,
                                                                        const std::string&,
                                                                        const std::string&);
template std::vector<runtime::field::FieldTopologyReportRow>
AmrSystem<kNativeDimension>::field_topology_report(const std::string&) const;
template void AmrSystem<kNativeDimension>::set_field_boundary_plan(const std::string&,
                                                                   const std::vector<std::string>&,
                                                                   const std::vector<double>&,
                                                                   const std::vector<double>&,
                                                                   const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_boundary_dependencies(
    const std::string&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&);
template void AmrSystem<kNativeDimension>::set_field_boundary_kernel(
    const std::string&, const CompiledFieldBoundaryKernel<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::set_field_logical_timepoint(
    const std::string&, const FieldLogicalTimePoint&);
template void AmrSystem<kNativeDimension>::set_field_boundary_parameters(
    const std::string&, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_newton_plan(const std::string&, double, int,
                                                                 double, int, int, double, double);
template void AmrSystem<kNativeDimension>::set_field_nullspace(const std::string&,
                                                               const std::string&,
                                                               const PreparedProviderOptions&);
template void AmrSystem<kNativeDimension>::register_elliptic_field(
    const std::string&, const std::string&,
    const std::vector<runtime::system::AuxiliaryComponentKey>&, int);
template void AmrSystem<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_field_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const MultiFab<kNativeDimension>*);
template void AmrSystem<kNativeDimension>::with_program_field_candidate_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const MultiFab<kNativeDimension>&, const std::function<void()>&);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_field_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_default_field(int);
template std::vector<double> AmrSystem<kNativeDimension>::named_field_values(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::field_provider_slots() const;
template int AmrSystem<kNativeDimension>::field_provider_levels(const std::string&);
template void AmrSystem<kNativeDimension>::set_field_potential(const std::string&,
                                                               const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_potential_level(const std::string&, int,
                                                                     const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::field_potential_global(
    const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::field_potential_level_global(
    const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_field_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_field_root_pieces(const ObserverMpiLane&, const std::string&,
                                                      int);
template void AmrSystem<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::string&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&,
    const std::vector<std::string>&);
template void AmrSystem<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const HyperbolicBoundary>);
template void AmrSystem<kNativeDimension>::discard_hyperbolic_boundaries();
template void AmrSystem<kNativeDimension>::set_density(const std::string&,
                                                       const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_conservative_state(const std::string&,
                                                                  const std::vector<double>&);
template void AmrSystem<kNativeDimension>::bind_bootstrap_subject(const std::string&,
                                                                  const std::string&,
                                                                  const std::string&);
template void AmrSystem<kNativeDimension>::stage_bootstrap_analytic_state(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const analytic::AnalyticOpcodeRows&, const analytic::AnalyticLiteralRows&);
template void AmrSystem<kNativeDimension>::stage_bootstrap_array(
    const std::string&, const std::string&, const std::string&, const std::string&, int,
    const Extent<kNativeDimension>&, const std::vector<double>&);
template std::size_t AmrSystem<kNativeDimension>::materialize_bootstrap_action(const std::string&,
                                                                               const std::string&,
                                                                               const std::string&,
                                                                               int);
template void AmrSystem<kNativeDimension>::add_dt_bound(const std::string&,
                                                        std::function<double()>);
template std::string AmrSystem<kNativeDimension>::last_dt_bound() const;
template void AmrSystem<kNativeDimension>::step(double);
template void AmrSystem<kNativeDimension>::advance(double, int);
template double AmrSystem<kNativeDimension>::step_cfl(double, double, double, double);
template void AmrSystem<kNativeDimension>::begin_step_transaction();
template void AmrSystem<kNativeDimension>::commit_step_transaction();
template void AmrSystem<kNativeDimension>::finalize_step_transaction();
template void AmrSystem<kNativeDimension>::rollback_step_transaction();
template bool AmrSystem<kNativeDimension>::has_active_step_transaction() const noexcept;
template void AmrSystem<kNativeDimension>::restore_active_step_transaction_for_program();
template std::map<std::string, double> AmrSystem<kNativeDimension>::step_change_l2() const;
template void AmrSystem<kNativeDimension>::begin_restart_transaction();
template void AmrSystem<kNativeDimension>::commit_restart_transaction();
template void AmrSystem<kNativeDimension>::rollback_restart_transaction();
template void AmrSystem<kNativeDimension>::preflight_regrid_on_restart();
template void AmrSystem<kNativeDimension>::regrid_on_restart();
template int AmrSystem<kNativeDimension>::checkpoint_regrid_count() const;
template std::uint64_t AmrSystem<kNativeDimension>::checkpoint_topology_epoch() const;
template void AmrSystem<kNativeDimension>::restore_checkpoint_counters(int, std::uint64_t);
template void AmrSystem<kNativeDimension>::install_program(const std::string&);
template void AmrSystem<kNativeDimension>::install_program_step(std::function<void(double)>);
template void AmrSystem<kNativeDimension>::install_program_hierarchy_refresh(std::function<void()>);
template void AmrSystem<kNativeDimension>::install_program_restart_hooks(std::function<void()>,
                                                                         std::function<void()>,
                                                                         std::function<void()>);
template void AmrSystem<kNativeDimension>::set_program_cadence(int, int);
template int AmrSystem<kNativeDimension>::program_substeps() const;
template int AmrSystem<kNativeDimension>::program_stride() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_dt() const;
template int AmrSystem<kNativeDimension>::program_cadence_window_steps() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_start_time() const;
template double AmrSystem<kNativeDimension>::program_last_dt() const;
template void AmrSystem<kNativeDimension>::restore_program_cadence_window(double, int, double,
                                                                          double, double, int);
template void AmrSystem<kNativeDimension>::set_program_block_map(const std::vector<int>&);
template const std::vector<int>& AmrSystem<kNativeDimension>::program_block_map() const;
template std::string AmrSystem<kNativeDimension>::installed_program_hash() const;
template void AmrSystem<kNativeDimension>::seed_program_params(int, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_program_params(int, const std::vector<double>&);
template RuntimeParams AmrSystem<kNativeDimension>::program_params(int) const;
template runtime::amr::AmrRuntime<kNativeDimension>* AmrSystem<kNativeDimension>::engine() const;
template bool AmrSystem<kNativeDimension>::uses_runtime_engine() const;
template runtime::program::Profiler& AmrSystem<kNativeDimension>::profiler_handle();
template runtime::program::ProgramRuntimeState<kNativeDimension>&
AmrSystem<kNativeDimension>::program_runtime_state_();
template void AmrSystem<kNativeDimension>::record_program_diagnostic(const std::string&, double);
template void AmrSystem<kNativeDimension>::record_program_balance_term(const std::string&,
                                                                       const std::string&, double);
template bool AmrSystem<kNativeDimension>::program_balance_consumer_is_due(const std::string&,
                                                                           const std::string&,
                                                                           int) const;
template double AmrSystem<kNativeDimension>::program_diagnostic(const std::string&) const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::program_diagnostics() const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::accepted_balance_terms(
    const std::string&) const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::selected_accepted_balance_terms(
    const std::string&, const std::string&, int, const std::vector<int>&,
    const std::vector<std::string>&) const;
template void AmrSystem<kNativeDimension>::begin_step_projection_report();
template void AmrSystem<kNativeDimension>::note_step_projection(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::consume_step_projections();
template void AmrSystem<kNativeDimension>::mark_bound();
template std::string AmrSystem<kNativeDimension>::lifecycle_state() const;
template Extent<kNativeDimension> AmrSystem<kNativeDimension>::spatial_shape() const;
template double AmrSystem<kNativeDimension>::time() const;
template int AmrSystem<kNativeDimension>::macro_step() const;
template void AmrSystem<kNativeDimension>::set_clock(double, int);
template void AmrSystem<kNativeDimension>::enable_profiling();
template void AmrSystem<kNativeDimension>::disable_profiling();
template bool AmrSystem<kNativeDimension>::is_profiling() const;
template void AmrSystem<kNativeDimension>::reset_profiling();
template std::string AmrSystem<kNativeDimension>::profile_report() const;
template int AmrSystem<kNativeDimension>::n_blocks() const;
template std::vector<std::string> AmrSystem<kNativeDimension>::block_names() const;
template EffectiveOptionsReport AmrSystem<kNativeDimension>::effective_options_report() const;
template int AmrSystem<kNativeDimension>::n_levels();
template int AmrSystem<kNativeDimension>::max_levels();
template int AmrSystem<kNativeDimension>::configured_n_levels();
template int AmrSystem<kNativeDimension>::n_vars();
template int AmrSystem<kNativeDimension>::block_n_vars(const std::string&);
template int AmrSystem<kNativeDimension>::n_patches();
template std::vector<AmrPatch<kNativeDimension>> AmrSystem<kNativeDimension>::patch_boxes();
template std::vector<AmrPatch<kNativeDimension>>
AmrSystem<kNativeDimension>::output_geometry_boxes();
template int AmrSystem<kNativeDimension>::coarse_local_boxes();
template int AmrSystem<kNativeDimension>::coarse_total_boxes();
template std::vector<double> AmrSystem<kNativeDimension>::level_state(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_state_global(int);
template void AmrSystem<kNativeDimension>::set_level_state(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::level_potential(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_potential_global(int);
template void AmrSystem<kNativeDimension>::set_level_potential(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state(const std::string&,
                                                                            int);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state_global(
    const std::string&, int);
template void AmrSystem<kNativeDimension>::set_block_level_state(const std::string&, int,
                                                                 const std::vector<double>&);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane&, const std::string&,
                                                      int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_embedded_boundary_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_embedded_boundary_root_pieces(const ObserverMpiLane&,
                                                                  const std::string&, int);
template double AmrSystem<kNativeDimension>::composite_reduce(const std::string&,
                                                              const std::string&, int,
                                                              const std::vector<int>&) const;
template double AmrSystem<kNativeDimension>::composite_reduce_field(const std::string&,
                                                                    const std::string&, int,
                                                                    const std::vector<int>&);
template void AmrSystem<kNativeDimension>::set_hierarchy(
    const std::vector<AmrPatch<kNativeDimension>>&);
template void AmrSystem<kNativeDimension>::rebuild_hierarchy(
    const std::vector<AmrPatch<kNativeDimension>>&, const std::vector<int>&);
template std::vector<int> AmrSystem<kNativeDimension>::rematerialize_hierarchy_ownership(
    const std::vector<AmrPatch<kNativeDimension>>&);
template std::vector<std::uint8_t>
AmrSystem<kNativeDimension>::rematerialize_program_accepted_state(
    const std::vector<std::vector<std::uint8_t>>&, const std::vector<std::vector<int>>&,
    const std::vector<std::vector<int>>&);
template std::vector<int> AmrSystem<kNativeDimension>::level_owner_ranks(int);
template std::vector<std::string> AmrSystem<kNativeDimension>::history_names() const;
template int AmrSystem<kNativeDimension>::history_depth(const std::string&) const;
template int AmrSystem<kNativeDimension>::history_ncomp(const std::string&) const;
template bool AmrSystem<kNativeDimension>::history_initialized(const std::string&) const;
template int AmrSystem<kNativeDimension>::history_fill_count(const std::string&) const;
template void AmrSystem<kNativeDimension>::set_history_initialized(const std::string&, bool);
template void AmrSystem<kNativeDimension>::restore_history_fill_count(const std::string&, int);
template void AmrSystem<kNativeDimension>::restore_history_metadata(const std::string&, bool, int);
template std::vector<double> AmrSystem<kNativeDimension>::history_global(const std::string&,
                                                                         int) const;
template void AmrSystem<kNativeDimension>::restore_history(const std::string&, int,
                                                           const std::vector<double>&);
template double AmrSystem<kNativeDimension>::history_slot_dt(const std::string&, int) const;
template void AmrSystem<kNativeDimension>::restore_history_slot_dt(const std::string&, int, double);
template int AmrSystem<kNativeDimension>::rebuild_history_slots(const std::string&,
                                                                const std::vector<int>&);
template std::vector<int> AmrSystem<kNativeDimension>::last_replay_regrid_steps() const;
template double AmrSystem<kNativeDimension>::mass();
template double AmrSystem<kNativeDimension>::mass(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::density();
template std::vector<double> AmrSystem<kNativeDimension>::density(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::potential();
template std::vector<std::uint8_t> AmrSystem<kNativeDimension>::program_accepted_state() const;
template void AmrSystem<kNativeDimension>::copy_program_accepted_state_into(
    std::vector<std::uint8_t>&) const;
template void AmrSystem<kNativeDimension>::restore_program_accepted_state(
    const std::vector<std::uint8_t>&);
template void AmrSystem<kNativeDimension>::restore_checkpoint_accepted_state(
    const std::vector<std::uint8_t>&);
template void AmrSystem<kNativeDimension>::materialize_program_restart_histories(
    const std::vector<std::uint8_t>&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<int>&);
template std::uint64_t AmrSystem<kNativeDimension>::program_accepted_state_revision() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_accepted_state_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_clock_manifest()
    const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_temporal_partition_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_interface_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_sync_manifest()
    const;
template std::vector<::pops::amr::ParentChildClockRelation>
AmrSystem<kNativeDimension>::prepared_program_temporal_relations() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::checkpoint_temporal_relations() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::checkpoint_transfer_routes() const;
}  // namespace pops
