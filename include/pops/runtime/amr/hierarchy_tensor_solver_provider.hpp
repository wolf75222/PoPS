/// @file
/// @brief Exact-ranked prepared provider protocol for hierarchy tensor elliptic solves.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/solve_report_consensus.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program {

struct HierarchyTensorSolveControls {
  Real relative_tolerance = Real(0);
  Real absolute_tolerance = Real(0);
  int maximum_iterations = 0;
};

/// Exact immutable spatial request for one materialized hierarchy level.
template <int Dim>
struct HierarchyTensorLevelBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3);

  Geometry<Dim> geometry;
  PhysicalBoundaryConditions<Dim> boundary;
  mesh::BoxArray<Dim> layout;
  mesh::Distribution<Dim> distribution;
  Index<Dim> local_rank{};
};

/// Provider-neutral build request. Spatial rank is a template argument, never a payload tag.
template <int Dim>
struct HierarchyTensorSolverBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3);

  static constexpr int dimension = Dim;
  using level_type = HierarchyTensorLevelBuildRequest<Dim>;

  std::size_t block = 0;
  int components = 0;
  std::vector<level_type> levels;
  std::vector<::pops::amr::RefinementRatio<Dim>> ratios;
  std::string plan_identity;
  std::string operator_contract_identity;
  std::vector<std::string> assembly_field_slots;
  std::string solution_field_slot;
  PreparedProviderOptions options;
};

enum class HierarchyTensorSolverExecutionPath : std::uint8_t {
  PreparedKrylovFallback,
  DirectProvider,
};

namespace hierarchy_tensor_detail {

template <int Dim>
Extent<Dim> ratio_extent(const ::pops::amr::RefinementRatio<Dim>& ratio) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = ratio[axis];
  return result;
}

template <int Dim>
void validate_request(const HierarchyTensorSolverBuildRequest<Dim>& request) {
  if (request.components < 1 || request.levels.empty() || request.plan_identity.empty() ||
      request.operator_contract_identity.empty() || request.solution_field_slot.empty())
    throw std::invalid_argument("hierarchy tensor request has an incomplete operator envelope");
  if (request.ratios.size() + 1 != request.levels.size())
    throw std::invalid_argument("hierarchy tensor request ratios do not cover every level edge");
  if (request.assembly_field_slots.empty() ||
      std::any_of(request.assembly_field_slots.begin(), request.assembly_field_slots.end(),
                  [](const std::string& slot) { return slot.empty(); }))
    throw std::invalid_argument("hierarchy tensor request has invalid assembly field slots");
  std::vector<std::string> ordered_slots = request.assembly_field_slots;
  std::sort(ordered_slots.begin(), ordered_slots.end());
  if (std::adjacent_find(ordered_slots.begin(), ordered_slots.end()) != ordered_slots.end())
    throw std::invalid_argument("hierarchy tensor request field slots must be unique");

  const auto& rank_space = request.levels.front().distribution.rank_space();
  const Index<Dim> local_rank = request.levels.front().local_rank;
  for (std::size_t level = 0; level < request.levels.size(); ++level) {
    const auto& current = request.levels[level];
    for (int axis = 0; axis < Dim; ++axis)
      if (current.boundary.spacing()[axis] != current.geometry.spacing(axis))
        throw std::invalid_argument(
            "hierarchy tensor boundary spacing differs from its exact geometry");
    if (!current.distribution.matches_layout(current.layout) ||
        current.distribution.rank_space() != rank_space || current.local_rank != local_rank ||
        !rank_space.contains(current.local_rank))
      throw std::invalid_argument(
          "hierarchy tensor level layout, distribution, and process coordinate disagree");
    for (const Box<Dim>& patch : current.layout.boxes())
      if (patch.intersect(current.geometry.domain()) != patch)
        throw std::invalid_argument("hierarchy tensor patch lies outside its exact geometry");
    if (level != 0) {
      const Geometry<Dim> expected =
          request.levels[level - 1].geometry.refine(ratio_extent(request.ratios[level - 1]));
      if (current.geometry != expected)
        throw std::invalid_argument("hierarchy tensor geometry is not the exact parent refinement");
    }
  }
}

template <int Dim>
std::string request_contract(const HierarchyTensorSolverBuildRequest<Dim>& request) {
  validate_request(request);
  ExactContractBuilder contract;
  contract.text("pops.hierarchy.tensor-solver-request")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(request.block))
      .scalar(request.components)
      .text(request.plan_identity)
      .text(request.operator_contract_identity)
      .sequence(request.assembly_field_slots,
                [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); })
      .text(request.solution_field_slot)
      .bytes(request.options.exact_contract())
      .scalar(static_cast<std::uint64_t>(request.levels.size()));
  for (const auto& level : request.levels) {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(level.geometry.domain().lo[axis])
          .scalar(level.geometry.domain().hi[axis])
          .scalar(level.geometry.lower()[axis])
          .scalar(level.geometry.upper()[axis])
          .scalar(level.distribution.rank_space().origin()[axis])
          .scalar(level.distribution.rank_space().extent()[axis]);
    for (int axis = 0; axis < Dim; ++axis)
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const PhysicalBoundaryFace& law = level.boundary.at(face);
        contract.scalar(level.boundary.topology().is_periodic(face))
            .scalar(law.kind)
            .scalar(law.value)
            .scalar(law.alpha)
            .scalar(law.beta);
      }
    contract.scalar(level.distribution.mode())
        .sequence(level.layout.boxes(),
                  [](ExactContractBuilder& item, const Box<Dim>& patch) {
                    for (int axis = 0; axis < Dim; ++axis)
                      item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                  })
        .sequence(level.distribution.owners(),
                  [](ExactContractBuilder& item, const Index<Dim>& owner) {
                    for (int axis = 0; axis < Dim; ++axis)
                      item.scalar(owner[axis]);
                  });
  }
  contract.scalar(static_cast<std::uint64_t>(request.ratios.size()));
  for (const auto& ratio : request.ratios)
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(ratio[axis]);
  return std::move(contract).release();
}

template <int Dim>
struct CopyAllocatedKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> source;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, component) = source(index, component);
  }
};

template <int Dim, class MemorySpace>
bool same_field_shape(const MultiFab<Dim, MemorySpace>& left,
                      const MultiFab<Dim, MemorySpace>& right) noexcept {
  return left.layout() == right.layout() && left.distribution() == right.distribution() &&
         left.local_rank() == right.local_rank() && left.ncomp() == right.ncomp() &&
         left.ghosts() == right.ghosts() && left.local_size() == right.local_size();
}

template <int Dim, class MemorySpace>
void copy_allocated(MultiFab<Dim, MemorySpace>& destination,
                    const MultiFab<Dim, MemorySpace>& source) {
  if (!same_field_shape(destination, source))
    throw std::invalid_argument("hierarchy tensor publication fields have different shapes");
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const FieldView<Real, Dim> output = destination.fab(local).view();
    const FieldView<const Real, Dim> input = std::as_const(source.fab(local)).view();
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(destination.fab(local).grown_box(),
                    CopyAllocatedKernel<Dim>{output, input, component});
  }
  Kokkos::fence();
}

}  // namespace hierarchy_tensor_detail

/// A prepared hierarchy solver owns one immutable native spatial specialization.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedHierarchyTensorSolver {
 public:
  static_assert(Dim >= 1 && Dim <= 3);
  using field_type = MultiFab<Dim, MemorySpace>;

  virtual ~PreparedHierarchyTensorSolver() = default;
  virtual std::string_view provider_identity() const noexcept = 0;
  virtual std::uint64_t provider_version() const noexcept = 0;
  virtual std::string_view exact_prepared_contract() const noexcept = 0;
  virtual HierarchyTensorSolverExecutionPath execution_path() const noexcept = 0;
  virtual int level_count() const noexcept = 0;
  virtual field_type& assembly_target(std::string_view field_slot_identity, int level) = 0;
  virtual field_type& solution(int level) = 0;
  virtual void stage_initial_guess(int level, const field_type* guess) = 0;

  FieldView<Real, Dim> assembly_target_view(std::string_view field_slot_identity, int level,
                                            std::size_t local_patch) {
    return assembly_target(field_slot_identity, level).fab(local_patch).view();
  }
  FieldView<Real, Dim> solution_view(int level, std::size_t local_patch) {
    return solution(level).fab(local_patch).view();
  }

  /// Materialize both rollback images during preparation, never on the solve hot path.
  void seal_preparation(const ExecutionLane& lane) {
    if (preparation_sealed_)
      throw std::logic_error("hierarchy tensor solver preparation is already sealed");
    const int levels = level_count();
    if (levels < 0)
      throw std::logic_error("hierarchy tensor provider has a negative level count");
    accepted_publication_.reserve(static_cast<std::size_t>(levels));
    candidate_publication_.reserve(static_cast<std::size_t>(levels));
    for (int level = 0; level < levels; ++level) {
      field_type& live = solution(level);
      accepted_publication_.emplace_back(live.layout(), live.distribution(), live.local_rank(),
                                         live.ncomp(), live.ghosts());
      candidate_publication_.emplace_back(live.layout(), live.distribution(), live.local_rank(),
                                          live.ncomp(), live.ghosts());
    }
    prepared_lane_ = &lane;
    prepared_lane_borrow_.emplace(lane.borrow_immutably());
    preparation_sealed_ = true;
  }

  SolveOutcome execute_collectively(const HierarchyTensorSolveControls& controls,
                                    const ExecutionLane& lane) {
    if (prepared_lane_ == nullptr)
      throw std::logic_error("hierarchy tensor solver has no prepared execution lane");
    if (all_reduce_max(&lane == prepared_lane_ ? 0L : 1L, *prepared_lane_) != 0)
      throw std::invalid_argument("hierarchy tensor solve requires its prepared execution lane");
    const ExecutionLane& execution_lane = *prepared_lane_;
    const bool invalid_controls =
        !std::isfinite(controls.relative_tolerance) || controls.relative_tolerance < Real(0) ||
        !std::isfinite(controls.absolute_tolerance) || controls.absolute_tolerance < Real(0) ||
        controls.maximum_iterations < 0;
    if (all_reduce_max(invalid_controls || !preparation_sealed_ ? 1L : 0L, execution_lane) != 0)
      throw std::invalid_argument(
          "hierarchy tensor solve requires valid controls and a sealed preparation");
    if (all_reduce_max(publication_active_ ? 1L : 0L, execution_lane) != 0)
      throw std::logic_error(
          "hierarchy tensor solve is reserved until its prior outcome is consumed");

    if (!collective_capture_(accepted_publication_, execution_lane))
      throw std::runtime_error(
          "hierarchy tensor accepted-state snapshot failed on at least one MPI rank");
    publication_active_ = true;

    SolveReport report;
    long solve_failed = 0;
    try {
      report = solve(controls, execution_lane);
    } catch (...) {
      solve_failed = 1;
    }
    if (all_reduce_max(solve_failed, execution_lane) != 0) {
      restore_or_terminate_(accepted_publication_);
      release_publication_();
      throw std::runtime_error("hierarchy tensor provider failed on at least one MPI rank");
    }
    if (all_reduce_max(!solve_report_is_publishable(report, controls.maximum_iterations) ? 1L : 0L,
                       execution_lane) != 0) {
      restore_or_terminate_(accepted_publication_);
      release_publication_();
      throw std::runtime_error("hierarchy tensor provider published a malformed SolveReport");
    }
    ExactSolveReportConsensusScratch report_consensus;
    if (!report_consensus.agrees(report, execution_lane)) {
      restore_or_terminate_(accepted_publication_);
      release_publication_();
      throw std::runtime_error("hierarchy tensor provider report differs between MPI ranks");
    }
    if (!report.solved_value_available()) {
      restore_or_terminate_(accepted_publication_);
      release_publication_();
      return SolveOutcome::collective_lane(std::move(report), execution_lane);
    }
    if (!collective_capture_(candidate_publication_, execution_lane)) {
      restore_or_terminate_(accepted_publication_);
      release_publication_();
      throw std::runtime_error("hierarchy tensor candidate staging failed collectively");
    }
    restore_or_terminate_(accepted_publication_);
    return SolveOutcome::collective_lane(
        std::move(report), execution_lane,
        SolveOutcome::PublicationHooks{
            this,
            [](void* context) noexcept {
              auto* prepared = static_cast<PreparedHierarchyTensorSolver*>(context);
              prepared->restore_or_terminate_(prepared->candidate_publication_);
            },
            nullptr,
            [](void* context) noexcept {
              static_cast<PreparedHierarchyTensorSolver*>(context)->release_publication_();
            },
            {},
            [](void* context) {
              static_cast<PreparedHierarchyTensorSolver*>(context)
                  ->validate_candidate_publication_();
            }});
  }

 protected:
  virtual SolveReport solve(const HierarchyTensorSolveControls& controls,
                            const ExecutionLane& lane) = 0;

 private:
  bool collective_capture_(std::vector<field_type>& storage, const ExecutionLane& lane) {
    long failure = 0;
    try {
      if (storage.size() != static_cast<std::size_t>(level_count()))
        throw std::logic_error("hierarchy tensor publication depth changed after preparation");
      for (int level = 0; level < level_count(); ++level)
        hierarchy_tensor_detail::copy_allocated(storage[static_cast<std::size_t>(level)],
                                                solution(level));
    } catch (...) {
      failure = 1;
    }
    return all_reduce_max(failure, lane) == 0;
  }

  void restore_or_terminate_(const std::vector<field_type>& storage) noexcept {
    try {
      if (storage.size() != static_cast<std::size_t>(level_count()))
        std::terminate();
      for (int level = 0; level < level_count(); ++level)
        hierarchy_tensor_detail::copy_allocated(solution(level),
                                                storage[static_cast<std::size_t>(level)]);
    } catch (...) {
      std::terminate();
    }
  }

  void validate_candidate_publication_() {
    if (candidate_publication_.size() != static_cast<std::size_t>(level_count()))
      throw std::logic_error("hierarchy tensor publication depth changed before Accept");
    for (int level = 0; level < level_count(); ++level)
      if (!hierarchy_tensor_detail::same_field_shape(
              solution(level), candidate_publication_[static_cast<std::size_t>(level)]))
        throw std::logic_error("hierarchy tensor publication shape changed before Accept");
  }

  void release_publication_() noexcept { publication_active_ = false; }

  std::vector<field_type> accepted_publication_;
  std::vector<field_type> candidate_publication_;
  const ExecutionLane* prepared_lane_ = nullptr;
  std::optional<ExecutionLane::ImmutableBorrow> prepared_lane_borrow_;
  bool preparation_sealed_ = false;
  bool publication_active_ = false;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class HierarchyTensorSolverProvider {
 public:
  using request_type = HierarchyTensorSolverBuildRequest<Dim>;
  using solver_type = PreparedHierarchyTensorSolver<Dim, MemorySpace>;

  virtual ~HierarchyTensorSolverProvider() = default;
  virtual std::string_view identity() const noexcept = 0;
  virtual std::uint64_t interface_version() const noexcept = 0;
  virtual std::string_view collective_contract() const noexcept = 0;
  virtual std::vector<std::string> capability_contracts() const = 0;
  virtual PreparedProviderOptions default_options() const = 0;
  virtual PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept = 0;
  virtual PreparedProviderSupport supports(const request_type& request) const noexcept = 0;
  virtual PreparedProviderSupport accepts_execution(
      const request_type& request, HierarchyTensorSolverExecutionPath execution) const noexcept = 0;
  virtual std::string expected_prepared_contract(const request_type& request) const = 0;
  virtual std::unique_ptr<solver_type> prepare(const request_type& request,
                                               const ExecutionLane& lane) const = 0;
};

template <int Dim, class MemorySpace>
SolveOutcome solve_prepared_hierarchy_tensor_collectively(
    PreparedHierarchyTensorSolver<Dim, MemorySpace>& solver,
    const HierarchyTensorSolveControls& controls, const ExecutionLane& lane) {
  return solver.execute_collectively(controls, lane);
}

template <int Dim, class MemorySpace>
std::string exact_hierarchy_tensor_solver_provider_declaration(
    const HierarchyTensorSolverProvider<Dim, MemorySpace>& provider) {
  std::vector<std::string> capabilities = provider.capability_contracts();
  std::sort(capabilities.begin(), capabilities.end());
  if (std::any_of(capabilities.begin(), capabilities.end(),
                  [](const std::string& value) { return value.empty(); }) ||
      std::adjacent_find(capabilities.begin(), capabilities.end()) != capabilities.end())
    throw std::invalid_argument(
        "hierarchy tensor provider capabilities require unique exact identities");
  ExactContractBuilder contract;
  contract.text("pops.hierarchy.tensor-solver-provider-declaration")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .text(provider.identity())
      .scalar(provider.interface_version())
      .text(provider.collective_contract())
      .sequence(capabilities, [](ExactContractBuilder& item,
                                 const std::string& capability) { item.text(capability); })
      .bytes(provider.default_options().exact_contract());
  return std::move(contract).release();
}

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class HierarchyTensorSolverProviderRegistry {
 public:
  using provider_type = HierarchyTensorSolverProvider<Dim, MemorySpace>;

  void add(std::shared_ptr<const provider_type> provider, const ExecutionLane& lane) {
    std::string identity;
    std::string declaration;
    long invalid = 0;
    try {
      if (!provider)
        throw std::invalid_argument("null hierarchy tensor provider");
      identity = std::string(provider->identity());
      declaration = exact_hierarchy_tensor_solver_provider_declaration(*provider);
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, lane) != 0)
      throw std::runtime_error("hierarchy tensor provider declaration failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"hierarchy-tensor-provider-declaration", declaration}}, lane))
      throw std::runtime_error("hierarchy tensor provider declaration differs across MPI ranks");
    const auto existing = providers_.find(identity);
    const long present = existing == providers_.end() ? 0L : 1L;
    if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
      throw std::runtime_error("hierarchy tensor provider registry differs across MPI ranks");
    if (present != 0) {
      if (exact_hierarchy_tensor_solver_provider_declaration(*existing->second) != declaration)
        throw std::invalid_argument("conflicting hierarchy tensor provider identity '" + identity +
                                    "'");
      return;
    }
    providers_.emplace(std::move(identity), std::move(provider));
  }

  std::shared_ptr<const provider_type> resolve(std::string_view identity) const {
    const auto found = providers_.find(std::string(identity));
    if (found == providers_.end())
      throw std::invalid_argument("unknown hierarchy tensor provider '" + std::string(identity) +
                                  "'");
    return found->second;
  }

 private:
  std::map<std::string, std::shared_ptr<const provider_type>> providers_;
};

template <int Dim, class MemorySpace>
std::unique_ptr<PreparedHierarchyTensorSolver<Dim, MemorySpace>>
prepare_hierarchy_tensor_solver_collectively(
    const HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>& registry,
    std::string_view provider_identity, HierarchyTensorSolverBuildRequest<Dim> request,
    const ExecutionLane& lane) {
  using solver_type = PreparedHierarchyTensorSolver<Dim, MemorySpace>;
  std::shared_ptr<const HierarchyTensorSolverProvider<Dim, MemorySpace>> provider;
  std::string declaration;
  std::string request_contract;
  std::string support_contract;
  std::string expected_contract;
  PreparedProviderSupport support;
  long inspection_failure = 0;
  try {
    hierarchy_tensor_detail::validate_request(request);
    const auto& rank_space = request.levels.front().distribution.rank_space();
    const Index<Dim>& local_rank = request.levels.front().local_rank;
    if (rank_space.size() != static_cast<std::size_t>(lane.size()) ||
        rank_space.linear_rank(local_rank) != static_cast<std::size_t>(lane.rank()))
      throw std::invalid_argument(
          "hierarchy tensor local process coordinate differs from its execution lane");
    provider = registry.resolve(provider_identity);
    declaration = exact_hierarchy_tensor_solver_provider_declaration(*provider);
    request_contract = hierarchy_tensor_detail::request_contract(request);
    support = provider->supports(request);
    support_contract = exact_prepared_provider_support(support);
    if (support.accepted())
      expected_contract = provider->expected_prepared_contract(request);
  } catch (...) {
    inspection_failure = 1;
  }
  if (all_reduce_max(inspection_failure, lane) != 0)
    throw std::runtime_error("hierarchy tensor support inspection failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"hierarchy-tensor-provider", declaration},
                                                 {"hierarchy-tensor-request", request_contract},
                                                 {"hierarchy-tensor-support", support_contract},
                                                 {"hierarchy-tensor-expected", expected_contract}},
                                                lane))
    throw std::runtime_error("hierarchy tensor preparation contracts differ across MPI ranks");
  if (!support.accepted())
    throw std::invalid_argument("hierarchy tensor provider rejected request (code " +
                                std::to_string(support.code) + "): " + std::string(support.reason));

  std::unique_ptr<solver_type> prepared;
  long preparation_failure = 0;
  try {
    prepared = provider->prepare(request, lane);
    if (!prepared || prepared->exact_prepared_contract() != expected_contract ||
        prepared->provider_identity() != provider->identity() ||
        prepared->provider_version() != provider->interface_version())
      throw std::runtime_error("hierarchy tensor provider returned an unauthenticated solver");
    const PreparedProviderSupport execution =
        provider->accepts_execution(request, prepared->execution_path());
    if (!execution.accepted())
      throw std::invalid_argument("hierarchy tensor provider rejected its execution path");
    prepared->seal_preparation(lane);
  } catch (...) {
    preparation_failure = 1;
  }
  if (all_reduce_max(preparation_failure, lane) != 0)
    throw std::runtime_error("hierarchy tensor preparation failed on at least one MPI rank");
  return prepared;
}

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
std::shared_ptr<HierarchyTensorSolverProviderRegistry<Dim, MemorySpace>>
make_default_hierarchy_tensor_solver_provider_registry(const ExecutionLane& lane);

}  // namespace pops::runtime::program
