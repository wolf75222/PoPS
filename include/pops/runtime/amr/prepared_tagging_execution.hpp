/// @file
/// @brief Prepared compile-time-ranked execution of resolved AMR tagging bytecode.

#pragma once

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/amr/tagging/tag_mask.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

/// Exact-ranked authoring image consumed once by PreparedTaggingExecutionPlan::prepare().
template <int Dim>
struct PreparedTaggingProgram {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedTaggingProgram only supports dimensions 1, 2, and 3");

  struct AxisStencil {
    std::int32_t axis = 0;
    std::int32_t derivative_order = 0;
    std::int32_t formal_order = 0;
    std::size_t ghost_lower = 0;
    std::size_t ghost_upper = 0;
    std::vector<std::int32_t> offsets{};
    std::vector<double> coefficients{};
  };

  struct Stencil {
    std::string identity{};
    std::string route{};
    std::string norm{};
    std::string scale{};
    std::string boundary_mode{};
    std::array<AxisStencil, Dim> axes{};
  };

  struct Leaf {
    std::size_t state_index = 0;
    std::size_t component = 0;
    std::int32_t opcode = 0;
    double threshold = 0.0;
    std::size_t stencil_index = POPS_TAGGING_NO_STENCIL_V1;
  };

  std::vector<Stencil> stencils{};
  std::vector<Leaf> leaves{};
  std::vector<std::int32_t> refine_ops{};
  std::vector<std::int32_t> refine_args{};
  std::vector<std::int32_t> coarsen_ops{};
  std::vector<std::int32_t> coarsen_args{};
  std::int32_t minimum_cycles = 0;
  std::int32_t equality_policy = 0;
  std::int32_t conflict_policy = 0;
  std::int32_t non_finite_policy = POPS_TAGGING_NON_FINITE_REJECT_V1;
  std::string clock_identity{};
  std::string provider_identity{};
  bool prepared = false;
};

template <int Dim, class MemorySpace>
struct PreparedTaggingField {
  std::string qualified_identity{};
  const MultiFab<Dim, MemorySpace>* values = nullptr;
};

template <int Dim>
struct PreparedTaggerCandidates {
  ::pops::amr::tagging::TagMask<Dim> refine;
  ::pops::amr::tagging::TagMask<Dim> coarsen;
  ::pops::amr::tagging::TagMask<Dim> refine_equalities;
  ::pops::amr::tagging::TagMask<Dim> coarsen_equalities;
};

/// Every topology-owned byte needed by preparation is authorized explicitly. Four candidate masks
/// each receive `candidate_mask`; scratch covers one byte per locally visible cell, and replicated
/// consensus covers the two persistent min/max images used to prove rank equality before publish.
struct PreparedTaggingExecutionBudget {
  ::pops::amr::tagging::TagMaskBudget candidate_mask{};
  std::size_t scratch_bytes = 0;
  std::size_t replicated_consensus_bytes = 0;

  bool operator==(const PreparedTaggingExecutionBudget&) const = default;
};

namespace tagging_detail {

constexpr std::size_t kPreparedTaggingMaximumLeaves = POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1;
constexpr std::size_t kPreparedTaggingMaximumStencils = POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1;

enum PreparedTaggingMask : std::uint8_t {
  kRefineMatch = 1u << 0,
  kRefineEquality = 1u << 1,
  kCoarsenMatch = 1u << 2,
  kCoarsenEquality = 1u << 3,
  kNonFinite = 1u << 4,
};

enum class DeviceTagTruth : std::uint8_t { False = 0, True = 1, Unknown = 2 };

struct PreparedTaggingAxisDevice {
  std::int32_t axis = 0;
  std::int32_t term_count = 0;
  std::array<std::int32_t, POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1> offsets{};
  std::array<Real, POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1> coefficients{};
};

template <int Dim>
struct PreparedTaggingStencilDevice {
  std::array<PreparedTaggingAxisDevice, Dim> axes{};
};

struct PreparedTaggingLeafDevice {
  std::int32_t state_index = 0;
  std::int32_t component = 0;
  std::int32_t opcode = 0;
  Real threshold = Real(0);
  std::int32_t stencil_index = -1;
};

template <int Dim>
struct PreparedTaggingMaskView {
  std::uint8_t* values = nullptr;
  Box<Dim> box{};

  POPS_HD std::uint8_t& operator()(const Index<Dim>& index) const {
    std::int64_t linear = 0;
    std::int64_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      linear += (static_cast<std::int64_t>(index[axis]) - box.lo[axis]) * stride;
      stride *= box.length(axis);
    }
    return values[linear];
  }
};

template <int Dim>
struct PreparedTaggingPatchKernel {
  const PreparedTaggingLeafDevice* leaves = nullptr;
  const PreparedTaggingStencilDevice<Dim>* stencils = nullptr;
  const std::int32_t* refine_ops = nullptr;
  const std::int32_t* refine_args = nullptr;
  const std::int32_t* coarsen_ops = nullptr;
  const std::int32_t* coarsen_args = nullptr;
  const FieldView<const Real, Dim>* leaf_fields = nullptr;
  std::int32_t refine_count = 0;
  std::int32_t coarsen_count = 0;
  std::array<Real, Dim> spacing{};
  PreparedTaggingMaskView<Dim> mask{};

  POPS_HD static DeviceTagTruth tag_not(DeviceTagTruth value) {
    if (value == DeviceTagTruth::Unknown)
      return DeviceTagTruth::Unknown;
    return value == DeviceTagTruth::True ? DeviceTagTruth::False : DeviceTagTruth::True;
  }

  POPS_HD static bool is_leaf_opcode(std::int32_t opcode) {
    return opcode == POPS_TAGGING_ABOVE_V1 || opcode == POPS_TAGGING_BELOW_V1 ||
           opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1 || opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
           opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
  }

  POPS_HD DeviceTagTruth evaluate(const std::int32_t* ops, const std::int32_t* args,
                                  std::int32_t count, const Index<Dim>& index, bool& finite) const {
    if (count == 0)
      return DeviceTagTruth::False;
    std::array<DeviceTagTruth, POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1> stack{};
    std::int32_t depth = 0;
    for (std::int32_t instruction = 0; instruction < count; ++instruction) {
      const std::int32_t opcode = ops[instruction];
      const std::int32_t argument = args[instruction];
      if (is_leaf_opcode(opcode)) {
        const PreparedTaggingLeafDevice& leaf = leaves[argument];
        const FieldView<const Real, Dim> values = leaf_fields[argument];
        Real sample = Real(0);
        if (opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 || opcode == POPS_TAGGING_GRADIENT_BELOW_V1) {
          const PreparedTaggingStencilDevice<Dim>& stencil = stencils[leaf.stencil_index];
          Real squared_norm = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            const PreparedTaggingAxisDevice& derivative = stencil.axes[axis];
            Real value_on_axis = Real(0);
            for (std::int32_t term = 0; term < derivative.term_count; ++term) {
              Index<Dim> sample_index = index;
              sample_index[axis] += derivative.offsets[term];
              const Real value = values(sample_index, leaf.component);
              finite = finite && Kokkos::isfinite(value);
              value_on_axis += derivative.coefficients[term] * value;
            }
            value_on_axis /= spacing[axis];
            finite = finite && Kokkos::isfinite(value_on_axis);
            squared_norm += value_on_axis * value_on_axis;
          }
          sample = Kokkos::sqrt(squared_norm);
          finite = finite && Kokkos::isfinite(sample);
        } else {
          sample = values(index, leaf.component);
          finite = finite && Kokkos::isfinite(sample);
          if (opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1)
            sample = Kokkos::abs(sample);
        }
        const bool greater = opcode == POPS_TAGGING_ABOVE_V1 ||
                             opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1 ||
                             opcode == POPS_TAGGING_GRADIENT_ABOVE_V1;
        if (!finite)
          stack[depth++] = DeviceTagTruth::False;
        else if (sample == leaf.threshold)
          stack[depth++] = DeviceTagTruth::Unknown;
        else
          stack[depth++] = (greater ? sample > leaf.threshold : sample < leaf.threshold)
                               ? DeviceTagTruth::True
                               : DeviceTagTruth::False;
        continue;
      }
      if (opcode == POPS_TAGGING_NOT_V1) {
        stack[depth - 1] = tag_not(stack[depth - 1]);
        continue;
      }
      const std::int32_t begin = depth - argument;
      bool unknown = false;
      bool decisive = false;
      if (opcode == POPS_TAGGING_ANY_OF_V1) {
        for (std::int32_t child = begin; child < depth; ++child) {
          decisive = decisive || stack[child] == DeviceTagTruth::True;
          unknown = unknown || stack[child] == DeviceTagTruth::Unknown;
        }
        stack[begin] = decisive ? DeviceTagTruth::True
                                : (unknown ? DeviceTagTruth::Unknown : DeviceTagTruth::False);
      } else {
        for (std::int32_t child = begin; child < depth; ++child) {
          decisive = decisive || stack[child] == DeviceTagTruth::False;
          unknown = unknown || stack[child] == DeviceTagTruth::Unknown;
        }
        stack[begin] = decisive ? DeviceTagTruth::False
                                : (unknown ? DeviceTagTruth::Unknown : DeviceTagTruth::True);
      }
      depth = begin + 1;
    }
    return stack[0];
  }

  POPS_HD void operator()(const Index<Dim>& index) const {
    bool finite = true;
    const DeviceTagTruth refine = evaluate(refine_ops, refine_args, refine_count, index, finite);
    const DeviceTagTruth coarsen =
        evaluate(coarsen_ops, coarsen_args, coarsen_count, index, finite);
    std::uint8_t bits = finite ? std::uint8_t{0} : std::uint8_t{kNonFinite};
    if (refine == DeviceTagTruth::True)
      bits |= kRefineMatch;
    else if (refine == DeviceTagTruth::Unknown)
      bits |= kRefineEquality;
    if (coarsen == DeviceTagTruth::True)
      bits |= kCoarsenMatch;
    else if (coarsen == DeviceTagTruth::Unknown)
      bits |= kCoarsenEquality;
    mask(index) = bits;
  }
};

static_assert(std::is_trivially_copyable_v<PreparedTaggingAxisDevice>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingLeafDevice>);

inline std::size_t checked_sum(std::size_t left, std::size_t right, const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(message);
  return left + right;
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error(message);
  return left * right;
}

template <int Dim>
void append_index(ExactContractBuilder& contract, const Index<Dim>& index) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(index[axis]));
}

template <int Dim>
void append_box(ExactContractBuilder& contract, const Box<Dim>& box) {
  append_index(contract, box.lo);
  append_index(contract, box.hi);
}

template <int Dim>
void append_level(ExactContractBuilder& contract,
                  const ::pops::amr::hierarchy::LevelLayoutIdentity<Dim>& level) {
  contract.scalar(static_cast<std::int32_t>(level.level));
  append_box(contract, level.domain);
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(level.ratio_from_parent[axis]));
  append_index(contract, level.rank_space.origin());
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int64_t>(level.rank_space.extent()[axis]));
  contract.scalar(static_cast<std::uint8_t>(level.distribution_mode))
      .scalar(static_cast<std::uint64_t>(level.validation_budget.boxes))
      .scalar(static_cast<std::uint64_t>(level.validation_budget.overlap_pairs))
      .sequence(level.patches,
                [](ExactContractBuilder& item, const Box<Dim>& patch) { append_box(item, patch); })
      .sequence(level.owners, [](ExactContractBuilder& item, const Index<Dim>& owner) {
        append_index(item, owner);
      });
}

template <int Dim>
bool same_level_layout(const ::pops::amr::hierarchy::LevelLayoutIdentity<Dim>& expected,
                       const ::pops::amr::hierarchy::LevelLayout<Dim>& actual) noexcept {
  if (expected.level != actual.level() || expected.domain != actual.domain() ||
      expected.ratio_from_parent != actual.ratio_from_parent() ||
      expected.rank_space != actual.distribution().rank_space() ||
      expected.distribution_mode != actual.distribution().mode() ||
      expected.validation_budget != actual.validation_budget() ||
      expected.patches.size() != actual.patches().size() ||
      expected.owners.size() != actual.distribution().owners().size())
    return false;
  for (std::size_t patch = 0; patch < expected.patches.size(); ++patch)
    if (expected.patches[patch] != actual.patches()[patch])
      return false;
  for (std::size_t owner = 0; owner < expected.owners.size(); ++owner)
    if (expected.owners[owner] != actual.distribution().owners()[owner])
      return false;
  return true;
}

template <int Dim>
struct PreparedTaggingFieldContract {
  std::string qualified_identity{};
  std::int32_t component_count = 0;
  Extent<Dim> ghosts{};
};

template <int Dim>
std::string exact_program_contract(
    const PreparedTaggingProgram<Dim>& program,
    const std::vector<std::vector<PreparedTaggingFieldContract<Dim>>>& field_contracts,
    const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
    std::uint64_t topology_generation) {
  ExactContractBuilder contract;
  contract.text("pops.amr.prepared-tagging-execution")
      .scalar(std::uint32_t{2})
      .scalar(static_cast<std::int32_t>(Dim))
      .text(program.provider_identity)
      .text(program.clock_identity)
      .scalar(program.minimum_cycles)
      .scalar(program.equality_policy)
      .scalar(program.conflict_policy)
      .scalar(program.non_finite_policy)
      .scalar(topology_generation)
      .sequence(program.stencils,
                [](ExactContractBuilder& item,
                   const typename PreparedTaggingProgram<Dim>::Stencil& stencil) {
                  item.text(stencil.identity)
                      .text(stencil.route)
                      .text(stencil.norm)
                      .text(stencil.scale)
                      .text(stencil.boundary_mode);
                  for (const auto& axis : stencil.axes) {
                    item.scalar(axis.axis)
                        .scalar(axis.derivative_order)
                        .scalar(axis.formal_order)
                        .scalar(static_cast<std::uint64_t>(axis.ghost_lower))
                        .scalar(static_cast<std::uint64_t>(axis.ghost_upper))
                        .sequence(axis.offsets)
                        .sequence(axis.coefficients);
                  }
                })
      .sequence(
          program.leaves,
          [](ExactContractBuilder& item, const typename PreparedTaggingProgram<Dim>::Leaf& leaf) {
            item.scalar(static_cast<std::uint64_t>(leaf.state_index))
                .scalar(static_cast<std::uint64_t>(leaf.component))
                .scalar(leaf.opcode)
                .scalar(leaf.threshold)
                .scalar(static_cast<std::uint64_t>(leaf.stencil_index));
          })
      .sequence(program.refine_ops)
      .sequence(program.refine_args)
      .sequence(program.coarsen_ops)
      .sequence(program.coarsen_args)
      .sequence(field_contracts, [](ExactContractBuilder& item, const auto& level) {
        item.sequence(level, [](ExactContractBuilder& field, const auto& identity) {
          field.text(identity.qualified_identity).scalar(identity.component_count);
          for (int axis = 0; axis < Dim; ++axis)
            field.scalar(static_cast<std::int64_t>(identity.ghosts[axis]));
        });
      });
  for (const auto& layout : layouts)
    append_level(contract, layout.exact_identity());
  return std::move(contract).release();
}

inline std::array<std::size_t, 8> budget_fields(
    const PreparedTaggingExecutionBudget& budget) noexcept {
  return {budget.candidate_mask.global_patches,
          budget.candidate_mask.owned_patches,
          budget.candidate_mask.cells_per_patch,
          budget.candidate_mask.owned_cells,
          budget.candidate_mask.bytes,
          budget.candidate_mask.identity_bytes,
          budget.scratch_bytes,
          budget.replicated_consensus_bytes};
}

inline std::string exact_rank_ordered_budget_contract(
    const std::vector<PreparedTaggingExecutionBudget>& budgets,
    const CommunicatorView& communicator) {
  const long invalid_level_count =
      budgets.size() > static_cast<std::size_t>(std::numeric_limits<long>::max()) ? 1L : 0L;
  if (all_reduce_max(invalid_level_count, communicator) != 0)
    throw std::length_error("prepared AMR tagging budget level count exceeds long");
  const long local_level_count = static_cast<long>(budgets.size());
  if (all_reduce_min(local_level_count, communicator) !=
      all_reduce_max(local_level_count, communicator))
    throw std::invalid_argument(
        "prepared AMR tagging budget level count differs between ranks");
  const std::size_t ranks = static_cast<std::size_t>(communicator.size());
  constexpr std::size_t kFields = 8;
  std::vector<std::uint64_t> gathered;
  long allocation_failure = 0;
  try {
    gathered.resize(checked_product(
        checked_product(ranks, budgets.size(),
                        "prepared AMR tagging rank-budget matrix exceeds size_t"),
        kFields, "prepared AMR tagging rank-budget matrix exceeds size_t"));
  } catch (...) {
    allocation_failure = 1;
  }
  if (all_reduce_max(allocation_failure, communicator) != 0)
    throw std::bad_alloc();

  for (std::size_t source = 0; source < ranks; ++source)
    for (std::size_t level = 0; level < budgets.size(); ++level) {
      const auto fields = budget_fields(budgets[level]);
      for (std::size_t field = 0; field < fields.size(); ++field) {
        const std::uint64_t local =
            static_cast<std::size_t>(communicator.rank()) == source
                ? static_cast<std::uint64_t>(fields[field])
                : std::uint64_t{0};
        constexpr unsigned kChunkBits = 30;
        constexpr std::uint64_t kChunkMask = (std::uint64_t{1} << kChunkBits) - 1u;
        std::uint64_t exact = 0;
        for (unsigned chunk = 0; chunk < 3; ++chunk) {
          const long piece = all_reduce_max(
              static_cast<long>((local >> (chunk * kChunkBits)) & kChunkMask), communicator);
          exact |= static_cast<std::uint64_t>(piece) << (chunk * kChunkBits);
        }
        gathered[(source * budgets.size() + level) * kFields + field] = exact;
      }
    }

  ExactContractBuilder contract;
  contract.text("pops.amr.prepared-tagging-rank-budgets")
      .scalar(std::uint32_t{1})
      .scalar(static_cast<std::uint64_t>(ranks))
      .scalar(static_cast<std::uint64_t>(budgets.size()))
      .sequence(gathered);
  return std::move(contract).release();
}

}  // namespace tagging_detail

/// Persistent native execution image of one resolved exact-ranked AMR tagging graph.
///
/// Preparation owns bytecode, field views, candidate masks, and every scratch/consensus byte. The
/// hot path submits one static-rank Kokkos kernel per local patch and performs only allocation-free
/// scalar or fixed-buffer collectives. A failure leaves the last accepted candidate masks intact.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedTaggingExecutionPlan {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedTaggingExecutionPlan only supports dimensions 1, 2, and 3");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, MemorySpace>::accessible,
                "tagging execution space cannot access the selected field memory space");

 public:
  using Program = PreparedTaggingProgram<Dim>;
  using Field = PreparedTaggingField<Dim, MemorySpace>;
  using Candidates = PreparedTaggerCandidates<Dim>;
  using DeviceLeaf = tagging_detail::PreparedTaggingLeafDevice;
  using DeviceStencil = tagging_detail::PreparedTaggingStencilDevice<Dim>;

  static_assert(std::is_trivially_copyable_v<DeviceStencil>);
  static_assert(std::is_trivially_copyable_v<tagging_detail::PreparedTaggingPatchKernel<Dim>>,
                "prepared AMR tagging kernels must remain static-rank device-copyable values");

  PreparedTaggingExecutionPlan() = default;
  PreparedTaggingExecutionPlan(const PreparedTaggingExecutionPlan&) = delete;
  PreparedTaggingExecutionPlan& operator=(const PreparedTaggingExecutionPlan&) = delete;
  PreparedTaggingExecutionPlan(PreparedTaggingExecutionPlan&&) noexcept = default;
  PreparedTaggingExecutionPlan& operator=(PreparedTaggingExecutionPlan&&) noexcept = default;

  static PreparedTaggingExecutionPlan prepare(
      const Program& program, const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      const CommunicatorView& communicator = world_communicator_view()) {
    std::optional<PreparedTaggingExecutionPlan> candidate;
    std::exception_ptr local_error;
    try {
      candidate.emplace(prepare_local_(program, fields_by_level, layouts, budgets,
                                       topology_generation, communicator));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "prepared AMR tagging construction failed on another communicator rank");
    }

    std::string rank_budget_contract;
    local_error = nullptr;
    try {
      rank_budget_contract =
          tagging_detail::exact_rank_ordered_budget_contract(budgets, communicator);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "prepared AMR tagging rank-budget authentication failed on another rank");
    }
    local_error = nullptr;
    try {
      ExactContractBuilder collective;
      collective.text("pops.amr.prepared-tagging-collective")
          .scalar(std::uint32_t{1})
          .bytes(candidate->collective_contract_)
          .bytes(rank_budget_contract);
      candidate->collective_contract_ = std::move(collective).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, communicator) != 0) {
      if (local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(
          "prepared AMR tagging collective budget authentication failed on another rank");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-tagging"),
              std::string_view(candidate->collective_contract_)}},
            communicator))
      throw std::invalid_argument(
          "prepared AMR tagging program, fields, topology, or budgets differ between ranks");
    candidate->prepared_ = true;
    return std::move(*candidate);
  }

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept { return topology_generation_; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  const Candidates& execute(std::size_t level_index,
                            const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
                            const std::array<Real, Dim>& spacing,
                            std::uint64_t topology_generation) {
    long preflight_failure =
        !prepared_ || level_index >= levels_.size() || topology_generation != topology_generation_
            ? 1L
            : 0L;
    for (int axis = 0; axis < Dim; ++axis)
      if (!(spacing[axis] > Real(0)) || !std::isfinite(static_cast<double>(spacing[axis])))
        preflight_failure = 1;
    if (preflight_failure == 0 &&
        !tagging_detail::same_level_layout(levels_[level_index].identity, layout))
      preflight_failure = 1;
    if (all_reduce_max(preflight_failure, communicator_) != 0)
      throw std::runtime_error("prepared AMR tagging collective execution preflight failed");

    Level& level = levels_[level_index];
    for (Patch& patch : level.patches) {
      const tagging_detail::PreparedTaggingPatchKernel<Dim> kernel{
          leaves_.data(),
          stencils_.data(),
          refine_ops_.data(),
          refine_args_.data(),
          coarsen_ops_.data(),
          coarsen_args_.data(),
          patch.leaf_fields.data(),
          static_cast<std::int32_t>(refine_ops_.size()),
          static_cast<std::int32_t>(coarsen_ops_.size()),
          spacing,
          tagging_detail::PreparedTaggingMaskView<Dim>{patch.scratch.data(), patch.box}};
      for_each_cell(patch.box, kernel);
    }
    device_fence();

    long non_finite = 0;
    for (const Patch& patch : level.patches)
      for (const std::uint8_t bits : patch.scratch)
        if ((bits & tagging_detail::kNonFinite) != 0)
          non_finite = 1;
    if (all_reduce_max(non_finite, communicator_) != 0)
      throw std::runtime_error(
          "prepared AMR tagging rejected a non-finite indicator sample on at least one rank");

    if (level.replicated) {
      std::size_t offset = 0;
      for (const Patch& patch : level.patches)
        for (const std::uint8_t bits : patch.scratch) {
          const char value = static_cast<char>(bits);
          level.replica_min[offset] = value;
          level.replica_max[offset] = value;
          ++offset;
        }
      all_reduce_min_inplace(level.replica_min.data(), level.replica_min.size(), communicator_);
      all_reduce_max_inplace(level.replica_max.data(), level.replica_max.size(), communicator_);
      if (!std::equal(level.replica_min.begin(), level.replica_min.end(),
                      level.replica_max.begin()))
        throw std::runtime_error(
            "prepared AMR tagging replicated fields produced different masks between ranks");
    }

    for (const Patch& patch : level.patches) {
      for_each_host_index_(patch.box, [&](const Index<Dim>& index, std::size_t ordinal) {
        const std::uint8_t bits = patch.scratch[ordinal];
        level.candidates.refine.set(patch.global_patch, index,
                                    (bits & tagging_detail::kRefineMatch) != 0);
        level.candidates.refine_equalities.set(patch.global_patch, index,
                                               (bits & tagging_detail::kRefineEquality) != 0);
        level.candidates.coarsen.set(patch.global_patch, index,
                                     (bits & tagging_detail::kCoarsenMatch) != 0);
        level.candidates.coarsen_equalities.set(patch.global_patch, index,
                                                (bits & tagging_detail::kCoarsenEquality) != 0);
      });
    }
    return level.candidates;
  }

 private:
  template <class T>
  using DeviceVector = std::vector<T, fab_allocator<T>>;

  template <class T>
  using CommunicationVector = std::vector<T, comm_allocator<T>>;

  struct Patch {
    Box<Dim> box{};
    std::size_t global_patch = 0;
    DeviceVector<FieldView<const Real, Dim>> leaf_fields{};
    DeviceVector<std::uint8_t> scratch{};

    Patch(Box<Dim> valid, std::size_t global, std::size_t leaf_count)
        : box(valid),
          global_patch(global),
          leaf_fields(leaf_count),
          scratch(checked_cell_count_(valid), std::uint8_t{0}) {}
  };

  struct Level {
    ::pops::amr::hierarchy::LevelLayoutIdentity<Dim> identity{};
    std::vector<Patch> patches{};
    bool replicated = false;
    // These images are consumed directly by MPI.  Keep them in the canonical pinned-host
    // communication space rather than unified Fab storage: CUDA-aware MPI must never infer a
    // device IPC route for this prepared collective buffer.
    CommunicationVector<char> replica_min{};
    CommunicationVector<char> replica_max{};
    Candidates candidates;

    Level(const ::pops::amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& local_rank,
          const PreparedTaggingExecutionBudget& budget, std::size_t local_cells)
        : identity(layout.exact_identity()),
          replicated(layout.distribution().replicated()),
          replica_min(replicated ? local_cells : 0, char{0}),
          replica_max(replicated ? local_cells : 0, char{0}),
          candidates{
              ::pops::amr::tagging::TagMask<Dim>(layout, local_rank, budget.candidate_mask),
              ::pops::amr::tagging::TagMask<Dim>(layout, local_rank, budget.candidate_mask),
              ::pops::amr::tagging::TagMask<Dim>(layout, local_rank, budget.candidate_mask),
              ::pops::amr::tagging::TagMask<Dim>(layout, local_rank, budget.candidate_mask)} {}
  };

  static std::size_t checked_cell_count_(const Box<Dim>& box) {
    const std::int64_t cells = box.numPts();
    if (cells < 0 || static_cast<std::uint64_t>(cells) >
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::length_error("prepared AMR tagging patch cell count exceeds size_t");
    return static_cast<std::size_t>(cells);
  }

  template <class Function>
  static void for_each_host_index_(const Box<Dim>& box, Function&& function) {
    const std::size_t cells = checked_cell_count_(box);
    for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
      Index<Dim> index{};
      std::size_t quotient = ordinal;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t length = static_cast<std::size_t>(box.length(axis));
        index[axis] = static_cast<int>(static_cast<std::int64_t>(box.lo[axis]) +
                                       static_cast<std::int64_t>(quotient % length));
        quotient /= length;
      }
      function(index, ordinal);
    }
  }

  static void validate_bytecode_(const Program& program, const std::vector<std::int32_t>& ops,
                                 const std::vector<std::int32_t>& args, bool required) {
    if (ops.empty()) {
      if (required)
        throw std::invalid_argument("prepared AMR tagging has no refine root");
      return;
    }
    if (ops.size() != args.size())
      throw std::invalid_argument("prepared AMR tagging bytecode arrays differ in length");
    std::int32_t depth = 0;
    for (std::size_t instruction = 0; instruction < ops.size(); ++instruction) {
      const std::int32_t opcode = ops[instruction];
      const std::int32_t argument = args[instruction];
      if (pops_tagging_opcode_is_leaf_v1(opcode)) {
        if (argument < 0 || static_cast<std::size_t>(argument) >= program.leaves.size() ||
            program.leaves[static_cast<std::size_t>(argument)].opcode != opcode)
          throw std::invalid_argument("prepared AMR tagging bytecode has an invalid leaf");
        ++depth;
      } else if (opcode == POPS_TAGGING_NOT_V1) {
        if (argument != 1 || depth < 1)
          throw std::invalid_argument("prepared AMR tagging bytecode has an invalid NOT");
      } else if (opcode == POPS_TAGGING_ANY_OF_V1 || opcode == POPS_TAGGING_ALL_OF_V1) {
        if (argument < 2 || depth < argument)
          throw std::invalid_argument("prepared AMR tagging bytecode has an invalid arity");
        depth -= argument - 1;
      } else {
        throw std::invalid_argument("prepared AMR tagging bytecode has an unknown opcode");
      }
    }
    if (depth != 1)
      throw std::invalid_argument("prepared AMR tagging bytecode has an invalid final depth");
  }

  static PreparedTaggingExecutionPlan prepare_local_(
      const Program& program, const std::vector<std::vector<Field>>& fields_by_level,
      const std::vector<::pops::amr::hierarchy::LevelLayout<Dim>>& layouts,
      const std::vector<PreparedTaggingExecutionBudget>& budgets, std::uint64_t topology_generation,
      const CommunicatorView& communicator) {
    static_assert(sizeof(Real) == sizeof(double),
                  "prepared AMR Tagger ABI requires PoPS binary64 state storage");
    const std::size_t instruction_count =
        tagging_detail::checked_sum(program.refine_ops.size(), program.coarsen_ops.size(),
                                    "prepared AMR tagging instruction count exceeds size_t");
    if (!program.prepared || program.provider_identity.empty() || program.clock_identity.empty() ||
        program.leaves.empty() || fields_by_level.empty() ||
        fields_by_level.size() != layouts.size() || layouts.size() != budgets.size() ||
        topology_generation == 0 || program.minimum_cycles < 0 ||
        program.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1 ||
        program.equality_policy < 0 || program.equality_policy > 2 || program.conflict_policy < 0 ||
        program.conflict_policy > 3 ||
        program.leaves.size() > tagging_detail::kPreparedTaggingMaximumLeaves ||
        program.stencils.size() > tagging_detail::kPreparedTaggingMaximumStencils ||
        instruction_count > POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1)
      throw std::invalid_argument("prepared AMR tagging execution exceeds its authenticated ABI");
    validate_bytecode_(program, program.refine_ops, program.refine_args, true);
    validate_bytecode_(program, program.coarsen_ops, program.coarsen_args, false);

    PreparedTaggingExecutionPlan plan;
    plan.topology_generation_ = topology_generation;
    plan.communicator_ = communicator;
    plan.leaves_.reserve(program.leaves.size());
    plan.stencils_.reserve(program.stencils.size());
    plan.refine_ops_.assign(program.refine_ops.begin(), program.refine_ops.end());
    plan.refine_args_.assign(program.refine_args.begin(), program.refine_args.end());
    plan.coarsen_ops_.assign(program.coarsen_ops.begin(), program.coarsen_ops.end());
    plan.coarsen_args_.assign(program.coarsen_args.begin(), program.coarsen_args.end());

    for (const auto& source_stencil : program.stencils) {
      if (source_stencil.identity.empty() ||
          source_stencil.route != POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1 ||
          source_stencil.norm != "l2" || source_stencil.scale != "inverse_cell_size" ||
          source_stencil.boundary_mode != "ghost_extension")
        throw std::invalid_argument("prepared AMR tagging stencil route is not supported");
      DeviceStencil target_stencil;
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& source = source_stencil.axes[static_cast<std::size_t>(axis)];
        if (source.axis != axis || source.derivative_order != 1 || source.formal_order < 1 ||
            source.offsets.empty() ||
            static_cast<std::size_t>(source.formal_order) > source.offsets.size() ||
            source.offsets.size() > POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 ||
            source.offsets.size() != source.coefficients.size())
          throw std::invalid_argument("prepared AMR tagging stencil is not exact-ranked");
        std::size_t required_lower = 0;
        std::size_t required_upper = 0;
        auto& target = target_stencil.axes[static_cast<std::size_t>(axis)];
        target.axis = axis;
        target.term_count = static_cast<std::int32_t>(source.offsets.size());
        for (std::size_t term = 0; term < source.offsets.size(); ++term) {
          const std::int32_t offset = source.offsets[term];
          for (std::size_t previous = 0; previous < term; ++previous)
            if (source.offsets[previous] == offset)
              throw std::invalid_argument("prepared AMR tagging stencil repeats an offset");
          if (!std::isfinite(source.coefficients[term]))
            throw std::invalid_argument("prepared AMR tagging stencil coefficient is non-finite");
          if (offset < 0) {
            const std::int64_t magnitude = -static_cast<std::int64_t>(offset);
            required_lower = std::max(required_lower, static_cast<std::size_t>(magnitude));
          } else {
            required_upper = std::max(required_upper, static_cast<std::size_t>(offset));
          }
          target.offsets[term] = offset;
          target.coefficients[term] = static_cast<Real>(source.coefficients[term]);
        }
        if (required_lower != source.ghost_lower || required_upper != source.ghost_upper)
          throw std::invalid_argument(
              "prepared AMR tagging stencil ghost metadata does not match its offsets");
        for (std::int32_t power = 0; power <= source.formal_order; ++power) {
          double moment = 0.0;
          double scale = 0.0;
          for (std::size_t term = 0; term < source.offsets.size(); ++term) {
            const double value = source.coefficients[term] *
                                 std::pow(static_cast<double>(source.offsets[term]), power);
            moment += value;
            scale += std::abs(value);
          }
          const double expected = power == source.derivative_order ? 1.0 : 0.0;
          if (std::abs(moment - expected) > 1.0e-13 * std::max(1.0, scale))
            throw std::invalid_argument(
                "prepared AMR tagging stencil falsely declares its formal order");
        }
      }
      plan.stencils_.push_back(target_stencil);
    }

    for (const auto& leaf : program.leaves) {
      const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
      const bool has_stencil = leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1;
      if (!pops_tagging_opcode_is_leaf_v1(leaf.opcode) || !std::isfinite(leaf.threshold) ||
          gradient != has_stencil || (has_stencil && leaf.stencil_index >= plan.stencils_.size()) ||
          leaf.state_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          leaf.component > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("prepared AMR tagging has an invalid leaf descriptor");
      plan.leaves_.push_back(DeviceLeaf{
          static_cast<std::int32_t>(leaf.state_index), static_cast<std::int32_t>(leaf.component),
          leaf.opcode, static_cast<Real>(leaf.threshold),
          has_stencil ? static_cast<std::int32_t>(leaf.stencil_index) : -1});
    }

    std::vector<std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>>> field_contracts;
    field_contracts.reserve(fields_by_level.size());
    std::vector<std::string> canonical_field_identities;
    plan.levels_.reserve(layouts.size());
    for (std::size_t level_index = 0; level_index < layouts.size(); ++level_index) {
      const auto& layout = layouts[level_index];
      const auto& fields = fields_by_level[level_index];
      const auto& budget = budgets[level_index];
      if (fields.empty() || fields.front().values == nullptr ||
          fields.front().qualified_identity.empty())
        throw std::invalid_argument("prepared AMR tagging level has no qualified field authority");
      const MultiFab<Dim, MemorySpace>& reference = *fields.front().values;
      if (reference.layout() != layout.patches() ||
          reference.distribution() != layout.distribution())
        throw std::invalid_argument(
            "prepared AMR tagging field layout differs from the authenticated level");
      const std::size_t rank_count = layout.distribution().rank_space().size();
      if (communicator.size() < 1 || communicator.rank() < 0 ||
          static_cast<std::size_t>(communicator.size()) != rank_count ||
          layout.distribution().rank_space().linear_rank(reference.local_rank()) !=
              static_cast<std::size_t>(communicator.rank()))
        throw std::invalid_argument(
            "prepared AMR tagging rank coordinate differs from its communicator rank space");

      std::vector<std::string> identities;
      identities.reserve(fields.size());
      std::vector<tagging_detail::PreparedTaggingFieldContract<Dim>> contracts;
      contracts.reserve(fields.size());
      for (const Field& field : fields) {
        if (field.values == nullptr || field.qualified_identity.empty() ||
            field.values->layout() != reference.layout() ||
            field.values->distribution() != reference.distribution() ||
            field.values->local_rank() != reference.local_rank() ||
            field.values->local_global_indices() != reference.local_global_indices())
          throw std::invalid_argument(
              "prepared AMR tagging fields do not share one exact qualified layout");
        if (std::find(identities.begin(), identities.end(), field.qualified_identity) !=
            identities.end())
          throw std::invalid_argument("prepared AMR tagging field identity is not unique");
        identities.push_back(field.qualified_identity);
        contracts.push_back(
            {field.qualified_identity, field.values->ncomp(), field.values->ghosts()});
      }
      if (canonical_field_identities.empty())
        canonical_field_identities = identities;
      else if (identities != canonical_field_identities)
        throw std::invalid_argument(
            "prepared AMR tagging field identities change between hierarchy levels");
      field_contracts.push_back(std::move(contracts));

      for (const DeviceLeaf& leaf : plan.leaves_) {
        if (leaf.state_index < 0 || static_cast<std::size_t>(leaf.state_index) >= fields.size() ||
            leaf.component < 0 || leaf.component >= fields[leaf.state_index].values->ncomp())
          throw std::invalid_argument("prepared AMR tagging leaf lost its qualified field");
        if (leaf.stencil_index >= 0) {
          const auto& stencil = program.stencils[static_cast<std::size_t>(leaf.stencil_index)];
          const Extent<Dim>& ghosts = fields[leaf.state_index].values->ghosts();
          for (int axis = 0; axis < Dim; ++axis) {
            const auto& axis_stencil = stencil.axes[static_cast<std::size_t>(axis)];
            if (axis_stencil.ghost_lower > static_cast<std::size_t>(ghosts[axis]) ||
                axis_stencil.ghost_upper > static_cast<std::size_t>(ghosts[axis]))
              throw std::invalid_argument(
                  "prepared AMR tagging stencil exceeds the bound field halo");
          }
        }
      }

      std::size_t local_cells = 0;
      for (std::size_t local = 0; local < reference.local_size(); ++local)
        local_cells =
            tagging_detail::checked_sum(local_cells, checked_cell_count_(reference.box(local)),
                                        "prepared AMR tagging local scratch count exceeds size_t");
      if (local_cells > budget.scratch_bytes)
        throw std::length_error("prepared AMR tagging exceeds its explicit scratch-byte budget");
      const std::size_t consensus_bytes =
          layout.distribution().replicated()
              ? tagging_detail::checked_product(
                    local_cells, 2u,
                    "prepared AMR tagging replicated consensus bytes exceed size_t")
              : 0u;
      if (consensus_bytes > budget.replicated_consensus_bytes)
        throw std::length_error(
            "prepared AMR tagging exceeds its explicit replicated-consensus budget");

      Level level(layout, reference.local_rank(), budget, local_cells);
      level.patches.reserve(reference.local_size());
      for (std::size_t local = 0; local < reference.local_size(); ++local) {
        const std::size_t global_patch = reference.global_index(local);
        const Box<Dim>& valid = reference.box(local);
        if (!layout.domain().contains(valid) || valid != layout.patches()[global_patch])
          throw std::invalid_argument(
              "prepared AMR tagging local patch differs from its authenticated level");
        level.patches.emplace_back(valid, global_patch, plan.leaves_.size());
        Patch& patch = level.patches.back();
        for (std::size_t leaf_index = 0; leaf_index < plan.leaves_.size(); ++leaf_index) {
          const DeviceLeaf& leaf = plan.leaves_[leaf_index];
          const auto* field = fields[static_cast<std::size_t>(leaf.state_index)].values;
          if (field->global_index(local) != global_patch)
            throw std::invalid_argument(
                "prepared AMR tagging local field patch identities disagree");
          patch.leaf_fields[leaf_index] = field->fab(local).view();
        }
      }
      plan.levels_.push_back(std::move(level));
    }

    plan.collective_contract_ = tagging_detail::exact_program_contract(
        program, field_contracts, layouts, topology_generation);
    return plan;
  }

  bool prepared_ = false;
  std::uint64_t topology_generation_ = 0;
  CommunicatorView communicator_{};
  std::string collective_contract_{};
  DeviceVector<DeviceLeaf> leaves_{};
  DeviceVector<DeviceStencil> stencils_{};
  DeviceVector<std::int32_t> refine_ops_{};
  DeviceVector<std::int32_t> refine_args_{};
  DeviceVector<std::int32_t> coarsen_ops_{};
  DeviceVector<std::int32_t> coarsen_args_{};
  std::vector<Level> levels_{};
};

}  // namespace pops::runtime::amr
