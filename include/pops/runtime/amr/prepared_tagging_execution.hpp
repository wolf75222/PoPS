#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/amr/prepared_component_providers.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

namespace tagging_detail {

constexpr std::size_t kPreparedTaggingMaximumLeaves =
    POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1;
constexpr std::size_t kPreparedTaggingMaximumStencils =
    POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1;
constexpr std::size_t kPreparedTaggingDimension = 2;

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

struct PreparedTaggingStencilDevice {
  std::int32_t axis_count = 0;
  std::array<PreparedTaggingAxisDevice, kPreparedTaggingDimension> axes{};
};

struct PreparedTaggingLeafDevice {
  std::int32_t state_index = 0;
  std::int32_t component = 0;
  std::int32_t opcode = 0;
  Real threshold = Real(0);
  std::int32_t stencil_index = -1;
};

struct PreparedTaggingMaskView {
  std::uint8_t* values = nullptr;
  int nx = 0;
  int lo_x = 0;
  int lo_y = 0;

  POPS_HD std::uint8_t& operator()(int i, int j) const {
    const std::int64_t row = static_cast<std::int64_t>(j) - lo_y;
    const std::int64_t column = static_cast<std::int64_t>(i) - lo_x;
    return values[row * nx + column];
  }
};

struct PreparedTaggingConstMaskView {
  const std::uint8_t* values = nullptr;
  int nx = 0;
  int lo_x = 0;
  int lo_y = 0;

  POPS_HD std::uint8_t operator()(int i, int j) const {
    const std::int64_t row = static_cast<std::int64_t>(j) - lo_y;
    const std::int64_t column = static_cast<std::int64_t>(i) - lo_x;
    return values[row * nx + column];
  }
};

struct PreparedTaggingCompactView {
  char* values = nullptr;
  int nx = 0;
  int lo_x = 0;
  int lo_y = 0;

  POPS_HD char& operator()(int i, int j) const {
    const std::int64_t row = static_cast<std::int64_t>(j) - lo_y;
    const std::int64_t column = static_cast<std::int64_t>(i) - lo_x;
    return values[row * nx + column];
  }
};

static_assert(std::is_trivially_copyable_v<PreparedTaggingAxisDevice>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingStencilDevice>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingLeafDevice>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingMaskView>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingConstMaskView>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingCompactView>);

struct PreparedTaggingPatchKernel {
  const PreparedTaggingLeafDevice* leaves = nullptr;
  const PreparedTaggingStencilDevice* stencils = nullptr;
  const std::int32_t* refine_ops = nullptr;
  const std::int32_t* refine_args = nullptr;
  const std::int32_t* coarsen_ops = nullptr;
  const std::int32_t* coarsen_args = nullptr;
  const ConstArray4* leaf_fields = nullptr;
  std::int32_t refine_count = 0;
  std::int32_t coarsen_count = 0;
  Real dx = Real(1);
  Real dy = Real(1);
  PreparedTaggingMaskView mask{};

  POPS_HD static DeviceTagTruth tag_not(DeviceTagTruth value) {
    if (value == DeviceTagTruth::Unknown)
      return DeviceTagTruth::Unknown;
    return value == DeviceTagTruth::True ? DeviceTagTruth::False : DeviceTagTruth::True;
  }

  POPS_HD DeviceTagTruth evaluate(const std::int32_t* ops, const std::int32_t* args,
                                  std::int32_t count, int i, int j, bool& finite) const {
    if (count == 0)
      return DeviceTagTruth::False;
    std::array<DeviceTagTruth, POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1> stack{};
    std::int32_t depth = 0;
    for (std::int32_t instruction = 0; instruction < count; ++instruction) {
      const std::int32_t opcode = ops[instruction];
      const std::int32_t argument = args[instruction];
      const bool leaf_opcode = opcode == POPS_TAGGING_ABOVE_V1 ||
                               opcode == POPS_TAGGING_BELOW_V1 ||
                               opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1 ||
                               opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                               opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
      if (leaf_opcode) {
        const PreparedTaggingLeafDevice& leaf = leaves[argument];
        const ConstArray4 values = leaf_fields[argument];
        Real sample = Real(0);
        if (opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
            opcode == POPS_TAGGING_GRADIENT_BELOW_V1) {
          const PreparedTaggingStencilDevice& stencil = stencils[leaf.stencil_index];
          Real squared_norm = Real(0);
          for (std::int32_t axis_index = 0; axis_index < stencil.axis_count; ++axis_index) {
            const PreparedTaggingAxisDevice& axis = stencil.axes[axis_index];
            Real derivative = Real(0);
            for (std::int32_t term = 0; term < axis.term_count; ++term) {
              const int x = axis.axis == 0 ? i + axis.offsets[term] : i;
              const int y = axis.axis == 1 ? j + axis.offsets[term] : j;
              const Real value = values(x, y, leaf.component);
              finite = finite && Kokkos::isfinite(value);
              derivative += axis.coefficients[term] * value;
            }
            derivative /= axis.axis == 0 ? dx : dy;
            finite = finite && Kokkos::isfinite(derivative);
            squared_norm += derivative * derivative;
          }
          sample = Kokkos::sqrt(squared_norm);
          finite = finite && Kokkos::isfinite(sample);
        } else {
          sample = values(i, j, leaf.component);
          finite = finite && Kokkos::isfinite(sample);
          if (opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1)
            sample = Kokkos::abs(sample);
        }
        const bool greater = opcode == POPS_TAGGING_ABOVE_V1 ||
                             opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1 ||
                             opcode == POPS_TAGGING_GRADIENT_ABOVE_V1;
        if (!finite) {
          stack[depth++] = DeviceTagTruth::False;
        } else if (sample == leaf.threshold) {
          stack[depth++] = DeviceTagTruth::Unknown;
        } else {
          stack[depth++] =
              (greater ? sample > leaf.threshold : sample < leaf.threshold)
                  ? DeviceTagTruth::True
                  : DeviceTagTruth::False;
        }
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

  POPS_HD void operator()(int i, int j) const {
    bool finite = true;
    const DeviceTagTruth refine = evaluate(refine_ops, refine_args, refine_count, i, j, finite);
    const DeviceTagTruth coarsen =
        evaluate(coarsen_ops, coarsen_args, coarsen_count, i, j, finite);
    std::uint8_t bits = finite ? std::uint8_t{0} : std::uint8_t{kNonFinite};
    if (refine == DeviceTagTruth::True)
      bits |= kRefineMatch;
    else if (refine == DeviceTagTruth::Unknown)
      bits |= kRefineEquality;
    if (coarsen == DeviceTagTruth::True)
      bits |= kCoarsenMatch;
    else if (coarsen == DeviceTagTruth::Unknown)
      bits |= kCoarsenEquality;
    mask(i, j) = bits;
  }
};

static_assert(std::is_trivially_copyable_v<PreparedTaggingPatchKernel>);

struct PreparedTaggingClearCompactKernel {
  PreparedTaggingCompactView compact{};

  POPS_HD void operator()(int i, int j) const { compact(i, j) = char{0}; }
};

struct PreparedTaggingCompactPatchKernel {
  PreparedTaggingConstMaskView mask{};
  PreparedTaggingCompactView compact{};

  POPS_HD void operator()(int i, int j) const {
    // Global valid patches are authenticated as non-overlapping during preparation, so assignment
    // is deterministic and needs no device atomic. MPI combines the disjoint rank images below.
    compact(i, j) = static_cast<char>(mask(i, j));
  }
};

static_assert(std::is_trivially_copyable_v<PreparedTaggingClearCompactKernel>);
static_assert(std::is_trivially_copyable_v<PreparedTaggingCompactPatchKernel>);

inline bool same_box(const Box2D& left, const Box2D& right) noexcept {
  return left.lo[0] == right.lo[0] && left.lo[1] == right.lo[1] &&
         left.hi[0] == right.hi[0] && left.hi[1] == right.hi[1];
}

}  // namespace tagging_detail

/// Persistent native execution image of one resolved AMR tagging graph.
///
/// Authoring vectors and qualified names are consumed once by prepare().  The hot path captures
/// only POD pointers, fixed-capacity bytecode and Array4 handles in a named Kokkos kernel.  Patch
/// masks and the dense rank-gather bitmap are topology-owned scratch and never allocate while a
/// fixed hierarchy is being tagged.
class PreparedTaggingExecutionPlan {
 public:
  using DeviceLeaf = tagging_detail::PreparedTaggingLeafDevice;
  using DeviceStencil = tagging_detail::PreparedTaggingStencilDevice;

  PreparedTaggingExecutionPlan() = default;
  PreparedTaggingExecutionPlan(const PreparedTaggingExecutionPlan&) = delete;
  PreparedTaggingExecutionPlan& operator=(const PreparedTaggingExecutionPlan&) = delete;
  PreparedTaggingExecutionPlan(PreparedTaggingExecutionPlan&&) noexcept = default;
  PreparedTaggingExecutionPlan& operator=(PreparedTaggingExecutionPlan&&) noexcept = default;

  static PreparedTaggingExecutionPlan prepare(
      const PreparedTaggingProgram& program,
      const std::vector<std::vector<PreparedTaggingField>>& fields_by_level,
      const std::vector<Box2D>& domains, std::uint64_t topology_generation) {
    static_assert(sizeof(Real) == sizeof(double),
                  "prepared AMR Tagger ABI requires PoPS binary64 state storage");
    if (!program.prepared || program.provider_identity.empty() || program.clock_identity.empty() ||
        program.leaves.empty() ||
        program.refine_ops.empty() || fields_by_level.empty() ||
        fields_by_level.size() != domains.size() || topology_generation == 0 ||
        program.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1 ||
        program.min_cycles != 0 || program.equality_policy < 0 || program.equality_policy > 2 ||
        program.conflict_policy < 0 || program.conflict_policy > 3 ||
        program.leaves.size() > tagging_detail::kPreparedTaggingMaximumLeaves ||
        program.stencils.size() > tagging_detail::kPreparedTaggingMaximumStencils ||
        program.refine_ops.size() != program.refine_args.size() ||
        program.coarsen_ops.size() != program.coarsen_args.size() ||
        program.refine_ops.size() + program.coarsen_ops.size() >
            POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1)
      throw std::invalid_argument("prepared AMR tagging execution exceeds its authenticated ABI");

    const auto validate_root = [&program](const std::vector<std::int32_t>& ops,
                                          const std::vector<std::int32_t>& args,
                                          bool required) {
      if (ops.empty()) {
        if (required)
          throw std::invalid_argument("prepared AMR tagging has no refine root");
        return;
      }
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
    };
    validate_root(program.refine_ops, program.refine_args, true);
    validate_root(program.coarsen_ops, program.coarsen_args, false);
    for (const auto& leaf : program.leaves) {
      const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
      if (!pops_tagging_opcode_is_leaf_v1(leaf.opcode) || !std::isfinite(leaf.threshold) ||
          gradient != (leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1) ||
          (gradient && leaf.stencil_index >= program.stencils.size()))
        throw std::invalid_argument("prepared AMR tagging has an invalid leaf descriptor");
    }

    PreparedTaggingExecutionPlan plan;
    plan.provider_identity_ = program.provider_identity;
    plan.clock_identity_ = program.clock_identity;
    plan.topology_generation_ = topology_generation;
    plan.leaves_.reserve(program.leaves.size());
    plan.stencils_.reserve(program.stencils.size());
    plan.refine_ops_.assign(program.refine_ops.begin(), program.refine_ops.end());
    plan.refine_args_.assign(program.refine_args.begin(), program.refine_args.end());
    plan.coarsen_ops_.assign(program.coarsen_ops.begin(), program.coarsen_ops.end());
    plan.coarsen_args_.assign(program.coarsen_args.begin(), program.coarsen_args.end());

    for (const auto& stencil : program.stencils) {
      if (stencil.axes.size() != tagging_detail::kPreparedTaggingDimension)
        throw std::invalid_argument("prepared AMR tagging requires an exact 2D stencil image");
      DeviceStencil device;
      device.axis_count = static_cast<std::int32_t>(stencil.axes.size());
      for (std::size_t axis_index = 0; axis_index < stencil.axes.size(); ++axis_index) {
        const auto& source = stencil.axes[axis_index];
        if (source.offsets.size() > POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 ||
            source.offsets.size() != source.coefficients.size())
          throw std::invalid_argument("prepared AMR tagging stencil exceeds its ABI capacity");
        auto& target = device.axes[axis_index];
        target.axis = source.axis;
        target.term_count = static_cast<std::int32_t>(source.offsets.size());
        for (std::size_t term = 0; term < source.offsets.size(); ++term) {
          target.offsets[term] = source.offsets[term];
          target.coefficients[term] = static_cast<Real>(source.coefficients[term]);
        }
      }
      plan.stencils_.push_back(device);
    }
    for (const auto& leaf : program.leaves) {
      const bool has_stencil = leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1;
      if (leaf.state_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          leaf.component > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          (has_stencil && leaf.stencil_index >= plan.stencils_.size()))
        throw std::invalid_argument("prepared AMR tagging leaf cannot be represented exactly");
      plan.leaves_.push_back(DeviceLeaf{
          static_cast<std::int32_t>(leaf.state_index), static_cast<std::int32_t>(leaf.component),
          leaf.opcode, static_cast<Real>(leaf.threshold),
          has_stencil ? static_cast<std::int32_t>(leaf.stencil_index) : -1});
    }

    plan.levels_.reserve(fields_by_level.size());
    for (std::size_t level_index = 0; level_index < fields_by_level.size(); ++level_index) {
      const auto& fields = fields_by_level[level_index];
      if (domains[level_index].empty() || fields.empty() || fields.front().values == nullptr ||
          fields.front().qualified_identity.empty())
        throw std::invalid_argument("prepared AMR tagging level has no qualified field authority");
      const MultiFab& reference = *fields.front().values;
      for (int global = 0; global < reference.box_array().size(); ++global) {
        const Box2D& box = reference.box_array()[global];
        if (box.empty() || !domains[level_index].contains(box.lo[0], box.lo[1]) ||
            !domains[level_index].contains(box.hi[0], box.hi[1]) ||
            reference.dmap()[global] < 0 || reference.dmap()[global] >= n_ranks())
          throw std::invalid_argument("prepared AMR tagging has an invalid global patch layout");
        for (int other = 0; other < global; ++other)
          if (!box.intersect(reference.box_array()[other]).empty())
            throw std::invalid_argument("prepared AMR tagging global patches overlap");
      }
      Level level(domains[level_index]);
      level.patches.reserve(static_cast<std::size_t>(reference.local_size()));
      if (level_index > 0 && fields.size() != plan.qualified_field_identities_.size())
        throw std::invalid_argument(
            "prepared AMR tagging field count changes between hierarchy levels");
      std::vector<std::string> identities;
      identities.reserve(fields.size());
      for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
        const auto& field = fields[field_index];
        if (field.values == nullptr || field.qualified_identity.empty() ||
            field.values->box_array().boxes() != reference.box_array().boxes() ||
            field.values->dmap().ranks() != reference.dmap().ranks() ||
            field.values->local_size() != reference.local_size())
          throw std::invalid_argument(
              "prepared AMR tagging fields do not share one exact qualified layout");
        if (std::find(identities.begin(), identities.end(), field.qualified_identity) !=
            identities.end())
          throw std::invalid_argument("prepared AMR tagging field identity is not unique");
        identities.push_back(field.qualified_identity);
        if (level_index == 0)
          plan.qualified_field_identities_.push_back(field.qualified_identity);
        else if (field_index >= plan.qualified_field_identities_.size() ||
                 field.qualified_identity != plan.qualified_field_identities_[field_index])
          throw std::invalid_argument(
              "prepared AMR tagging field identity changes between hierarchy levels");
      }
      for (const DeviceLeaf& leaf : plan.leaves_)
        if (leaf.state_index < 0 || static_cast<std::size_t>(leaf.state_index) >= fields.size() ||
            leaf.component < 0 || leaf.component >= fields[leaf.state_index].values->ncomp())
          throw std::invalid_argument("prepared AMR tagging leaf lost its qualified field");
      validate_tagging_stencil_program(
          program, std::vector<std::string>{POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1},
          POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1, 2, [&fields](std::size_t state_index) {
            if (state_index >= fields.size() || fields[state_index].values == nullptr)
              throw std::invalid_argument("prepared AMR tagging stencil lost its qualified field");
            return static_cast<std::size_t>(fields[state_index].values->n_grow());
          });
      for (int local = 0; local < reference.local_size(); ++local) {
        const Box2D valid = reference.box(local);
        if (valid.empty() || !domains[level_index].contains(valid.lo[0], valid.lo[1]) ||
            !domains[level_index].contains(valid.hi[0], valid.hi[1]))
          throw std::invalid_argument("prepared AMR tagging patch lies outside its level domain");
        Patch patch(valid, reference.global_index(local), plan.leaves_.size());
        for (std::size_t leaf_index = 0; leaf_index < plan.leaves_.size(); ++leaf_index) {
          const DeviceLeaf& leaf = plan.leaves_[leaf_index];
          MultiFab& field = *fields[static_cast<std::size_t>(leaf.state_index)].values;
          if (field.global_index(local) != patch.global_index)
            throw std::invalid_argument("prepared AMR tagging local patch identities disagree");
          patch.leaf_fields[leaf_index] = field.fab(local).const_array();
        }
        level.patches.push_back(std::move(patch));
      }
      plan.levels_.push_back(std::move(level));
    }
    plan.prepared_ = true;
    return plan;
  }

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }

  const PreparedTaggerCandidates& execute(int level_index, const Box2D& domain, Real dx, Real dy,
                                          std::uint64_t topology_generation) {
    std::uint64_t local_preflight_failure = 0;
    if (!prepared_ || level_index < 0 ||
        static_cast<std::size_t>(level_index) >= levels_.size() ||
        topology_generation != topology_generation_ || !(dx > Real(0)) || !(dy > Real(0)) ||
        !std::isfinite(static_cast<double>(dx)) || !std::isfinite(static_cast<double>(dy)))
      local_preflight_failure = 1;
    if (local_preflight_failure == 0 &&
        !tagging_detail::same_box(levels_[static_cast<std::size_t>(level_index)].domain, domain))
      local_preflight_failure = 1;
    if (all_reduce_max(local_preflight_failure) != 0)
      throw std::runtime_error("prepared AMR tagging collective execution preflight failed");

    Level& level = levels_[static_cast<std::size_t>(level_index)];
    for (Patch& patch : level.patches) {
      ::pops::detail::ensure_kokkos_initialized();
      const tagging_detail::PreparedTaggingPatchKernel kernel{
          leaves_.data(),
          stencils_.data(),
          refine_ops_.data(),
          refine_args_.data(),
          coarsen_ops_.data(),
          coarsen_args_.data(),
          patch.leaf_fields.data(),
          static_cast<std::int32_t>(refine_ops_.size()),
          static_cast<std::int32_t>(coarsen_ops_.size()),
          dx,
          dy,
          tagging_detail::PreparedTaggingMaskView{patch.mask.data(), patch.box.nx(),
                                                  patch.box.lo[0], patch.box.lo[1]}};
      Kokkos::parallel_for(
          "pops_amr_prepared_tagging_patch",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
              {patch.box.lo[0], patch.box.lo[1]},
              {patch.box.hi[0] + 1, patch.box.hi[1] + 1}),
          kernel);
    }
    const tagging_detail::PreparedTaggingCompactView compact{
        level.compact.data(), domain.nx(), domain.lo[0], domain.lo[1]};
    Kokkos::parallel_for(
        "pops_amr_prepared_tagging_clear_compact",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
            {domain.lo[0], domain.lo[1]}, {domain.hi[0] + 1, domain.hi[1] + 1}),
        tagging_detail::PreparedTaggingClearCompactKernel{compact});
    for (const Patch& patch : level.patches) {
      const tagging_detail::PreparedTaggingConstMaskView mask{
          patch.mask.data(), patch.box.nx(), patch.box.lo[0], patch.box.lo[1]};
      Kokkos::parallel_for(
          "pops_amr_prepared_tagging_compact_patch",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
              {patch.box.lo[0], patch.box.lo[1]},
              {patch.box.hi[0] + 1, patch.box.hi[1] + 1}),
          tagging_detail::PreparedTaggingCompactPatchKernel{mask, compact});
    }
    device_fence();
    all_reduce_or_inplace(level.compact.data(), level.compact.size());
    if (std::any_of(level.compact.begin(), level.compact.end(), [](char value) {
          return (static_cast<std::uint8_t>(value) & tagging_detail::kNonFinite) != 0;
        }))
      throw std::runtime_error(
          "prepared AMR tagging rejected a non-finite indicator sample on at least one rank");

    for (std::size_t index = 0; index < level.compact.size(); ++index) {
      const auto bits = static_cast<std::uint8_t>(level.compact[index]);
      level.candidates.refine.t[index] = (bits & tagging_detail::kRefineMatch) != 0;
      level.candidates.refine_equalities.t[index] =
          (bits & tagging_detail::kRefineEquality) != 0;
      level.candidates.coarsen.t[index] = (bits & tagging_detail::kCoarsenMatch) != 0;
      level.candidates.coarsen_equalities.t[index] =
          (bits & tagging_detail::kCoarsenEquality) != 0;
    }
    return level.candidates;
  }

 private:
  template <class T>
  using DeviceVector = std::vector<T, fab_allocator<T>>;

  struct Patch {
    Box2D box{};
    int global_index = -1;
    DeviceVector<ConstArray4> leaf_fields{};
    DeviceVector<std::uint8_t> mask{};

    Patch() = default;
    Patch(Box2D valid, int global, std::size_t leaf_count)
        : box(valid),
          global_index(global),
          leaf_fields(leaf_count),
          mask(static_cast<std::size_t>(valid.num_cells()), std::uint8_t{0}) {}
  };

  struct Level {
    Box2D domain{};
    std::vector<Patch> patches{};
    DeviceVector<char> compact{};
    PreparedTaggerCandidates candidates{};

    explicit Level(const Box2D& level_domain)
        : domain(level_domain),
          compact(static_cast<std::size_t>(level_domain.num_cells()), char{0}),
          candidates{TagBox(level_domain), TagBox(level_domain), TagBox(level_domain),
                     TagBox(level_domain)} {}
  };

  bool prepared_ = false;
  std::uint64_t topology_generation_ = 0;
  std::string provider_identity_{};
  std::string clock_identity_{};
  std::vector<std::string> qualified_field_identities_{};
  DeviceVector<DeviceLeaf> leaves_{};
  DeviceVector<DeviceStencil> stencils_{};
  DeviceVector<std::int32_t> refine_ops_{};
  DeviceVector<std::int32_t> refine_args_{};
  DeviceVector<std::int32_t> coarsen_ops_{};
  DeviceVector<std::int32_t> coarsen_args_{};
  std::vector<Level> levels_{};
};

}  // namespace pops::runtime::amr
