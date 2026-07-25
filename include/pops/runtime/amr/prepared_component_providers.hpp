#pragma once

#include <pops/amr/tagging/cluster.hpp>
#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/amr/tagging/tag_box.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/dynamic/prepared_execution_context.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

struct PreparedTaggerSpec {
  std::string provider_identity;
  std::string component_id;
  std::string manifest_identity;
  std::string layout_identity;
  std::string clock_identity;
  std::vector<std::int32_t> leaf_opcodes;
  std::vector<std::int32_t> logical_opcodes;
  std::vector<std::string> indicator_stencil_routes;
  std::size_t maximum_stencil_terms = 0;
  std::size_t maximum_instruction_count = 0;
  std::int32_t non_finite_policy = 0;
  PopsTaggerExecutionModeV2 execution_mode = POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2;
  PopsTaggerCollectiveScopeV2 collective_scope = POPS_TAGGER_COLLECTIVE_NONE_V2;
  std::vector<PopsMemorySpaceV1> memory_spaces;
  std::uint32_t interface_version = 2;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

/// Exact bound representation of the resolved AMRTagging graph.  External Taggers evaluate this
/// program; they are not an independent scientific policy.  PoPS remains the sole authority for
/// equality, refine/coarsen conflicts, current fine coverage and temporal hysteresis.
struct PreparedTaggingProgram {
  struct AxisStencil {
    std::int32_t axis = 0;
    std::int32_t derivative_order = 1;
    std::int32_t formal_order = 1;
    std::size_t ghost_lower = 0;
    std::size_t ghost_upper = 0;
    std::vector<std::int32_t> offsets;
    std::vector<double> coefficients;
  };
  struct Stencil {
    std::string identity;
    std::string route;
    std::string norm;
    std::string scale;
    std::string boundary_mode;
    std::int32_t dimension = 0;
    std::vector<AxisStencil> axes;
  };
  struct Leaf {
    std::size_t state_index = 0;
    std::size_t component = 0;
    std::int32_t opcode = 0;
    double threshold = 0.0;
    std::size_t stencil_index = POPS_TAGGING_NO_STENCIL_V1;
  };
  std::vector<Stencil> stencils;
  std::vector<Leaf> leaves;
  std::vector<std::int32_t> refine_ops, refine_args;
  std::vector<std::int32_t> coarsen_ops, coarsen_args;
  std::int32_t equality_policy = 0;
  std::int32_t conflict_policy = 0;
  std::int32_t min_cycles = 0;
  std::int32_t non_finite_policy = POPS_TAGGING_NON_FINITE_REJECT_V1;
  std::string clock_identity;
  std::string provider_identity;
  bool prepared = false;
};

/// One validation authority shared by the builtin VM and every external Tagger adapter.  The
/// caller supplies only its negotiated routes/capacity and a state-index -> allocated halo query.
template <class AvailableGhostDepth>
void validate_tagging_stencil_program(const PreparedTaggingProgram& program,
                                      const std::vector<std::string>& supported_routes,
                                      std::size_t maximum_stencil_terms,
                                      std::int32_t runtime_dimension,
                                      AvailableGhostDepth&& available_ghost_depth) {
  std::vector<bool> referenced(program.stencils.size(), false);
  for (const auto& leaf : program.leaves) {
    const bool gradient = leaf.opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                          leaf.opcode == POPS_TAGGING_GRADIENT_BELOW_V1;
    if (gradient != (leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1) ||
        (gradient && leaf.stencil_index >= program.stencils.size()))
      throw std::invalid_argument("AMR Tagger leaf lost its exact discrete stencil");
    if (gradient)
      referenced[leaf.stencil_index] = true;
  }
  for (std::size_t stencil_index = 0; stencil_index < program.stencils.size(); ++stencil_index) {
    const auto& stencil = program.stencils[stencil_index];
    if (!referenced[stencil_index] || stencil.identity.empty() ||
        std::find(supported_routes.begin(), supported_routes.end(), stencil.route) ==
            supported_routes.end() ||
        stencil.norm != "l2" || stencil.scale != "inverse_cell_size" ||
        stencil.boundary_mode != "ghost_extension" || stencil.dimension != runtime_dimension ||
        stencil.axes.size() != static_cast<std::size_t>(stencil.dimension))
      throw std::invalid_argument("AMR Tagger received an unsupported discrete stencil route");
    for (std::size_t axis_index = 0; axis_index < stencil.axes.size(); ++axis_index) {
      const auto& axis = stencil.axes[axis_index];
      if (axis.axis != static_cast<std::int32_t>(axis_index) || axis.derivative_order != 1 ||
          axis.formal_order < 1 || axis.offsets.empty() ||
          axis.offsets.size() != axis.coefficients.size() ||
          static_cast<std::size_t>(axis.formal_order) > axis.offsets.size() ||
          axis.offsets.size() > maximum_stencil_terms)
        throw std::invalid_argument("AMR Tagger received an invalid axis stencil");
      std::vector<std::int32_t> unique = axis.offsets;
      std::sort(unique.begin(), unique.end());
      if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
        throw std::invalid_argument("AMR Tagger axis stencil has duplicate offsets");
      std::size_t lower = 0, upper = 0;
      for (std::size_t term = 0; term < axis.offsets.size(); ++term) {
        const auto offset = axis.offsets[term];
        const auto widened_offset = static_cast<std::int64_t>(offset);
        const auto coefficient = axis.coefficients[term];
        if (!std::isfinite(coefficient))
          throw std::invalid_argument("AMR Tagger axis stencil has a non-finite coefficient");
        lower =
            std::max(lower, static_cast<std::size_t>(std::max<std::int64_t>(0, -widened_offset)));
        upper =
            std::max(upper, static_cast<std::size_t>(std::max<std::int64_t>(0, widened_offset)));
      }
      if (lower != axis.ghost_lower || upper != axis.ghost_upper)
        throw std::invalid_argument("AMR Tagger axis stencil has inconsistent halos");
      for (std::int32_t power = 0; power <= axis.formal_order; ++power) {
        double moment = 0.0, scale = 0.0;
        for (std::size_t term = 0; term < axis.offsets.size(); ++term) {
          const double value =
              axis.coefficients[term] * std::pow(static_cast<double>(axis.offsets[term]), power);
          moment += value;
          scale += std::abs(value);
        }
        const double expected = power == 1 ? 1.0 : 0.0;
        const double tolerance = 1.0e-13 * std::max(1.0, scale);
        if (std::abs(moment - expected) > tolerance)
          throw std::invalid_argument("AMR Tagger axis stencil falsely declares its formal order");
      }
    }
  }
  for (const auto& leaf : program.leaves)
    if (leaf.stencil_index != POPS_TAGGING_NO_STENCIL_V1) {
      const auto available = available_ghost_depth(leaf.state_index);
      for (const auto& axis : program.stencils[leaf.stencil_index].axes)
        if (axis.ghost_lower > available || axis.ghost_upper > available)
          throw std::invalid_argument("AMR Tagger field halo is thinner than its resolved stencil");
    }
}

struct PreparedTaggingField {
  std::string qualified_identity;
  MultiFab* values = nullptr;
};

struct PreparedTaggerCandidates {
  TagBox refine;
  TagBox coarsen;
  TagBox refine_equalities;
  TagBox coarsen_equalities;
};

struct PreparedClusteringSpec {
  std::string provider_identity;
  std::string component_id;
  std::string manifest_identity;
  std::string layout_identity;
  std::uint32_t interface_version = 1;
  std::shared_ptr<const component::PreparedExecutionContextV1> execution;
};

/// Prepared external Tagger.  One invocation per local patch sees every graph input as a qualified
/// borrowed SoA view and evaluates the exact resolved graph program.  Only four Boolean candidate
/// bitmaps are reduced across ranks; state arrays are never packed or globally reduced.
class PreparedTaggerComponent final {
 public:
  PreparedTaggerComponent(PreparedTaggerSpec spec,
                          std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate_();
    local_execution_ = std::make_shared<const component::PreparedExecutionContextV1>(
        spec_.execution->without_collective_authority());
    state_owner_ = component_->prepare_fresh_state(
        POPS_NATIVE_INTERFACE_TAGGER_V2, spec_.interface_version, local_execution_->view());
    state_ = state_owner_.get();
  }

  [[nodiscard]] const std::string& provider_identity() const noexcept {
    return spec_.provider_identity;
  }

  PreparedTaggerCandidates tag(const std::vector<PreparedTaggingField>& fields,
                               const PreparedTaggingProgram& program, const Box2D& domain,
                               int level, std::int64_t tick, double physical_time, double dx,
                               double dy, bool periodic_x, bool periodic_y,
                               bool parent_replicated) const {
    static_assert(sizeof(Real) == sizeof(double),
                  "Tagger ABI v2 requires the binary64 PoPS backend");
    PreparedTaggerCandidates result{TagBox(domain), TagBox(domain), TagBox(domain), TagBox(domain)};
    std::string local_failure;
    try {
      if (domain.empty() || fields.empty() || !program.prepared ||
          program.provider_identity.empty() || !std::isfinite(dx) || !std::isfinite(dy) ||
          dx <= 0.0 || dy <= 0.0)
        throw std::invalid_argument("external AMR Tagger received an incomplete graph evaluation");
      const std::size_t points = static_cast<std::size_t>(domain.num_cells());
      validate_program_(program, fields);
      validate_layout_(fields, domain);
      std::vector<std::vector<PopsTaggingAxisStencilV1>> abi_axes(program.stencils.size());
      std::vector<PopsTaggingStencilV1> stencils;
      stencils.reserve(program.stencils.size());
      for (std::size_t index = 0; index < program.stencils.size(); ++index) {
        const auto& stencil = program.stencils[index];
        auto& axes = abi_axes[index];
        axes.reserve(stencil.axes.size());
        for (const auto& axis : stencil.axes)
          axes.push_back(PopsTaggingAxisStencilV1{
              sizeof(PopsTaggingAxisStencilV1), axis.axis, axis.derivative_order, axis.formal_order,
              axis.ghost_lower, axis.ghost_upper, axis.offsets.size(), axis.offsets.data(),
              axis.coefficients.data()});
        stencils.push_back(PopsTaggingStencilV1{
            sizeof(PopsTaggingStencilV1), stencil.identity.c_str(), stencil.route.c_str(),
            stencil.norm.c_str(), stencil.scale.c_str(), stencil.boundary_mode.c_str(),
            stencil.dimension, axes.size(), axes.data()});
      }
      std::vector<PopsTaggingLeafV1> leaves;
      leaves.reserve(program.leaves.size());
      for (const auto& leaf : program.leaves)
        leaves.push_back(PopsTaggingLeafV1{sizeof(PopsTaggingLeafV1), leaf.state_index,
                                           leaf.component, leaf.opcode, leaf.threshold,
                                           leaf.stencil_index});
      const PopsTaggingProgramV1 abi_program{sizeof(PopsTaggingProgramV1),
                                             program.provider_identity.c_str(),
                                             stencils.size(),
                                             stencils.data(),
                                             leaves.size(),
                                             leaves.data(),
                                             program.refine_ops.size(),
                                             program.refine_ops.data(),
                                             program.refine_args.data(),
                                             program.coarsen_ops.size(),
                                             program.coarsen_ops.data(),
                                             program.coarsen_args.data(),
                                             program.min_cycles,
                                             program.equality_policy,
                                             program.conflict_policy,
                                             program.non_finite_policy};
      const auto& api = component_->table<PopsTaggerApiV2>(POPS_NATIVE_INTERFACE_TAGGER_V2,
                                                           spec_.interface_version);
      const bool native_execution =
          spec_.execution_mode == POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2;
      const PopsMemorySpaceV1 field_memory_space =
          native_execution ? native_storage_memory_space_() : POPS_MEMORY_SPACE_HOST_V1;
      MultiFab& reference = *fields.front().values;
      for (int local = 0; local < reference.local_size(); ++local) {
        const Box2D valid = reference.box(local);
        const int global = reference.global_index(local);
        const std::size_t local_points = static_cast<std::size_t>(valid.num_cells());
        std::vector<std::vector<Real>> host_storage;
        std::vector<ConstArray4> invocation_values;
        host_storage.reserve(fields.size());
        invocation_values.reserve(fields.size());
        for (const auto& field : fields) {
          const Fab2D& source = field.values->fab(local);
          const ConstArray4 values = source.const_array();
          if (native_execution) {
            invocation_values.push_back(values);
          } else {
            // Host execution is a declared provider capability, never an implicit fallback.  Only
            // this exact mode stages a host image of the state.
            host_storage.emplace_back(static_cast<std::size_t>(source.size()));
            using SharedView = Kokkos::View<const Real*, Kokkos::SharedSpace,
                                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            using HostView = Kokkos::View<Real*, Kokkos::HostSpace,
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
            const SharedView shared(values.p, host_storage.back().size());
            HostView host(host_storage.back().data(), host_storage.back().size());
            Kokkos::deep_copy(host, shared);
            invocation_values.push_back(ConstArray4{host.data(), values.nx_tot, values.comp_stride,
                                                     values.ig0, values.jg0});
          }
        }
        std::vector<std::string> patch_identities;
        std::vector<PopsQualifiedConstFieldV1> states;
        patch_identities.reserve(fields.size());
        states.reserve(fields.size());
        for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
          const auto& field = fields[field_index];
          const ConstArray4 values = invocation_values[field_index];
          patch_identities.push_back(field.qualified_identity + "@level=" + std::to_string(level) +
                                     "/patch=" + std::to_string(global));
          const std::size_t ghosts = static_cast<std::size_t>(field.values->n_grow());
          const PopsConstFieldViewV1 view{sizeof(PopsConstFieldViewV1),
                                          values.p,
                                          2,
                                          {static_cast<std::size_t>(valid.nx()) + 2u * ghosts,
                                           static_cast<std::size_t>(valid.ny()) + 2u * ghosts, 1},
                                          {1, static_cast<std::ptrdiff_t>(values.nx_tot), 0},
                                          static_cast<std::size_t>(field.values->ncomp()),
                                          static_cast<std::ptrdiff_t>(values.comp_stride),
                                          POPS_FIELD_CENTERING_CELL_V1,
                                          0,
                                          {ghosts, ghosts, 0},
                                          {ghosts, ghosts, 0},
                                          POPS_SCALAR_FLOAT64_V1,
                                          field_memory_space,
                                          spec_.layout_identity.c_str(),
                                          patch_identities.back().c_str(),
                                          POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
          states.push_back(PopsQualifiedConstFieldV1{sizeof(PopsQualifiedConstFieldV1), 1,
                                                     field.qualified_identity.c_str(), view});
        }
        if (local_points > std::numeric_limits<std::size_t>::max() / 4u)
          throw std::overflow_error("external AMR Tagger compact mask size overflow");
        const std::size_t mask_points = 4u * local_points;
        if (mask_points > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
          throw std::overflow_error("external AMR Tagger mask launch range overflow");
        std::vector<std::uint8_t> compact_masks(mask_points, std::uint8_t{0xff});
        using NativeMask = Kokkos::View<std::uint8_t*, Kokkos::SharedSpace>;
        NativeMask native_masks;
        if (native_execution) {
          native_masks = NativeMask("pops_external_tagger_masks", mask_points);
          Kokkos::deep_copy(native_masks, std::uint8_t{0xff});
        }
        const auto mask_view = [&](std::size_t output) {
          return PopsTaggerMaskViewV2{
              sizeof(PopsTaggerMaskViewV2),
              native_execution ? native_masks.data() + output * local_points
                               : compact_masks.data() + output * local_points,
              local_points,
              field_memory_space,
              POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1};
        };
        const PopsLogicalTimeV1 logical_time{sizeof(PopsLogicalTimeV1),
                                             spec_.clock_identity.c_str(),
                                             tick,
                                             level,
                                             0,
                                             0,
                                             0,
                                             1,
                                             0.0,
                                             physical_time};
        const PopsTaggerRequestV2 request{
            sizeof(PopsTaggerRequestV2),
            spec_.execution_mode,
            spec_.collective_scope,
            states.size(),
            states.data(),
            abi_program,
            {valid.lo[0], valid.lo[1], 0},
            {domain.lo[0], domain.lo[1], 0},
            {domain.hi[0], domain.hi[1], 0},
            {dx, dy, 0.0},
            static_cast<std::uint32_t>((periodic_x ? 1u : 0u) | (periodic_y ? 2u : 0u)),
            mask_view(0),
            mask_view(1),
            mask_view(2),
            mask_view(3),
            logical_time,
            local_execution_->view()};
        PopsComponentStatusV1 status = component::unwritten_component_status();
        const int code = component::tag_batch(api, state_, request, status);
        if (!component::component_status_is_well_formed(status) || code != 0 || status.code != 0 ||
            status.action != POPS_COMPONENT_CONTINUE_V1)
          throw std::runtime_error(status.reason == nullptr ? "native AMR Tagger failed"
                                                            : status.reason);
        if (native_execution) {
          std::size_t invalid_values = 0;
          using MaskPolicy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                                                  Kokkos::IndexType<std::int64_t>>;
          Kokkos::parallel_reduce(
              "pops_external_tagger_validate_masks",
              MaskPolicy(0, static_cast<std::int64_t>(mask_points)),
              KOKKOS_LAMBDA(std::int64_t point, std::size_t& invalid) {
                invalid += native_masks(static_cast<std::size_t>(point)) > 1u ? 1u : 0u;
              },
              invalid_values);
          if (invalid_values != 0)
            throw std::runtime_error("native AMR Tagger returned a non-Boolean candidate value");
          using HostMask = Kokkos::View<std::uint8_t*, Kokkos::HostSpace,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
          HostMask host(compact_masks.data(), mask_points);
          Kokkos::deep_copy(host, native_masks);
        } else if (std::any_of(compact_masks.begin(), compact_masks.end(),
                               [](std::uint8_t value) { return value > 1u; })) {
          throw std::runtime_error("host AMR Tagger returned a non-Boolean candidate value");
        }
        std::array<TagBox*, 4> outputs{&result.refine, &result.coarsen, &result.refine_equalities,
                                       &result.coarsen_equalities};
        const std::size_t patch_width = static_cast<std::size_t>(valid.nx());
        const std::size_t domain_width = static_cast<std::size_t>(domain.nx());
        const std::size_t domain_x_offset =
            static_cast<std::size_t>(valid.lo[0] - domain.lo[0]);
        for (std::size_t output = 0; output < outputs.size(); ++output)
          for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
            const std::size_t patch_row = static_cast<std::size_t>(j - valid.lo[1]);
            const std::size_t domain_row = static_cast<std::size_t>(j - domain.lo[1]);
            const auto* source = compact_masks.data() + output * local_points +
                                 patch_row * patch_width;
            auto* destination = outputs[output]->t.data() + domain_row * domain_width +
                                domain_x_offset;
            std::copy_n(source, patch_width, destination);
          }
      }
    } catch (const std::exception& error) {
      local_failure = error.what();
    } catch (...) {
      local_failure = "unknown native AMR Tagger failure";
    }
    const long failure_count = all_reduce_sum(local_failure.empty() ? 0L : 1L);
    if (failure_count != 0)
      throw std::runtime_error(n_ranks() == 1 ? local_failure
                                              : "native AMR Tagger failed on at least one rank");
    // Exchange all four compact candidate bitmaps in a constant number of collectives.  A
    // replicated parent is an exact-consensus contract, not an invitation to hide divergent
    // providers with a union.  A distributed parent instead gathers the disjoint local evidence.
    const std::array<TagBox*, 4> outputs{&result.refine, &result.coarsen, &result.refine_equalities,
                                         &result.coarsen_equalities};
    if (outputs.front()->t.size() > std::numeric_limits<std::size_t>::max() / outputs.size())
      throw std::overflow_error("native AMR Tagger consensus payload overflow");
    const std::size_t points = outputs.front()->t.size();
    std::vector<char> payload(points * outputs.size());
    for (std::size_t output = 0; output < outputs.size(); ++output)
      std::copy(outputs[output]->t.begin(), outputs[output]->t.end(),
                payload.begin() + static_cast<std::ptrdiff_t>(output * points));
    if (parent_replicated) {
      std::vector<char> minimum = payload;
      std::vector<char> maximum = payload;
      all_reduce_min_inplace(minimum.data(), minimum.size());
      all_reduce_max_inplace(maximum.data(), maximum.size());
      if (minimum != maximum)
        throw std::runtime_error(
            "native AMR Tagger returned rank-dependent masks for a replicated parent");
    } else {
      all_reduce_or_inplace(payload.data(), payload.size());
      for (std::size_t output = 0; output < outputs.size(); ++output)
        std::copy(payload.begin() + static_cast<std::ptrdiff_t>(output * points),
                  payload.begin() + static_cast<std::ptrdiff_t>((output + 1) * points),
                  outputs[output]->t.begin());
    }
    return result;
  }

 private:
  static PopsMemorySpaceV1 native_storage_memory_space_() noexcept {
    if constexpr (std::is_same_v<Kokkos::SharedSpace, Kokkos::HostSpace>)
      return POPS_MEMORY_SPACE_HOST_V1;
    return POPS_MEMORY_SPACE_MANAGED_V1;
  }

  void validate_program_(const PreparedTaggingProgram& program,
                         const std::vector<PreparedTaggingField>& fields) const {
    const auto supported = [](std::int32_t opcode, const std::vector<std::int32_t>& values) {
      return std::find(values.begin(), values.end(), opcode) != values.end();
    };
    if (program.min_cycles != 0)
      throw std::invalid_argument(
          "external AMR Tagger minimum_cycles requires native persistent tagging state");
    if (program.non_finite_policy != spec_.non_finite_policy ||
        program.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1 ||
        program.clock_identity != spec_.clock_identity || program.leaves.empty() ||
        program.refine_ops.empty() || program.refine_ops.size() != program.refine_args.size() ||
        program.coarsen_ops.size() != program.coarsen_args.size() ||
        program.refine_ops.size() + program.coarsen_ops.size() > spec_.maximum_instruction_count)
      throw std::invalid_argument("external AMR Tagger graph exceeds negotiated capacity");
    for (const auto& leaf : program.leaves) {
      if (leaf.state_index >= fields.size() || fields[leaf.state_index].values == nullptr ||
          leaf.component >= static_cast<std::size_t>(fields[leaf.state_index].values->ncomp()) ||
          !pops_tagging_opcode_is_leaf_v1(leaf.opcode) ||
          !supported(leaf.opcode, spec_.leaf_opcodes) || !std::isfinite(leaf.threshold))
        throw std::invalid_argument("external AMR Tagger graph has an unsupported leaf");
    }
    validate_tagging_stencil_program(
        program, spec_.indicator_stencil_routes, spec_.maximum_stencil_terms, 2,
        [&fields](std::size_t state_index) {
          return static_cast<std::size_t>(fields[state_index].values->n_grow());
        });
    for (const auto& opcodes : {&program.refine_ops, &program.coarsen_ops})
      for (const std::int32_t opcode : *opcodes)
        if (!((pops_tagging_opcode_is_leaf_v1(opcode) && supported(opcode, spec_.leaf_opcodes)) ||
              (pops_tagging_opcode_is_logical_v1(opcode) &&
               supported(opcode, spec_.logical_opcodes))))
          throw std::invalid_argument("external AMR Tagger graph has an unsupported opcode");
  }

  static void validate_layout_(const std::vector<PreparedTaggingField>& fields,
                               const Box2D& domain) {
    const MultiFab* reference = fields.front().values;
    if (reference == nullptr || reference->box_array().size() == 0)
      throw std::invalid_argument("external AMR Tagger has no parent patch layout");
    TagBox ownership(domain);
    for (int global = 0; global < reference->box_array().size(); ++global) {
      const Box2D& box = reference->box_array()[global];
      if (box.empty() || !domain.contains(box.lo[0], box.lo[1]) ||
          !domain.contains(box.hi[0], box.hi[1]))
        throw std::invalid_argument("external AMR Tagger parent patch lies outside its domain");
      const int owner = reference->dmap()[global];
      if (owner < 0 || owner >= n_ranks())
        throw std::invalid_argument("external AMR Tagger parent patch has an invalid owner");
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          if (ownership(i, j) != 0)
            throw std::invalid_argument("external AMR Tagger parent patches overlap");
          ownership(i, j) = 1;
        }
    }
    for (const auto& field : fields) {
      if (field.values == nullptr || field.qualified_identity.empty() ||
          field.values->ncomp() < 1 ||
          field.values->box_array().boxes() != reference->box_array().boxes() ||
          field.values->dmap().ranks() != reference->dmap().ranks() ||
          field.values->local_size() != reference->local_size())
        throw std::invalid_argument(
            "external AMR Tagger inputs do not share one exact patch layout");
      for (int local = 0; local < reference->local_size(); ++local)
        if (field.values->global_index(local) != reference->global_index(local))
          throw std::invalid_argument(
              "external AMR Tagger inputs disagree on local patch ownership");
    }
  }

  void validate_() const {
    if (!component_ || !spec_.execution || spec_.provider_identity.empty() ||
        spec_.component_id.empty() || spec_.manifest_identity.empty() ||
        spec_.layout_identity.empty() || spec_.clock_identity.empty() ||
        spec_.leaf_opcodes.empty() || spec_.logical_opcodes.empty() ||
        spec_.indicator_stencil_routes.empty() || spec_.maximum_stencil_terms == 0 ||
        spec_.maximum_stencil_terms > POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 ||
        spec_.maximum_instruction_count == 0 ||
        spec_.maximum_instruction_count > POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1 ||
        spec_.non_finite_policy != POPS_TAGGING_NON_FINITE_REJECT_V1 ||
        (spec_.execution_mode != POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2 &&
         spec_.execution_mode != POPS_TAGGER_EXECUTION_HOST_V2) ||
        spec_.collective_scope != POPS_TAGGER_COLLECTIVE_NONE_V2 ||
        spec_.memory_spaces.empty() || spec_.interface_version != 2)
      throw std::invalid_argument("prepared AMR Tagger specification is incomplete");
    std::vector<PopsMemorySpaceV1> memory_spaces = spec_.memory_spaces;
    std::sort(memory_spaces.begin(), memory_spaces.end());
    if (std::adjacent_find(memory_spaces.begin(), memory_spaces.end()) != memory_spaces.end() ||
        std::any_of(memory_spaces.begin(), memory_spaces.end(), [](PopsMemorySpaceV1 space) {
          return space != POPS_MEMORY_SPACE_HOST_V1 &&
                 space != POPS_MEMORY_SPACE_MANAGED_V1 &&
                 space != POPS_MEMORY_SPACE_DEVICE_V1;
        }))
      throw std::invalid_argument("prepared AMR Tagger declares invalid memory spaces");
    if (spec_.execution_mode == POPS_TAGGER_EXECUTION_HOST_V2) {
      if (memory_spaces != std::vector<PopsMemorySpaceV1>{POPS_MEMORY_SPACE_HOST_V1})
        throw std::invalid_argument(
            "prepared host AMR Tagger must declare exactly HostSpace execution");
    } else {
      const PopsMemorySpaceV1 storage_space = native_storage_memory_space_();
      const PopsExecutionContextV1 execution = spec_.execution->view();
      if (execution.memory_space != storage_space ||
          std::find(memory_spaces.begin(), memory_spaces.end(), storage_space) ==
              memory_spaces.end())
        throw std::invalid_argument(
            "prepared native AMR Tagger does not match the runtime field memory space");
    }
    std::vector<std::string> stencil_routes = spec_.indicator_stencil_routes;
    std::sort(stencil_routes.begin(), stencil_routes.end());
    if (std::adjacent_find(stencil_routes.begin(), stencil_routes.end()) != stencil_routes.end() ||
        std::any_of(stencil_routes.begin(), stencil_routes.end(), [](const std::string& route) {
          return route != POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1;
        }))
      throw std::invalid_argument("prepared AMR Tagger declares an unsupported stencil route");
    for (const auto opcode : spec_.leaf_opcodes)
      if (!pops_tagging_opcode_is_leaf_v1(opcode))
        throw std::invalid_argument("prepared AMR Tagger declares an invalid leaf opcode");
    for (const auto opcode : spec_.logical_opcodes)
      if (!pops_tagging_opcode_is_logical_v1(opcode))
        throw std::invalid_argument("prepared AMR Tagger declares an invalid logical opcode");
    component::validate_execution_context(spec_.execution->view());
    const auto& api = component_->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        spec_.component_id != api.component_id || spec_.manifest_identity != api.manifest_identity)
      throw std::invalid_argument("prepared AMR Tagger changed native component identity");
    component::require_operation(
        component_->table<PopsTaggerApiV2>(POPS_NATIVE_INTERFACE_TAGGER_V2, spec_.interface_version)
                .tag_batch != nullptr,
        "tag_batch");
  }

  PreparedTaggerSpec spec_;
  std::shared_ptr<component::LoadedComponent> component_;
  std::shared_ptr<const component::PreparedExecutionContextV1> local_execution_;
  component::LoadedComponent::PreparedState state_owner_;
  void* state_ = nullptr;
};

/// External Clustering ABI contract: each result is `2 * dimension` signed integers laid out as
/// `[lo_0, ..., lo_(d-1), hi_0, ..., hi_(d-1)]`, inclusive and relative to the supplied region.
class PreparedClusteringComponent final : public pops::amr::ClusteringProvider {
 public:
  PreparedClusteringComponent(PreparedClusteringSpec spec,
                              std::shared_ptr<component::LoadedComponent> component)
      : spec_(std::move(spec)), component_(std::move(component)) {
    validate_();
    state_ = component_->prepared_state(POPS_NATIVE_INTERFACE_CLUSTERING_V1,
                                        spec_.interface_version, spec_.execution->view());
  }

  [[nodiscard]] const std::string& provider_identity() const noexcept {
    return spec_.provider_identity;
  }

  std::vector<Box2D> cluster(const TagBox& tags) const override {
    const std::size_t tagged = static_cast<std::size_t>(tags.count());
    if (tagged == 0)
      return {};
    constexpr std::int32_t dimension = 2;
    if (tagged > std::numeric_limits<std::size_t>::max() / (2u * dimension))
      throw std::overflow_error("external AMR Clustering box capacity overflow");
    const std::int64_t extents[dimension] = {tags.box.nx(), tags.box.ny()};
    std::vector<std::uint8_t> mask;
    std::vector<std::int64_t> raw;
    std::size_t count = 0;
    std::string local_failure;
    try {
      mask.resize(tags.t.size());
      for (std::size_t index = 0; index < tags.t.size(); ++index)
        mask[index] = tags.t[index] == 0 ? 0u : 1u;
      raw.assign(tagged * 2u * dimension, std::numeric_limits<std::int64_t>::min());
      const PopsClusteringRequestV1 request{sizeof(PopsClusteringRequestV1),
                                            {sizeof(PopsConstByteViewV1), mask.data(), mask.size()},
                                            extents,
                                            dimension,
                                            raw.data(),
                                            tagged,
                                            &count,
                                            spec_.execution->view()};
      PopsComponentStatusV1 status = component::unwritten_component_status();
      const auto& api = component_->table<PopsClusteringApiV1>(POPS_NATIVE_INTERFACE_CLUSTERING_V1,
                                                               spec_.interface_version);
      const int code = component::cluster_tags(api, state_, request, status);
      if (!component::component_status_is_well_formed(status) || code != 0 || status.code != 0 ||
          status.action != POPS_COMPONENT_CONTINUE_V1)
        local_failure =
            status.reason == nullptr ? "native AMR Clustering component failed" : status.reason;
    } catch (const std::exception& error) {
      local_failure = error.what();
    } catch (...) {
      local_failure = "unknown native AMR Clustering failure";
    }
    if (all_reduce_sum(local_failure.empty() ? 0L : 1L) != 0)
      throw std::runtime_error(
          n_ranks() == 1 ? local_failure : "native AMR Clustering failed on at least one rank");
    std::vector<Box2D> result;
    local_failure.clear();
    try {
      if (count > tagged)
        throw std::runtime_error(
            "native AMR Clustering returned more boxes than its exact capacity");
      result.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        const auto* row = raw.data() + index * 2u * dimension;
        if (row[0] < 0 || row[1] < 0 || row[2] < row[0] || row[3] < row[1] ||
            row[2] >= extents[0] || row[3] >= extents[1] ||
            row[0] > std::numeric_limits<int>::max() || row[1] > std::numeric_limits<int>::max() ||
            row[2] > std::numeric_limits<int>::max() || row[3] > std::numeric_limits<int>::max())
          throw std::runtime_error(
              "native AMR Clustering returned an out-of-region or invalid box");
        result.push_back(Box2D{
            {static_cast<int>(row[0]) + tags.box.lo[0], static_cast<int>(row[1]) + tags.box.lo[1]},
            {static_cast<int>(row[2]) + tags.box.lo[0],
             static_cast<int>(row[3]) + tags.box.lo[1]}});
      }
      // Canonicalize before every structural comparison and before the boxes can be published.
      std::sort(result.begin(), result.end(), [](const Box2D& left, const Box2D& right) {
        return std::tie(left.lo[0], left.lo[1], left.hi[0], left.hi[1]) <
               std::tie(right.lo[0], right.lo[1], right.hi[0], right.hi[1]);
      });
      // One dense ownership bitmap proves both non-overlap and tag coverage in O(domain + covered
      // area + boxes). The former pairwise/none_of checks were O(B^2 + T*B), pathological for a
      // checkerboard provider allowed to return one box per tag.
      std::vector<std::uint8_t> covered(mask.size(), 0);
      const std::size_t nx = static_cast<std::size_t>(tags.box.nx());
      for (const Box2D& box : result)
        for (int j = box.lo[1]; j <= box.hi[1]; ++j)
          for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
            const std::size_t point = static_cast<std::size_t>(j - tags.box.lo[1]) * nx +
                                      static_cast<std::size_t>(i - tags.box.lo[0]);
            if (covered[point] != 0)
              throw std::runtime_error(
                  "native AMR Clustering returned duplicate or overlapping parent boxes");
            covered[point] = 1;
          }
      for (std::size_t point = 0; point < mask.size(); ++point)
        if (mask[point] != 0 && covered[point] == 0)
          throw std::runtime_error(
              "native AMR Clustering failed to cover every tagged parent cell");
    } catch (const std::exception& error) {
      local_failure = error.what();
    } catch (...) {
      local_failure = "unknown native AMR Clustering validation failure";
    }
    if (all_reduce_sum(local_failure.empty() ? 0L : 1L) != 0)
      throw std::runtime_error(
          n_ranks() == 1 ? local_failure
                         : "native AMR Clustering validation failed on at least one rank");
    const long local_count = static_cast<long>(result.size());
    if (all_reduce_min(local_count) != all_reduce_max(local_count))
      throw std::runtime_error("native AMR Clustering returned a different box count across ranks");
    std::vector<long> coordinates;
    coordinates.reserve(result.size() * 4u);
    for (const Box2D& box : result) {
      coordinates.push_back(box.lo[0]);
      coordinates.push_back(box.lo[1]);
      coordinates.push_back(box.hi[0]);
      coordinates.push_back(box.hi[1]);
    }
    std::vector<long> minimum = coordinates;
    std::vector<long> maximum = coordinates;
    all_reduce_min_inplace(minimum.data(), minimum.size());
    all_reduce_max_inplace(maximum.data(), maximum.size());
    if (minimum != maximum)
      throw std::runtime_error("native AMR Clustering returned different boxes across ranks");
    return result;
  }

 private:
  void validate_() const {
    if (!component_ || !spec_.execution || spec_.provider_identity.empty() ||
        spec_.component_id.empty() || spec_.manifest_identity.empty() ||
        spec_.layout_identity.empty() || spec_.interface_version != 1)
      throw std::invalid_argument("prepared AMR Clustering specification is incomplete");
    component::validate_execution_context(spec_.execution->view());
    const auto& api = component_->api();
    if (api.component_id == nullptr || api.manifest_identity == nullptr ||
        spec_.component_id != api.component_id || spec_.manifest_identity != api.manifest_identity)
      throw std::invalid_argument("prepared AMR Clustering changed native component identity");
    component::require_operation(
        component_
                ->table<PopsClusteringApiV1>(POPS_NATIVE_INTERFACE_CLUSTERING_V1,
                                             spec_.interface_version)
                .cluster != nullptr,
        "cluster");
  }

  PreparedClusteringSpec spec_;
  std::shared_ptr<component::LoadedComponent> component_;
  void* state_ = nullptr;
};

}  // namespace pops::runtime::amr
