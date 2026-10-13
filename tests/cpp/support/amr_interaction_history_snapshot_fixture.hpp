#pragma once

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pops::runtime::program::detail {
// Negative-fixture access only. Production consumers still use issued keeper carriers.
template <int Dim>
struct AmrInteractionHistorySnapshotTestAccess {
  using Context = AmrProgramContext<Dim>;
  using Image = typename Context::InteractionHistorySnapshot;
  using Declaration = typename Context::InteractionHistoryDeclaration;
  static auto& manager(Context& context) { return context.runtime_state().hist_; }
  static std::string key(Context& context, int level) {
    return context.history_key_("tracer.prior", level);
  }
  static auto image(const Context& context) {
    return context.interaction_history_snapshots_.at({0, "tracer.prior"});
  }
  static auto require(const Context& context, std::int64_t id = 100) {
    return context.require_interaction_history_snapshot_(
        id, context.interaction_history_declarations_.at(id));
  }
  static std::size_t image_count(const Context& context) {
    return context.interaction_history_snapshots_.size();
  }
  static void revoke(const Context& context) { context.revoke_interaction_history_snapshots_(); }
  static void cancel_mapping(const Context& context) { context.cancel_mapping_hierarchy_(); }
  static auto* runtime(const Context& context) { return context.runtime_; }
  static void new_attempt(const Context& context) {
    (void)context.prepared_resources_.begin_attempt(context.prepared_execution_lane());
  }
  static void restore_old_image(const Context& context, std::shared_ptr<const Image> image) {
    context.interaction_history_snapshots_.insert_or_assign({0, "tracer.prior"}, std::move(image));
  }
  static Image& corrupt_image(const Context& context) { return const_cast<Image&>(*image(context)); }
  static void pending_remap(Context& context, int level) {
    context.pending_history_remaps_.try_emplace(key(context, level));
  }
  static void half_window(const Context& context) {
    context.current_interval_end_phase_ = {1, 2};
    context.current_dt_ /= 2.;
  }
};
}  // namespace pops::runtime::program::detail

namespace pops::test::interaction_history {
inline constexpr int Dim = pops::kNativeDimension;
using Field = pops::MultiFab<Dim>;
using Context = pops::runtime::program::AmrProgramContext<Dim>;
using Access = pops::runtime::program::detail::AmrInteractionHistorySnapshotTestAccess<Dim>;
inline constexpr std::uint64_t kBudget = 32U << 20;
inline constexpr const char* kName = "tracer.prior";
inline constexpr const char* kIdentity = "tests.history-macro-snapshot/operator@1";

struct ScalarModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim, n_vars = 1;
  Law law{};
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.history-macro-snapshot.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& proof) const {
    for (int axis = 0; axis < Dim; ++axis) proof.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD auto recover(const State& state) const { return law.recover(state); }
  POPS_HD auto make_conservative(const Primitive& state) const { return law.make_conservative(state); }
  POPS_HD auto admissibility(const State& state) const { return law.admissibility(state); }
  template <int Axis> POPS_HD State flux(const State& state) const { return law.template flux<Axis>(state); }
  template <int Axis> POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

inline void expect_constant(const Field& field, pops::Real expected, bool grown = false) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto view = field.fab(local).view();
    const auto box = grown ? field.fab(local).grown_box() : field.box(local);
    EXPECT_EQ(pops::for_each_cell_reduce_max(box, [=] POPS_HD(const pops::Index<Dim>& cell) {
      return std::bit_cast<pops::runtime::program::InteractionRealWord>(view(cell, 0)) ==
          std::bit_cast<pops::runtime::program::InteractionRealWord>(expected) ? pops::Real(0) : pops::Real(1);
    }), pops::Real(0));
  }
}

struct Fixture {
  pops::AmrSystem<Dim> system;
  std::shared_ptr<Context> context;
  std::function<void(double)> body;
  std::uint64_t budget;
  int ghosts;
  static pops::AmrSystemConfig<Dim> config() {
    pops::AmrSystemConfig<Dim> value;
    value.level_count = 2;
    value.regrid_every = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      value.shape[axis] = 8;
      value.transition_buffers.front()[axis] = 0;
      value.transition_lookaheads.front()[axis] = 0;
    }
    return value;
  }
  explicit Fixture(bool declare = true, int ratio = 1, std::uint64_t max_bytes = kBudget,
                   int history_ghosts = -1)
      : system(config()), budget(max_bytes), ghosts(history_ghosts) {
    pops::test::install_amr_runtime_authority(system, "tests.history-macro-snapshot/runtime");
    system.set_temporal_relations({ratio}, {1}, {"integral_only"});
    system.install_block_state_route("tracer", "state/tracer");
    pops::add_compiled_model<Dim>(system, "tracer", ScalarModel{}, "minmod", "rusanov",
        "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
        {}, {}, 0., static_cast<double>(pops::kWenoEpsilon), false,
        "tests.history-macro-snapshot/flux");
    std::size_t cells = 1, centre = 0, stride = 1;
    for (int axis = 0; axis < Dim; ++axis) { centre += 4 * stride; stride *= 8; cells *= 8; }
    std::vector<double> initial(cells, .25);
    initial[centre] = 1.;
    system.set_conservative_state("tracer", initial);
    pops::test::install_prepared_refine_coarsen_threshold(
        system, {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Above},
        {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Below},
        "tests.history-macro-snapshot/tagging");
    system.execute_prepared_tagging(0);
    (void)system.regrid_from_prepared_tagging(0);
    if (system.engine()->hierarchy().num_levels() != 2)
      throw std::runtime_error("snapshot witness requires two materialized levels");
    context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("clock.macro");
    context->declare_clock_relation("clock.macro", "clock.level.1", ratio);
    context->install([this](double dt) {
      context->advance_hierarchy(dt, [&](double local_dt) { if (body) body(local_dt); });
    }, context);
    system.set_program_block_map({0});
    using Budget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
    system.install_prepared_amr_program_flux_expression_budget(
        "tests.history-macro-snapshot/flux-budget", std::vector<Budget>(1, Budget{2, 1}), 0, 0);
    context->for_each_program_resource_level([&](int) {
      context->state(0).set_val(pops::Real(2));
      context->register_history(kName, 2, 1, 0, "tracer.U", "cell.scalar", "clock.macro", "none", ghosts);
    });
    if (declare) { declaration(100, 1); declaration(101, 2); }
  }
  void declaration(std::int64_t id, int lag, std::uint64_t bytes = 0) const {
    context->declare_spatial_interaction_history_source(id, 0, kName, lag,
        "tracer.U", "cell.scalar", "clock.macro", "none", bytes ? bytes : budget, kIdentity);
  }
  void store(pops::Real value) const {
    auto& sample = context->scalar_scratch(900, 0, context->state(0), 1,
        ghosts < 0 ? context->state(0).ghosts()[0] : ghosts);
    sample.set_val(value);
    context->store_history(kName, sample, 0);
    context->rotate_histories("clock.macro");
  }
  void warm() {
    body = [&](double) { store(pops::Real(7)); };
    system.step(.125);
    body = {};
  }
  Field& interaction(std::int64_t id = 100, int lag = 1) const {
    const std::array<int, 1> components{0};
    return context->spatial_interaction_history_owned(id, 0, context->history(kName, lag, 0),
        components, budget, kIdentity, "clock.macro", false, kName, lag,
        "tracer.U", "cell.scalar", "none", [] POPS_HD(const auto&, const auto&) { return pops::Real(1); });
  }
  // Independent weighted composite oracle, using actual active/coverage/volume masks.
  pops::Real composite_integral(pops::Real value) const {
    pops::Real local_sum = 0;
    context->for_each_program_resource_level([&](int level) {
      const auto& field = context->state(0);
      const auto* active = system.prepared_amr_block_level_active_mask(0, level);
      const auto& coverage = system.prepared_amr_block_level_coverage_mask(0, level);
      const auto geom = system.prepared_amr_level_geometry(level);
      pops::Real volume = 1;
      for (int axis = 0; axis < Dim; ++axis) volume *= geom.spacing(axis);
      for (std::size_t global = 0; global < field.layout().size(); ++global) {
        if (!field.contains_local(global) || (field.distribution().replicated() && pops::my_rank() != 0)) continue;
        const auto c = coverage.fab_global(global).view();
        const auto a = active ? active->fab_global(global).view() : pops::FieldView<const pops::Real, Dim>{};
        const bool has_active = active;
        local_sum += pops::for_each_cell_reduce_sum(field.layout()[global], [=] POPS_HD(const pops::Index<Dim>& cell) {
          return (!has_active || a(cell, 0) == 1) && c(cell, 0) == 1 ? volume * value : pops::Real(0);
        });
      }
    });
    return pops::all_reduce_sum(local_sum, context->prepared_execution_lane());
  }
  void expect_output(const Field& output, pops::Real expected) const {
    const auto& coverage = system.prepared_amr_block_level_coverage_mask(0, context->level());
    for (std::size_t local = 0; local < output.local_size(); ++local) {
      const auto result = output.fab(local).view(), mask = coverage.fab(local).view();
      EXPECT_EQ(pops::for_each_cell_reduce_max(output.box(local), [=] POPS_HD(const pops::Index<Dim>& cell) {
        return Kokkos::abs(result(cell, 0) - (mask(cell, 0) == 1 ? expected : pops::Real(0)));
      }), pops::Real(0));
    }
  }
};
}  // namespace pops::test::interaction_history
