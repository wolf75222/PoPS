#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/state/state.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pops::test::positivity {
inline constexpr char kProviderOwner[] = "test.amr.positivity";
inline constexpr char kProviderSpaceKind[] = "input";
inline constexpr char kProviderSpaceName[] = "transport";
inline constexpr char kProviderComponent[] = "speed";
inline constexpr char kProviderRepresentation[] = "cell-average";
inline constexpr char kProviderCentering[] = "cell";
inline constexpr char kProviderUnit[] = "length/time";
inline constexpr char kProviderLayout[] = "amr-transport";
inline constexpr char kProviderValueKind[] = "scalar";
inline constexpr char kProviderIdentity[] = "test.amr.positivity.speed@1";
inline constexpr char kProviderConsumer[] = "test.amr.positivity/physical-flux";

template <int Dim>
struct DensityAdvection {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 1;
  static constexpr int n_flux_providers = 1;
  static constexpr std::array<pops::QualifiedProviderRequirement, 1> flux_provider_requirements{{
      {kProviderOwner, kProviderSpaceKind, kProviderSpaceName, kProviderComponent,
       kProviderRepresentation, kProviderCentering, kProviderUnit, kProviderLayout,
       kProviderValueKind, kProviderIdentity, true, 0},
  }};

  Law conversion{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.density-advection", 2};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim});
  }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"rho"}, 1, {pops::VariableRole::Density}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"rho"}, 1, {pops::VariableRole::Density}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return conversion.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return conversion.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return conversion.admissibility(state);
  }

  template <int Axis>
  POPS_HD State flux(const State& state,
                     const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers) const {
    static_assert(Axis >= 0 && Axis < Dim);
    return State{providers.template provider<0>() * state[Schema::scalar]};
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(
      const State&, const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers) const {
    static_assert(Axis >= 0 && Axis < Dim);
    const pops::Real speed = providers.template provider<0>();
    return speed < pops::Real(0) ? -speed : speed;
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State&,
                           const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers,
                           pops::Real& lower, pops::Real& upper) const {
    static_assert(Axis >= 0 && Axis < Dim);
    const pops::Real speed = providers.template provider<0>();
    lower = speed;
    upper = speed;
  }
  POPS_HD State source(const State&, const pops::ProviderValues<1>&) const { return State{}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

struct ProgramEvidence {
  int accepted_source_solves = 0;
  int nonzero_block_provider_binds = 0;
};

template <int Dim>
std::shared_ptr<ProgramEvidence> install_step(
    const std::shared_ptr<pops::runtime::program::AmrProgramContext<Dim>>& context) {
  auto evidence = std::make_shared<ProgramEvidence>();
  context->configure_primary_clock("test.amr.positivity.clock");
  context->install(
      [context, evidence](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, evidence](double level_dt) {
          const int selected = context->level();
          context->set_stage_time(0, 1);
          std::vector<pops::MultiFab<Dim>*> states;
          std::vector<pops::MultiFab<Dim>*> residuals;
          states.reserve(static_cast<std::size_t>(context->n_blocks()));
          residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
          for (int block = 0; block < context->n_blocks(); ++block) {
            pops::MultiFab<Dim>& state = context->state(block);
            if (selected > 0) {
              const auto provider_at = [context, block](std::size_t local_patch) {
                return context->template provider_values_view<1>(kProviderConsumer, block,
                                                                 local_patch);
              };
              if (block != 0)
                ++evidence->nonzero_block_provider_binds;
              pops::SolveOutcome source = pops::backward_euler_source(
                  DensityAdvection<Dim>{}, provider_at, state, pops::Real(level_dt),
                  pops::NewtonOptions{}, context->prepared_execution_lane());
              const pops::SolveReport accepted = source.consume(pops::SolveConsumption::kAccept);
              if (!accepted.solved())
                throw std::runtime_error("positivity test Program source solve failed: " +
                                         accepted.reason);
              ++evidence->accepted_source_solves;
            }
            pops::MultiFab<Dim>& residual = context->rhs_scratch(1000 + block, 0, state);
            context->rhs_into(block, state, residual, 3000 + block);
            states.push_back(&state);
            residuals.push_back(&residual);
          }
          for (std::size_t block = 0; block < states.size(); ++block)
            context->axpy(*states[block], pops::Real(level_dt), *residuals[block]);
        });
      },
      context);
  return evidence;
}

}  // namespace pops::test::positivity
