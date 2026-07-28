#pragma once

#include <pops/coupling/source/coupled_source.hpp>
#include <pops/coupling/system/system_coupler.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/numerics/time/integrators/time_steppers.hpp>
#include <pops/numerics/time/schemes/scheduler.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>

/// @file
/// @brief Test-only numerical oracle for the retired static coupled-system time driver.
///
/// Production `System` and `AmrSystem` execute only their installed `ProgramGraph`. This helper
/// preserves the former static-driver formulas for isolated numerical regression tests without
/// leaving a second scheme/cadence authority in installed PoPS headers.

namespace pops::test_support {

template <class>
inline constexpr bool reference_always_false_v = false;

template <CoupledSystemLike System, class RhsAssembler, class Elliptic = GeometricMG>
class ReferenceSystemDriver {
 public:
  template <class FactoryT = DefaultEllipticFactory<Elliptic>>
    requires pops::EllipticFactory<FactoryT, Elliptic>
  ReferenceSystemDriver(System system, const Geometry& geom, const BoxArray& ba,
                        const BCRec& bc_phi, RhsAssembler rhs_assembler,
                        ActiveRegionProvider2D active = {}, ScalarFieldProvider2D bz = {},
                        FactoryT elliptic_factory = {})
      : assembler_(std::move(system), geom, ba, bc_phi, std::move(rhs_assembler), std::move(active),
                   std::move(bz), std::move(elliptic_factory)) {}

  System& system() { return assembler_.system(); }
  const System& system() const { return assembler_.system(); }
  MultiFab& phi() { return assembler_.phi(); }
  const MultiFab& aux() const { return assembler_.aux(); }
  void solve_fields() { assembler_.solve_fields(); }
  SystemAssembler<System, RhsAssembler, Elliptic>& assembler() { return assembler_; }

  template <class ImplicitAdvance>
  void step(Real dt, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    advance_subcycled(assembler_.system(), dt, macro_step_,
                      [&](auto& block, Real h, int substep, int count) {
                        advance_block_dispatch(block, h, substep, count, advance_implicit);
                      });
    ++macro_step_;
  }

  template <class ImplicitAdvance>
  Real step_adaptive(Real cfl, ImplicitAdvance&& implicit_advance) {
    ImplicitAdvance& advance_implicit = implicit_advance;
    assembler_.solve_fields();
    const Real h = std::min(assembler_.geom().dx(), assembler_.geom().dy());
    const Real wmax = system_max_wave_speed();
    const Real macro_dt = cfl * h / std::max(wmax, kCflSpeedFloor);
    assembler_.system().for_each_block([&](auto& block) {
      using Block = std::decay_t<decltype(block)>;
      if constexpr (block_time_treatment_v<Block> != TimeTreatment::Prescribed) {
        const Real wave_speed = max_wave_speed_mf(block.model, block.U(), assembler_.aux());
        const int stride =
            wave_speed <= Real(0) ? 1 : std::max(1, static_cast<int>(wmax / wave_speed));
        if (macro_step_ % stride == 0) {
          constexpr int count = block_substeps_v<Block>;
          const Real h_substep = (macro_dt * static_cast<Real>(stride)) / static_cast<Real>(count);
          for (int substep = 0; substep < count; ++substep)
            advance_block_dispatch(block, h_substep, substep, count, advance_implicit);
        }
      }
    });
    ++macro_step_;
    return macro_dt;
  }

  Real step_adaptive(Real cfl) {
    return step_adaptive(cfl, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(reference_always_false_v<Block>,
                    "ReferenceSystemDriver::step_adaptive(cfl) cannot advance an implicit/IMEX "
                    "block without a callback");
    });
  }

  void step(Real dt) {
    step(dt, [](auto&, auto& block, Real, int, int) {
      using Block = std::decay_t<decltype(block)>;
      static_assert(reference_always_false_v<Block>,
                    "ReferenceSystemDriver::step(dt) cannot advance an implicit/IMEX block without "
                    "a callback");
    });
  }

  Real cfl_dt(Real cfl) {
    assembler_.solve_fields();
    const Real h = std::min(assembler_.geom().dx(), assembler_.geom().dy());
    return cfl * h / std::max(system_max_wave_speed(), kCflSpeedFloor);
  }

  template <class ImplicitAdvance>
  Real step_cfl(Real cfl, ImplicitAdvance&& implicit_advance) {
    const Real dt = cfl_dt(cfl);
    step(dt, std::forward<ImplicitAdvance>(implicit_advance));
    return dt;
  }

  Real step_cfl(Real cfl) {
    const Real dt = cfl_dt(cfl);
    step(dt);
    return dt;
  }

  template <class CoupledSource>
  void coupled_source_step(CoupledSource&& source, Real dt) {
    static_assert(CoupledSourceFor<std::decay_t<CoupledSource>, System>,
                  "coupled_source_step expects a CoupledSource: apply(system, aux, dt)");
    assembler_.solve_fields();
    source.apply(assembler_.system(), assembler_.aux(), dt);
  }

 private:
  Real system_max_wave_speed() {
    Real wmax = 0;
    assembler_.system().for_each_block([&](auto& block) {
      wmax = std::max(wmax, max_wave_speed_mf(block.model, block.U(), assembler_.aux()));
    });
    return wmax;
  }

  template <class Block, class ImplicitAdvance>
  void advance_block_dispatch(Block& block, Real dt, int substep, int count,
                              ImplicitAdvance& advance_implicit) {
    constexpr TimeTreatment treatment = block_time_treatment_v<Block>;
    if constexpr (treatment == TimeTreatment::Explicit) {
      advance_explicit_block(block, dt);
    } else if constexpr (treatment == TimeTreatment::Implicit || treatment == TimeTreatment::IMEX) {
      assembler_.solve_fields();
      if constexpr (treatment == TimeTreatment::IMEX)
        explicit_transport(block, dt);
      advance_implicit(*this, block, dt, substep, count);
    }
  }

  template <class Block>
  void advance_explicit_block(Block& block, Real dt) {
    using Time = TimePolicyTraits<typename Block::Time>;
    using Method = typename Time::Method;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    static_assert(Time::treatment == TimeTreatment::Explicit,
                  "advance_explicit_block expects an explicit block");

    auto rhs_eval = [&](MultiFab& stage, MultiFab& residual) {
      assembler_.template block_residual<Limiter, NumericalFlux>(block, stage, residual,
                                                                 /*recompute_aux=*/true);
    };
    if constexpr (std::is_same_v<Method, SSPRK3>)
      SSPRK3Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (std::is_same_v<Method, SSPRK2>)
      SSPRK2Step{}.take_step(rhs_eval, block.U(), dt);
    else if constexpr (TimeStepper<Method>)
      Method{}.take_step(rhs_eval, block.U(), dt);
    else
      static_assert(reference_always_false_v<Method>,
                    "explicit Method must be SSPRK2, SSPRK3, or a TimeStepper");
  }

  template <class Block>
  void explicit_transport(Block& block, Real dt) {
    using Model = typename Block::Model;
    using Limiter = typename Block::Spatial::Limiter;
    using NumericalFlux = typename Block::Spatial::NumericalFlux;
    const SourceFreeModel<Model> source_free{block.model};
    MultiFab residual(assembler_.ba(), assembler_.dm(), Model::n_vars, 0);
    fill_ghosts(block.U(), assembler_.geom().domain, block.bc);
    assemble_rhs<Limiter, NumericalFlux>(source_free, block.U(), assembler_.aux(),
                                         assembler_.geom(), residual);
    saxpy(block.U(), dt, residual);
  }

  SystemAssembler<System, RhsAssembler, Elliptic> assembler_;
  int macro_step_ = 0;
};

template <class... Args>
auto make_reference_system_driver(Args&&... args) {
  return ReferenceSystemDriver(std::forward<Args>(args)...);
}

}  // namespace pops::test_support
