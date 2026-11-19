#pragma once

#include <pops/numerics/diffusion/prepared_diffusion.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace pops::test {

inline constexpr const char* kPreparedConstitutiveAmrClock = "tests.amr.prepared-diffusion/macro";

// Exercise the same constitutive face -> conservative reflux adapter as generated Programs.
// The older default-provider path below has its own physical flux and cannot detect a reversed
// sign when +div(G) constitutive faces enter the shared -div(F) reflux ledger.
template <int Dim>
void install_prepared_constitutive_step(
    const std::shared_ptr<pops::runtime::program::AmrProgramContext<Dim>>& context) {
  // As in the generated AMR install, resolve these tuples on the loader's block map
  // before installing the step and consuming the candidate binding.
  // The shared RHS has distinct constitutive and transport temporal families, as in codegen.
  context->install_flux_temporal_families(
      {{0, 3000, 1, "tests.amr.prepared-diffusion/transport"},
       {0, 3000, 4, "tests.amr.prepared-diffusion/constitutive"}});
  context->install(
      [context](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context](double level_dt) {
          context->set_stage_time(0, 1);
          auto& state = context->state(0);
          auto& rhs = context->rhs_scratch(1000, 0, state);
          pops::runtime::program::PreparedDiffusion<Dim> prepared(*context, state, {}, true);
          context->prepare_generated_state(0, state, 3000);
          prepared.apply(state, rhs, [&](std::size_t local) {
            const auto values = std::as_const(state).fab(local).view();
            return [=] POPS_HD(const pops::Index<Dim>& cell) {
              std::array<pops::Real, Dim + 2> law{};
              law[0] = values(cell, 0);
              for (int axis = 0; axis < Dim; ++axis)
                law[axis + 1] = pops::Real(0.05);
              law[Dim + 1] = pops::Real(1);
              return law;
            };
          });
          context->attach_diffusive_flux_basis(0, rhs, 3000, prepared.faces(), pops::Real(1),
                                               "tests.amr.prepared-diffusion/constitutive");
          auto& transport = context->rhs_scratch(1000, 1, state);
          std::vector<pops::nd::FaceField<Dim>> transport_faces;
          context->neg_div_flux_default_with_faces_into(0, state, transport, 3000, transport_faces,
                                                        "tests.amr.prepared-diffusion/transport");
          context->axpy(rhs, pops::Real(1), transport, level_dt, {{0, 1, 1}});
          context->axpy(state, pops::Real(level_dt), rhs);
        });
      },
      context);
}

}  // namespace pops::test
