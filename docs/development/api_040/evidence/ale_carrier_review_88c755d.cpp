// Independent bounded host probe, compiled against git-archive 88c755d headers.
// Does not instantiate System, run a PDE, or qualify the public ALE descriptor.
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/mesh/geometry/swept_interval.hpp>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

static pops::Real first(const pops::Fab<1>& field) {
  const auto host = field.create_host_mirror();
  field.copy_to_host(host);
  return host(0);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using namespace pops;
    using runtime::program::ProgramRuntimeState;
    const Box<1> box{Index<1>{0}, Index<1>{1}};
    const mesh::BoxArray<1> layout(std::vector<Box<1>>{box});
    const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
    const auto distribution = mesh::Distribution<1>::replicated(layout, ranks);
    ProgramRuntimeState<1> live;
    auto& geometry = live.moving_interval_geometry_["mesh"];
    geometry.runtime_block = 0;
    geometry.measures = MultiFab<1>(layout, distribution, Index<1>{0}, 1, Extent<1>{});
    geometry.measures.set_val(.5);
    geometry.coordinates.emplace_back(box, 1);
    geometry.coordinates.back().set_val(.25);
    geometry.swept_volumes.emplace_back(box, 1);
    geometry.swept_volumes.back().set_val(.125);
    live.last_dt_ = .1;
    live.diagnostics_["marker"] = 7;
    live.cache_.store(31, geometry.measures, 0);

    // Exactly the aggregate copy used by System::Impl::AcceptedSnapshot.
    ProgramRuntimeState<1> accepted(live);
    geometry.measures.set_val(9);
    geometry.coordinates.back().set_val(10);
    geometry.swept_volumes.back().set_val(11);
    geometry.generation = 8;
    geometry.last_interval = "trial";
    assert(first(accepted.moving_interval_geometry_.at("mesh").measures.fab(0)) == .5);
    assert(first(accepted.moving_interval_geometry_.at("mesh").coordinates[0].field<0>()) == .25);
    assert(first(accepted.moving_interval_geometry_.at("mesh").swept_volumes[0].field<0>()) == .125);

    auto prepared = live.prepare_accepted_restore(accepted);
    // The prepared restore must itself detach every Kokkos allocation.
    accepted.moving_interval_geometry_.at("mesh").measures.set_val(21);
    accepted.moving_interval_geometry_.at("mesh").coordinates[0].set_val(22);
    accepted.moving_interval_geometry_.at("mesh").swept_volumes[0].set_val(23);
    live.last_dt_ = .9;
    live.diagnostics_["marker"] = 99;
    live.publish_prepared_accepted_restore(std::move(prepared));
    const auto& restored = live.moving_interval_geometry_.at("mesh");
    assert(first(restored.measures.fab(0)) == .5);
    assert(first(restored.coordinates[0].field<0>()) == .25);
    assert(first(restored.swept_volumes[0].field<0>()) == .125);
    assert(restored.generation == 0 && restored.last_interval.empty());
    assert(live.last_dt_ == .1 && live.diagnostics_.at("marker") == 7);
    auto cache = restored.measures;
    live.cache_.restore_into(31, cache);
    assert(first(cache.fab(0)) == .5);
    std::cout << "deep_copy_prepared_restore_pass\n";

    // Finite inputs and exact GCL alone do not authenticate a physical state.
    const double volume = .5, density = 2.3;
    const double amount = SweptInterval::integrated_amount_update(
        density * volume, 0, 0, 0, 0, -(density + 1) * volume, 0, 0);
    const double candidate = amount / volume;
    assert(std::isfinite(candidate) && candidate < 0);
    std::cout << "finite_unrecoverable_candidate " << candidate << '\n';

    // Independent PDE oracle: U(t)=2.3+.4*t, F=.7*x, S=.4+.7.
    // Both projection integrals are exact for each linear face trajectory.
    for (int cells : {8, 24}) for (double h : {.1, .2, .3}) for (double t : {0., .4}) {
      double total_change = 0;
      for (int cell = 0; cell < cells; ++cell) {
        const double left = double(cell) / cells, right = double(cell + 1) / cells;
        const auto displacement = [&](double x) {
          return x == 0. || x == 1. ? 0. : .08 * h * std::sin(2. * std::acos(-1.) * x);
        };
        const double sl = displacement(left), sr = displacement(right);
        const auto interval = SweptInterval::prepare(left, right, left + sl, right + sr,
                                                     sl, sr, 1e-13);
        const double u0 = 2.3 + .4 * t, midpoint = u0 + .4 * h / 2.;
        const double fl = .7 * h * (left + (left + sl)) / 2.;
        const double fr = .7 * h * (right + (right + sr)) / 2.;
        const double source = ( .4 + .7) * h *
                              (interval.old_measure() + interval.new_measure()) / 2.;
        const double q0 = u0 * interval.old_measure();
        const double q1 = interval.updated_amount(q0, midpoint, midpoint, fl, fr, source);
        assert(std::abs(q1 / interval.new_measure() - (2.3 + .4 * (t + h))) < 1e-12);
        total_change += q1 - q0;
      }
      assert(std::abs(total_change - .4 * h) < 1e-12);
    }
    bool stale_refused = false, nan_refused = false;
    try { (void)SweptInterval::prepare(0., .5, 0., .52, 0., .01, 1e-13); }
    catch (const std::invalid_argument&) { stale_refused = true; }
    try { (void)SweptInterval::prepare(0., .5, 0., .52, 0., std::nan(""), 1e-13); }
    catch (const std::invalid_argument&) { nan_refused = true; }
    assert(stale_refused && nan_refused);
    std::cout << "manufactured_reynolds_balance_stale_nan_pass\n";
  }
  Kokkos::finalize();
}
