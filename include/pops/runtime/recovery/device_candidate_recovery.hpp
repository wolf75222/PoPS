#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>

namespace pops {

// The variable-recovery check of a state publication candidate, run where the candidate lives
// (spec DM8, G13 lot 4). It counts, over the valid cells of every local box (the active cells only
// when a mask is given), the cells whose conservative values are not finite and, among the other
// cells, those that the prepared recovery rejects. No candidate value reaches the host: the two
// counts are the only result.
struct CandidateRecoveryCensus {
  std::int64_t nonfinite_conservative = 0;
  std::int64_t unrecoverable = 0;

  POPS_HD std::int64_t failed() const { return nonfinite_conservative + unrecoverable; }
};

template <int Dim>
using CandidateRecoveryCheck = std::function<CandidateRecoveryCensus(
    const MultiFab<Dim>& candidate, const MultiFab<Dim>* active_cells)>;

namespace candidate_recovery_detail {

template <int Dim, int N, class Recovers>
struct CellCensus {
  using value_type = CandidateRecoveryCensus;

  Recovers recovers;
  FieldView<const Real, Dim> candidate{};
  FieldView<const Real, Dim> active{};
  bool has_active_mask = false;
  Index<Dim> lower{};
  Extent<Dim> extent{};

  POPS_HD void init(value_type& value) const { value = value_type{}; }
  POPS_HD void join(value_type& destination, const value_type& source) const {
    destination.nonfinite_conservative += source.nonfinite_conservative;
    destination.unrecoverable += source.unrecoverable;
  }
  POPS_HD void operator()(std::int64_t ordinal, value_type& value) const {
    const Index<Dim> index = detail::cell_index_from_ordinal(lower, extent, ordinal);
    if (has_active_mask && active(index) < Real(0.5))
      return;
    Real conservative[N];
    bool finite = true;
    for (int component = 0; component < N; ++component) {
      conservative[component] = candidate(index, component);
      finite = finite && Kokkos::isfinite(conservative[component]);
    }
    if (!finite)
      ++value.nonfinite_conservative;
    else if (!recovers(conservative))
      ++value.unrecoverable;
  }
};

// The prepared model recovery on one cell: the model's declared inversion, then the ordered
// admissible set at acceptance, which starts with the finiteness of every primitive component.
// This is the decision of PreparedModelVariableInversionRecovery::recover, whose acceptance
// schedule checks without projecting, evaluated by its device-clean parts.
template <class Model>
struct ModelRecovers {
  static constexpr int N = Model::n_vars;
  using Admissibility =
      decltype(prepare_model_recovery_admissibility(std::declval<const Model&>()));

  ModelVariableInversionSource<Model> source;
  Admissibility admissibility;

  POPS_HD bool operator()(const Real (&conservative)[N]) const {
    typename Model::State state{};
    for (int component = 0; component < N; ++component)
      state[component] = conservative[component];
    const auto inverted = source(state, ProviderValues<0>{}, InversionWorkspaceView{});
    if (!inverted.candidate_available)
      return false;
    return admissibility.evaluate(inverted.candidate).accepted;
  }
};

}  // namespace candidate_recovery_detail

// A check from a device-clean cell decision `recovers(const Real (&conservative)[N]) -> bool`,
// called only on cells whose N conservative values are finite.
template <int Dim, int N, class Recovers>
CandidateRecoveryCheck<Dim> make_device_candidate_recovery_check(Recovers recovers) {
  static_assert(N > 0, "candidate recovery requires a positive state width");
  return [recovers](const MultiFab<Dim>& candidate, const MultiFab<Dim>* active_cells) {
    if (candidate.ncomp() != N)
      throw std::invalid_argument("candidate recovery check differs from the state width");
    if (active_cells != nullptr &&
        (active_cells->ncomp() != 1 || active_cells->local_size() != candidate.local_size()))
      throw std::invalid_argument("candidate recovery active mask differs from the candidate");
    detail::ensure_kokkos_initialized();
    CandidateRecoveryCensus total;
    for (std::size_t local = 0; local < candidate.local_size(); ++local) {
      const Fab<Dim>& fab = candidate.fab(local);
      const Box<Dim>& box = fab.box();
      if (box.empty())
        continue;
      if (active_cells != nullptr && active_cells->fab(local).box() != box)
        throw std::invalid_argument("candidate recovery active mask differs from the local box");
      detail::require_iterable_box(box);
      candidate_recovery_detail::CellCensus<Dim, N, Recovers> census{
          recovers,
          std::as_const(fab).view(),
          active_cells == nullptr ? FieldView<const Real, Dim>{}
                                  : std::as_const(active_cells->fab(local)).view(),
          active_cells != nullptr,
          box.lo,
          box.extent()};
      CandidateRecoveryCensus local_census;
      Kokkos::parallel_reduce(
          "pops_candidate_recovery_census",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::int64_t>>(
              Kokkos::DefaultExecutionSpace{}, 0, box.numPts()),
          census, local_census);
      total.nonfinite_conservative += local_census.nonfinite_conservative;
      total.unrecoverable += local_census.unrecoverable;
    }
    return total;
  };
}

// The check of a generated model, equal to its prepared host recovery.
template <int Dim, class Model>
CandidateRecoveryCheck<Dim> make_model_candidate_recovery_check(const Model& model) {
  return make_device_candidate_recovery_check<Dim, Model::n_vars>(
      candidate_recovery_detail::ModelRecovers<Model>{ModelVariableInversionSource<Model>{model},
                                                      prepare_model_recovery_admissibility(model)});
}

}  // namespace pops
