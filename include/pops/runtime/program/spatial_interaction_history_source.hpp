#pragma once

#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/spatial_direct_interaction.hpp>

namespace pops::runtime::program {

/// Dedicated IR18 consumer authority. The general prior-field matcher is unchanged.
template <int Dim>
bool interaction_history_selected(const HistoryManager<Dim>& history, const std::string& name,
                                  int lag, int owner, std::string_view state,
                                  std::string_view space, std::string_view clock,
                                  std::string_view interpolation, ExactContractBuilder& exact) {
  if (lag < 1 || lag >= history.depth.at(name) ||
      history.owner.at(name) != owner || history.state_identity.at(name) != state ||
      history.space_identity.at(name) != space || history.clock_identity.at(name) != clock ||
      history.interpolation_identity.at(name) != interpolation)
    throw std::invalid_argument("interaction history differs from its exact State.n keeper seed");
  const auto& samples = history.validated_samples(name);
  if (static_cast<std::size_t>(history.depth.at(name)) != samples.size())
    throw std::invalid_argument("interaction history declaration differs from its physical ring");
  const int filled = history.fill_count.at(name);
  if (filled < 0 || filled > history.depth.at(name))
    throw std::invalid_argument("interaction history has invalid maturity");
  const bool cold = filled == 0;
  const bool initialized = history.initialized.at(name);
  const bool pending = history.store_pending.at(name);
  if ((cold && initialized != pending) || (!cold && !initialized))
    throw std::invalid_argument("interaction history has inconsistent cold/retained lifecycle");
  const auto& selected = samples.at(static_cast<std::size_t>(lag));
  validate_history_sample_provenance(selected, initialized, history.slot_dt.at(name).at(lag));
  if (!selected.authenticated() || (!cold && selected.kind != HistorySampleKind::Publication))
    throw std::invalid_argument("interaction history selected sample is unauthenticated");
  exact.text(name).scalar(lag).scalar(cold).scalar(initialized).scalar(pending).scalar(filled);
  history.append_descriptor_contract(name, exact);
  // Include the complete preexisting lifecycle, while requiring equality only
  // of the selected sample between levels. Pending slot zero may be newer.
  for (std::size_t slot = 0; slot < samples.size(); ++slot) {
    const Real dt = history.slot_dt.at(name).at(slot);
    validate_history_sample_provenance(samples[slot], initialized, dt);
    if (!samples[slot].authenticated() || !std::isfinite(dt) ||
        (samples[slot].kind == HistorySampleKind::RegisteredZeroStart && dt != Real(0)))
      throw std::invalid_argument("interaction history has invalid slot window authority");
    samples[slot].append_contract(exact);
    exact.scalar(dt);
  }
  return cold;
}

template <int Dim, class MemorySpace>
void interaction_history_cold_equal(const MultiFab<Dim, MemorySpace>& retained,
                                    const MultiFab<Dim, MemorySpace>& seed) {
  if (retained.layout() != seed.layout() || retained.distribution() != seed.distribution() ||
      retained.local_rank() != seed.local_rank() || retained.ncomp() != seed.ncomp())
    throw std::invalid_argument("interaction cold history has foreign seed topology");
  for (std::size_t global = 0; global < seed.layout().size(); ++global) {
    if (!seed.contains_local(global)) continue;
    const auto left = retained.fab_global(global).view();
    const auto right = seed.fab_global(global).view();
    for (int component = 0; component < seed.ncomp(); ++component) {
      const Real mismatch = for_each_cell_reduce_max(seed.layout()[global],
          [=] POPS_HD(const Index<Dim>& cell) {
            return std::bit_cast<InteractionRealWord>(left(cell, component)) ==
                   std::bit_cast<InteractionRealWord>(right(cell, component)) ? Real(0) : Real(1);
          });
      if (mismatch != Real(0))
        throw std::invalid_argument("interaction cold history is not the exact accepted State.n seed");
    }
  }
}

}  // namespace pops::runtime::program
