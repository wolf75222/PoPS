/// @file
/// @brief Compile-time-ranked elliptic right-hand-side assemblers.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/model/coupled_system.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pops {

namespace detail {

template <int Dim, class MemorySpace>
bool has_exact_field_identity(const MultiFab<Dim, MemorySpace>& source,
                              const MultiFab<Dim, MemorySpace>& destination) {
  return source.layout() == destination.layout() &&
         source.distribution() == destination.distribution() &&
         source.local_rank() == destination.local_rank() &&
         source.local_size() == destination.local_size();
}

template <int Dim, class MemorySpace>
void require_distinct_scalar_destination(const MultiFab<Dim, MemorySpace>& source,
                                         const MultiFab<Dim, MemorySpace>& destination,
                                         const char* operation) {
  if (!has_exact_field_identity(source, destination) || destination.ncomp() != 1 ||
      source.shares_storage_with(destination))
    throw std::invalid_argument(operation);
}

template <class Model, int Dim, class MemorySpace>
void require_model_rhs_target(const MultiFab<Dim, MemorySpace>& state,
                              const MultiFab<Dim, MemorySpace>& rhs, const char* operation) {
  require_distinct_scalar_destination(state, rhs, operation);
  if (Model::n_vars < 1 || state.ncomp() < Model::n_vars)
    throw std::invalid_argument(operation);
}

template <int Dim, class MemorySpace>
void require_component_rhs_target(const MultiFab<Dim, MemorySpace>& state, int component,
                                  const MultiFab<Dim, MemorySpace>& rhs, const char* operation) {
  require_distinct_scalar_destination(state, rhs, operation);
  if (component < 0 || component >= state.ncomp())
    throw std::invalid_argument(operation);
}

template <int Dim, class Model>
struct SingleModelEllipticRhsKernel {
  Model model;
  FieldView<const Real, Dim> state;
  FieldView<Real, Dim> rhs;

  POPS_HD void operator()(const Index<Dim>& index) const {
    rhs(index) = model.elliptic_rhs(load_state<Model>(state, index));
  }
};

template <int Dim, class Model>
struct AddModelEllipticRhsKernel {
  Model model;
  FieldView<const Real, Dim> state;
  FieldView<Real, Dim> rhs;

  POPS_HD void operator()(const Index<Dim>& index) const {
    rhs(index) += model.elliptic_rhs(load_state<Model>(state, index));
  }
};

template <int Dim>
struct TwoFieldChargeDensityRhsKernel {
  FieldView<Real, Dim> rhs;
  FieldView<const Real, Dim> first;
  FieldView<const Real, Dim> second;
  Real first_scale;
  Real second_scale;
  int first_component;
  int second_component;

  POPS_HD void operator()(const Index<Dim>& index) const {
    rhs(index) = first_scale * first(index, first_component) +
                 second_scale * second(index, second_component);
  }
};

template <int Dim>
struct AddScaledComponentKernel {
  FieldView<Real, Dim> rhs;
  FieldView<const Real, Dim> state;
  Real scale;
  int component;

  POPS_HD void operator()(const Index<Dim>& index) const {
    rhs(index) += scale * state(index, component);
  }
};

template <int Dim, class MemorySpace>
void add_scaled_component_unchecked(const MultiFab<Dim, MemorySpace>& state, Real scale,
                                    int component, MultiFab<Dim, MemorySpace>& rhs) {
  for (std::size_t local = 0; local < rhs.local_size(); ++local)
    for_each_cell(rhs.box(local),
                  AddScaledComponentKernel<Dim>{rhs.fab(local).view(), state.fab(local).view(),
                                                scale, component});
}

}  // namespace detail

/// Assemble one model's scalar elliptic source over one immutable field specialization.
template <int Dim, class Model,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct SingleModelEllipticRhs {
  Model model;

  void operator()(const MultiFab<Dim, MemorySpace>& state, MultiFab<Dim, MemorySpace>& rhs) const {
    detail::require_model_rhs_target<Model>(
        state, rhs,
        "SingleModelEllipticRhs requires distinct fields with exact layout/distribution identity, "
        "one RHS component, and the model's complete state width");
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local), detail::SingleModelEllipticRhsKernel<Dim, Model>{
                                          model, state.fab(local).view(), rhs.fab(local).view()});
  }
};

/// Accumulate one model contribution into an exactly co-distributed scalar RHS.
template <int Dim, class Model, class MemorySpace>
void add_model_elliptic_rhs(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                            MultiFab<Dim, MemorySpace>& rhs) {
  detail::require_model_rhs_target<Model>(
      state, rhs,
      "add_model_elliptic_rhs requires distinct fields with exact layout/distribution identity, "
      "one RHS component, and the model's complete state width");
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), detail::AddModelEllipticRhsKernel<Dim, Model>{
                                        model, state.fab(local).view(), rhs.fab(local).view()});
}

/// Assemble q0 * U0(comp0) + q1 * U1(comp1) over one exact ND field identity.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct TwoFieldChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  void operator()(const MultiFab<Dim, MemorySpace>& first, const MultiFab<Dim, MemorySpace>& second,
                  MultiFab<Dim, MemorySpace>& rhs) const {
    constexpr const char* operation =
        "TwoFieldChargeDensityRhs requires distinct fields with exact layout/distribution "
        "identity, valid source components, and one RHS component";
    detail::require_component_rhs_target(first, comp0, rhs, operation);
    detail::require_component_rhs_target(second, comp1, rhs, operation);
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      for_each_cell(rhs.box(local), detail::TwoFieldChargeDensityRhsKernel<Dim>{
                                        rhs.fab(local).view(), first.fab(local).view(),
                                        second.fab(local).view(), q0, q1, comp0, comp1});
  }
};

/// Two-block system adapter over the same ranked two-field assembler.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct TwoBlockChargeDensityRhs {
  Real q0 = Real(1);
  Real q1 = Real(-1);
  int comp0 = 0;
  int comp1 = 0;

  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab<Dim, MemorySpace>& rhs) const {
    static_assert(System::n_blocks >= 2,
                  "TwoBlockChargeDensityRhs requires at least two system blocks");
    TwoFieldChargeDensityRhs<Dim, MemorySpace>{q0, q1, comp0, comp1}(
        system.template block<0>().U(), system.template block<1>().U(), rhs);
  }
};

/// Charge (including sign) and density component of one species.
struct SpeciesCharge {
  Real charge = Real(0);
  int comp = 0;
};

/// Accumulate q * U(comp) after proving exact ND field identity.
template <int Dim, class MemorySpace>
void add_scaled_component(const MultiFab<Dim, MemorySpace>& state, Real scale, int component,
                          MultiFab<Dim, MemorySpace>& rhs) {
  detail::require_component_rhs_target(
      state, component, rhs,
      "add_scaled_component requires distinct fields with exact layout/distribution identity, "
      "a valid source component, and one RHS component");
  detail::add_scaled_component_unchecked(state, scale, component, rhs);
}

/// Assemble Sum_s q_s U_s(component_s) across every block of one ranked system.
/// All sources are preflighted before the destination is cleared, so a late mismatch cannot publish
/// a zeroed or partially accumulated RHS.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct ChargeDensityRhs {
  std::vector<SpeciesCharge> species;

  template <CoupledSystemLike System>
  void operator()(const System& system, MultiFab<Dim, MemorySpace>& rhs) const {
    if (species.size() != System::n_blocks)
      throw std::invalid_argument(
          "ChargeDensityRhs requires exactly one SpeciesCharge per block; neutral species use "
          "charge zero");

    constexpr const char* operation =
        "ChargeDensityRhs requires distinct fields with exact layout/distribution identity, valid "
        "source components, and one RHS component";
    if (rhs.ncomp() != 1)
      throw std::invalid_argument(operation);

    std::vector<const MultiFab<Dim, MemorySpace>*> sources;
    sources.reserve(species.size());
    system.for_each_block([&](const auto& block) {
      if (sources.size() >= species.size())
        throw std::invalid_argument(operation);
      const auto& state = block.U();
      detail::require_component_rhs_target(state, species[sources.size()].comp, rhs, operation);
      sources.push_back(&state);
    });
    if (sources.size() != species.size())
      throw std::invalid_argument(operation);

    rhs.set_val(Real(0));
    for (std::size_t index = 0; index < sources.size(); ++index)
      detail::add_scaled_component_unchecked(*sources[index], species[index].charge,
                                             species[index].comp, rhs);
  }
};

}  // namespace pops
