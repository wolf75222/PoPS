#pragma once

#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

namespace pops::runtime::amr {

template <int Dim>
POPS_EXPORT std::shared_ptr<const ExactAmrFieldSolverProvider<Dim>>
make_component_exact_amr_field_solver_provider(field::PreparedFieldSolverSpec spec,
                                               std::shared_ptr<component::LoadedComponent> topology,
                                               std::shared_ptr<component::LoadedComponent> solver);

}  // namespace pops::runtime::amr
