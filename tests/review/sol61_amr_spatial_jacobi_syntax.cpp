// Source-only instantiation probe. No executable, PDE result or native qualification.
#include <pops/runtime/program/prepared_amr_field_residual.hpp>

template class pops::runtime::program::PreparedAmrFieldResidual<2>;
template class pops::runtime::program::PreparedAmrFieldResidual<3>;

template <int Dim>
void instantiate_original_right_jacobi_solve(
    pops::runtime::program::PreparedAmrFieldResidual<Dim>& prepared,
    const pops::runtime::program::AmrFieldResidualAuthority& authority,
    const pops::ExecutionLane& lane) {
  (void)prepared.solve(authority, nullptr,
      [](const auto&, const auto&, auto&, int) {}, lane);
}

template void instantiate_original_right_jacobi_solve<2>(
    pops::runtime::program::PreparedAmrFieldResidual<2>&,
    const pops::runtime::program::AmrFieldResidualAuthority&, const pops::ExecutionLane&);
template void instantiate_original_right_jacobi_solve<3>(
    pops::runtime::program::PreparedAmrFieldResidual<3>&,
    const pops::runtime::program::AmrFieldResidualAuthority&, const pops::ExecutionLane&);
