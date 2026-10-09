// Actual native variable-coefficient operator instantiations, syntax-only.
#include <pops/numerics/elliptic/nd/general_field_operator.hpp>
#include <type_traits>
using namespace pops;
using namespace pops::elliptic::nd;
static_assert(std::is_same_v<decltype(&apply_general_field<1, 1, 1>),
                             decltype(&apply_general_field<1, 1, 1, true>)>);
static_assert(std::is_same_v<decltype(&apply_general_field<2, 3, 9>),
                             decltype(&apply_general_field<2, 3, 9, true>)>);
template void pops::elliptic::nd::apply_general_field<1, 1, 1, true>(
    MultiFab<1>&, MultiFab<1>&, const MultiFab<1>&,
    const runtime::program::PreparedScalarBoundarySession<1>&,
    const std::array<Real, 1>&, const std::array<PhysicalFieldBoundary, 2>&);
template void pops::elliptic::nd::apply_general_field<2, 3, 9, true>(
    MultiFab<2>&, MultiFab<2>&, const MultiFab<2>&,
    const runtime::program::PreparedScalarBoundarySession<2>&,
    const std::array<Real, 9>&, const std::array<PhysicalFieldBoundary, 4>&);
template void pops::elliptic::nd::apply_general_field<3, 2, 4, true>(
    MultiFab<3>&, MultiFab<3>&, const MultiFab<3>&,
    const runtime::program::PreparedScalarBoundarySession<3>&,
    const std::array<Real, 4>&, const std::array<PhysicalFieldBoundary, 6>&);
