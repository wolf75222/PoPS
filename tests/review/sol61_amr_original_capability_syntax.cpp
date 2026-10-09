// Independent syntax reception only: real provider/carrier/storage/Kokkos interfaces.
#include <pops/numerics/elliptic/nd/prepared_composite_general_field.hpp>
#include <pops/runtime/program/prepared_amr_field_residual.hpp>

template class pops::elliptic::nd::PreparedCompositeGeneralField<2>;
template class pops::runtime::program::PreparedAmrFieldResidual<2>;

void instantiate_original_body(
    pops::runtime::program::PreparedAmrFieldResidual<2>& carrier,
    const pops::runtime::program::AmrFieldResidualAuthority& authority,
    const pops::ExecutionLane& lane) {
  auto original = [](const auto& values, const auto& captures, auto& residual, int) {
    for (std::size_t level = 0; level < residual.size(); ++level)
      for (std::size_t patch = 0; patch < residual[level].local_size(); ++patch) {
        const auto q = values[level].fab(patch).view();
        const auto parameter = captures.at(0)[level].fab(patch).view();
        const auto output = residual[level].fab(patch).view();
        const int width = residual[level].ncomp();
        pops::for_each_cell(residual[level].box(patch), [=] POPS_HD(const pops::Index<2>& cell) {
          for (int i = 0; i < width; ++i) {
            const auto u = q(cell, i);
            output(cell, i) += (pops::Real(.31) + parameter(cell, 0)) * u +
                              pops::Real(.2) * u * u * u;
          }
        });
      }
  };
  (void)carrier.solve(authority, nullptr, original, lane);
}
