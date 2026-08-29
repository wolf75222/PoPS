#pragma once

/// Zero every source-residual component that is not in the authored keep set.
///
/// This is the executable Program primitive behind a partial IMEX mask. Both the uniform
/// ``ProgramExecutionServices`` and the AMR program context call it after ``source_default_into``. The keep
/// set is resolved in Python against the block StateSpace; C++ only applies the integer mask.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <stdexcept>
#include <vector>

namespace pops::runtime::program {

template <int Dim>
void apply_component_keep_mask(MultiFab<Dim>& field, const std::vector<int>& keep) {
  const int ncomp = field.ncomp();
  if (ncomp < 1)
    throw std::invalid_argument("Program source mask requires a field with at least one component");
  std::vector<char> keep_flag(static_cast<std::size_t>(ncomp), 0);
  for (int component : keep) {
    if (component < 0 || component >= ncomp)
      throw std::invalid_argument("Program source mask component is out of range");
    keep_flag[static_cast<std::size_t>(component)] = 1;
  }
  for (int component = 0; component < ncomp; ++component) {
    if (keep_flag[static_cast<std::size_t>(component)] != 0)
      continue;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const FieldView<Real, Dim> view = field.fab(local).view();
      for_each_cell(field.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        view(cell, component) = Real(0);
      });
    }
  }
}

}  // namespace pops::runtime::program
