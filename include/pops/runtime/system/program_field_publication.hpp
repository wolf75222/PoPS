#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <string>
#include <vector>

namespace pops::runtime::system {

/// An accepted observation and its exact sealed producer destination. Source storage stays
/// borrowed until the enclosing all-level publication completes or rolls back.
template <int Dim>
struct ProgramFieldComponent {
  AuxiliaryComponentKey key;
  std::string expected_provider_identity;
  const MultiFab<Dim>* values = nullptr;
  int component = 0;
};

template <int Dim>
struct ProgramFieldLevel {
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::vector<ProgramFieldComponent<Dim>> components;
};

}  // namespace pops::runtime::system
