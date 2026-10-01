#pragma once

#include <pops/core/foundation/types.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace pops {
/// Explicit replicated host realization, never a default nonlinear preconditioner.
/// The caller assembles the full original-residual Jacobian on the active quotient.
class FullResidualBasisLU final {
 public:
  static constexpr const char* identity = "pops.amr.full-residual-basis-lu@1";
  static std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
      throw std::length_error("FullResidualBasisLU@1 byte count overflow");
    return a + b;
  }
  static std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
      throw std::length_error("FullResidualBasisLU@1 byte count overflow");
    return a * b;
  }
  static std::size_t required_bytes(std::size_t n, std::size_t external_bytes = 0) {
    // One in-place matrix, caller RHS/output, pivot indices and external map/towers.
    return checked_add(external_bytes, checked_add(checked_product(checked_product(n, n), sizeof(Real)),
        checked_product(n, 2 * sizeof(Real) + sizeof(std::size_t))));
  }
  void allocate(std::size_t n) {
    ready_ = false;
    n_ = n;
    values_.assign(checked_product(n, n), Real(0));
    pivots_.resize(n);
  }
  Real& entry(std::size_t row, std::size_t column) { return values_.at(row * n_ + column); }
  void begin_assembly() { ready_ = false; }
  void factor() {
    ready_ = false;
    for (Real value : values_)
      if (!std::isfinite(value)) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite original Jacobian");
    for (std::size_t k = 0; k < n_; ++k) {
      std::size_t pivot = k;
      for (std::size_t row = k + 1; row < n_; ++row)
        if (std::abs(entry(row, k)) > std::abs(entry(pivot, k))) pivot = row;
      if (entry(pivot, k) == Real(0))
        throw std::invalid_argument("FullResidualBasisLU@1 singular full original Jacobian on active quotient");
      pivots_[k] = pivot;
      if (pivot != k)
        for (std::size_t column = 0; column < n_; ++column) std::swap(entry(k, column), entry(pivot, column));
      for (std::size_t row = k + 1; row < n_; ++row) {
        entry(row, k) /= entry(k, k);
        if (!std::isfinite(entry(row, k))) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite pivot factor");
        for (std::size_t column = k + 1; column < n_; ++column) {
          entry(row, column) -= entry(row, k) * entry(k, column);
          if (!std::isfinite(entry(row, column))) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite factorization");
        }
      }
    }
    ready_ = true;
  }
  void apply(std::span<const Real> rhs, std::span<Real> result) const {
    if (!ready_ || rhs.size() != n_ || result.size() != n_)
      throw std::logic_error("FullResidualBasisLU@1 missing factor/active vector shape");
    std::copy(rhs.begin(), rhs.end(), result.begin());
    for (Real value : result) if (!std::isfinite(value)) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite RHS");
    for (std::size_t k = 0; k < n_; ++k) {
      if (pivots_[k] != k) std::swap(result[k], result[pivots_[k]]);
    }
    for (std::size_t row = 0; row < n_; ++row)
      for (std::size_t column = 0; column < row; ++column) result[row] -= values_[row * n_ + column] * result[column];
    for (std::size_t row = n_; row-- > 0;) {
      for (std::size_t column = row + 1; column < n_; ++column) result[row] -= values_[row * n_ + column] * result[column];
      result[row] /= values_[row * n_ + row];
      if (!std::isfinite(result[row])) throw std::invalid_argument("FullResidualBasisLU@1 nonfinite triangular solution");
    }
  }
 private:
  std::size_t n_ = 0;
  std::vector<Real> values_;
  std::vector<std::size_t> pivots_;
  bool ready_ = false;
};
}  // namespace pops
