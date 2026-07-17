#pragma once

/// @file
/// @brief Side-effect-free field algebra used by prepared linear solves.
///
/// These operations deliberately bypass ProgramContext.  In particular, an AMR ProgramContext may
/// attach time-integration and reflux-ledger semantics to its public axpy/lincomb methods; Krylov
/// recurrences are private algebra on scratch fields and must never mutate that ledger.

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops {

namespace detail {

/// Device-clean fill over valid cells. Stencil applications overwrite their ghosts through the
/// authenticated boundary/halo plan before reading them, so prepared iterations do not need a host
/// fill of allocated ghost storage.
struct FillValidKernel {
  Array4 values;
  Real value;
  int component;
  POPS_HD void operator()(int i, int j) const { values(i, j, component) = value; }
};

inline void fill_valid(MultiFab& field, Real value) {
  field.sync_device();
  for (int local = 0; local < field.local_size(); ++local) {
    const Array4 values = field.fab(local).array();
    const Box2D valid = field.box(local);
    for (int component = 0; component < field.ncomp(); ++component)
      for_each_cell(valid, FillValidKernel{values, value, component});
  }
}

}  // namespace detail

struct PureFieldAlgebra {
  static bool same_vector_space(const MultiFab& left, const MultiFab& right) {
    return left.box_array().boxes() == right.box_array().boxes() &&
           left.dmap().ranks() == right.dmap().ranks() && left.ncomp() == right.ncomp();
  }

  static void require_same_vector_space(const MultiFab& left, const MultiFab& right,
                                        const char* where) {
    if (!same_vector_space(left, right))
      throw std::invalid_argument(std::string(where) +
                                  ": fields do not share box, distribution, and component space");
  }

  static void zero(MultiFab& value) { value.set_val(Real(0)); }

  /// Fill valid cells on the active Kokkos execution space. Ghosts are deliberately left for the
  /// next typed halo/boundary fill; this is the initialization primitive for prepared hot paths.
  static void fill_valid(MultiFab& value, Real fill) { detail::fill_valid(value, fill); }
  static void zero_valid(MultiFab& value) { fill_valid(value, Real(0)); }

  static void copy(MultiFab& destination, const MultiFab& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::copy");
    pops::lincomb(destination, Real(1), source, Real(0), source);
  }

  /// Copy valid cells and every allocated ghost cell without replacing either storage object. This
  /// is the allocation-free transaction primitive used when a prepared evaluation temporarily
  /// substitutes a live field and must restore its exact prior storage contents.
  static void copy_allocated(MultiFab& destination, const MultiFab& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::copy_allocated");
    if (destination.n_grow() != source.n_grow())
      throw std::invalid_argument(
          "PureFieldAlgebra::copy_allocated: fields have different ghost footprints");
    for (int local = 0; local < destination.local_size(); ++local) {
      Array4 out = destination.fab(local).array();
      const ConstArray4 in = source.fab(local).const_array();
      const Box2D allocated = destination.fab(local).grown_box();
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(allocated, detail::LincombKernel{out, in, in, Real(1), Real(0), component});
    }
  }

  static void axpy(MultiFab& destination, Real coefficient, const MultiFab& source) {
    require_same_vector_space(destination, source, "PureFieldAlgebra::axpy");
    pops::saxpy(destination, coefficient, source);
  }

  static void lincomb(MultiFab& destination, Real left_coefficient, const MultiFab& left,
                      Real right_coefficient, const MultiFab& right) {
    require_same_vector_space(destination, left, "PureFieldAlgebra::lincomb(left)");
    require_same_vector_space(destination, right, "PureFieldAlgebra::lincomb(right)");
    pops::lincomb(destination, left_coefficient, left, right_coefficient, right);
  }

  /// One collective full-vector dot product. Every rank calls the same reduction, including ranks
  /// with no local box.
  static Real dot(const MultiFab& left, const MultiFab& right) {
    require_same_vector_space(left, right, "PureFieldAlgebra::dot");
    return left.ncomp() == 1 ? pops::dot(left, right) : pops::dot_all(left, right);
  }

  static Real norm(const MultiFab& value) {
    const Real square = dot(value, value);
    if (!std::isfinite(static_cast<double>(square)) || square < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    return std::sqrt(square);
  }
};

namespace detail {

/// Unchecked algebra for an already authenticated prepared solve.  The public helpers above remain
/// defensive for transaction and extension call sites; the Krylov hot path validates its complete
/// vector space once when the problem/workspace are bound and must not rescan every box/rank vector
/// for each recurrence primitive.
struct PreparedFieldAlgebra {
  static void zero(MultiFab& value) { fill_valid(value, Real(0)); }

  static void copy(MultiFab& destination, const MultiFab& source) {
    pops::lincomb(destination, Real(1), source, Real(0), source);
  }

  static void axpy(MultiFab& destination, Real coefficient, const MultiFab& source) {
    pops::saxpy(destination, coefficient, source);
  }

  static void lincomb(MultiFab& destination, Real left_coefficient, const MultiFab& left,
                      Real right_coefficient, const MultiFab& right) {
    pops::lincomb(destination, left_coefficient, left, right_coefficient, right);
  }

  static Real dot(const MultiFab& left, const MultiFab& right) {
    return left.ncomp() == 1 ? pops::dot(left, right) : pops::dot_all(left, right);
  }

  static Real local_dot(const MultiFab& left, const MultiFab& right) {
    return left.ncomp() == 1 ? pops::dot_local(left, right) : pops::dot_all_local(left, right);
  }

  static Real norm(const MultiFab& value) {
    const Real square = dot(value, value);
    if (!std::isfinite(static_cast<double>(square)) || square < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    return std::sqrt(square);
  }
};

}  // namespace detail

}  // namespace pops
