#pragma once
// Shared surface for the split pybind11 bindings of `_pops` (ADC-365). bindings.cpp is the thin
// PYBIND11_MODULE that calls init_core / init_system / init_amr; each lives in its own TU so the
// py::class_/.def template instantiations compile in parallel (better incremental, lower peak pybind
// memory per TU). This header carries the common includes, the small array/POD helpers (moved verbatim
// from the old monolithic bindings.cpp), and the init_* declarations.

#include <pybind11/functional.h>  // std::function<double()> <- Python callable (add_dt_bound)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/core/foundation/kokkos_env.hpp>  // Kokkos_Core under POPS_HAS_KOKKOS (kokkos_is_initialized)
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/mesh/boundary/periodicity.hpp>
#include <pops/parallel/comm.hpp>  // pops::my_rank / n_ranks: rank-0 guard of the multi-rank IO facade
#include <pops/runtime/dynamic/abi_key.hpp>  // pops::abi_key: ABI key exposed to the DSL ("production" path)
#include <pops/runtime/config/runtime_params.hpp>          // kMaxRuntimeParams (ADC-618 hard_limit)
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>  // DFT-fallback counter (ADC-618 diagnostic)
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system.hpp>

#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>  // std::tuple: argument of AmrSystem.set_hierarchy (patch_boxes boxes) (ADC-65)
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pops;

template <int Dim>
py::tuple ranked_periodicity_to_python(const std::array<bool, Dim>& value) {
  py::tuple result(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value[static_cast<std::size_t>(axis)];
  return result;
}

template <int Dim>
std::array<bool, Dim> ranked_periodicity_from_python(const py::handle& value, const char* owner) {
  if (!PyTuple_CheckExact(value.ptr()) || py::len(value) != Dim)
    throw py::type_error(std::string(owner) +
                         ".periodicity must be an exact tuple matching the native dimension");
  const py::tuple tuple = py::reinterpret_borrow<py::tuple>(value);
  std::array<bool, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (!PyBool_Check(tuple[axis].ptr()))
      throw py::type_error(std::string(owner) + ".periodicity entries must be exact bool values");
    result[static_cast<std::size_t>(axis)] = tuple[axis].ptr() == Py_True;
  }
  return result;
}

template <int Dim>
py::tuple ranked_extent_to_python(const Extent<Dim>& value) {
  py::tuple result(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value[axis];
  return result;
}

template <int Dim>
Extent<Dim> ranked_extent_from_python(const py::handle& value, const char* owner,
                                      bool allow_zero = false) {
  if (!PyTuple_CheckExact(value.ptr()) || py::len(value) != Dim)
    throw py::type_error(std::string(owner) +
                         " must be an exact tuple matching the native dimension");
  const py::tuple tuple = py::reinterpret_borrow<py::tuple>(value);
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (!PyLong_CheckExact(tuple[axis].ptr()) || PyBool_Check(tuple[axis].ptr()))
      throw py::type_error(std::string(owner) + " entries must be exact integers");
    const std::int64_t component = py::cast<std::int64_t>(tuple[axis]);
    if (component < (allow_zero ? 0 : 1))
      throw py::value_error(std::string(owner) + (allow_zero
                                                      ? " entries must be non-negative"
                                                      : " entries must be strictly positive"));
    result[axis] = component;
  }
  return result;
}

template <int Dim>
py::tuple ranked_extents_to_python(const std::vector<Extent<Dim>>& values) {
  py::tuple result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    result[index] = ranked_extent_to_python(values[index]);
  return result;
}

template <int Dim>
std::vector<Extent<Dim>> ranked_extents_from_python(const py::handle& value, const char* owner,
                                                    std::int64_t minimum) {
  if (!PyTuple_CheckExact(value.ptr()) && !PyList_CheckExact(value.ptr()))
    throw py::type_error(std::string(owner) + " must be an ordered sequence of ranked tuples");
  const py::sequence rows = py::reinterpret_borrow<py::sequence>(value);
  std::vector<Extent<Dim>> result;
  result.reserve(rows.size());
  for (std::size_t row_index = 0; row_index < static_cast<std::size_t>(rows.size()); ++row_index) {
    const py::handle row = rows[static_cast<py::ssize_t>(row_index)];
    if (!PyTuple_CheckExact(row.ptr()) || py::len(row) != Dim)
      throw py::type_error(std::string(owner) + " rows must be exact native-rank tuples");
    const py::tuple components = py::reinterpret_borrow<py::tuple>(row);
    Extent<Dim> extent{};
    for (int axis = 0; axis < Dim; ++axis) {
      if (!PyLong_CheckExact(components[axis].ptr()) || PyBool_Check(components[axis].ptr()))
        throw py::type_error(std::string(owner) + " entries must be exact integers");
      const std::int64_t component = py::cast<std::int64_t>(components[axis]);
      if (component < minimum)
        throw py::value_error(std::string(owner) +
                              " entries must be >= " + std::to_string(minimum));
      extent[axis] = component;
    }
    result.push_back(extent);
  }
  return result;
}

template <int Dim>
py::tuple ranked_real_vector_to_python(const RealVector<Dim>& value) {
  py::tuple result(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value[axis];
  return result;
}

template <int Dim>
RealVector<Dim> ranked_real_vector_from_python(const py::handle& value, const char* owner) {
  if (!PyTuple_CheckExact(value.ptr()) || py::len(value) != Dim)
    throw py::type_error(std::string(owner) +
                         " must be an exact tuple matching the native dimension");
  const py::tuple tuple = py::reinterpret_borrow<py::tuple>(value);
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (PyBool_Check(tuple[axis].ptr()) ||
        (!PyFloat_CheckExact(tuple[axis].ptr()) && !PyLong_CheckExact(tuple[axis].ptr())))
      throw py::type_error(std::string(owner) + " entries must be exact real scalars");
    result[axis] = py::cast<double>(tuple[axis]);
    if (!std::isfinite(result[axis]))
      throw py::value_error(std::string(owner) + " entries must be finite");
  }
  return result;
}

/// Validate the retired mapped-periodicity compatibility payload at the native boundary. Ordinary
/// axis translations are carried exclusively by the 2*Dim face table. A non-empty table therefore
/// proves that the request needs a separately capability-qualified mapped-topology provider.
template <int Dim>
void reject_unqualified_periodic_identifications(const std::vector<std::vector<int>>& rows,
                                                 const char* owner) {
  for (const auto& row : rows) {
    if (row.size() != static_cast<std::size_t>(2 + 2 * Dim))
      throw py::value_error(std::string(owner) +
                            " periodic identification row must contain 2+2*Dim integers");
    PeriodicIdentification<Dim> identification;
    identification.source_face = row[0];
    identification.target_face = row[1];
    for (int axis = 0; axis < Dim; ++axis) {
      identification.permutation[static_cast<std::size_t>(axis)] =
          row[static_cast<std::size_t>(2 + axis)];
      identification.signs[static_cast<std::size_t>(axis)] =
          row[static_cast<std::size_t>(2 + Dim + axis)];
    }
    identification.validate();
  }
  if (!rows.empty())
    throw py::value_error(std::string(owner) +
                          " requires a capability-qualified mapped-topology provider");
}

template <int Dim>
py::tuple ranked_boxes_to_python(const std::vector<Box<Dim>>& boxes) {
  py::tuple result(boxes.size());
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    py::tuple lower(Dim), upper(Dim);
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = boxes[index].lo[axis];
      upper[axis] = boxes[index].hi[axis] + 1;
    }
    result[index] = py::make_tuple(std::move(lower), std::move(upper));
  }
  return result;
}

template <int Dim>
std::vector<Box<Dim>> ranked_boxes_from_python(const py::handle& value, const char* owner) {
  if (!PyTuple_CheckExact(value.ptr()))
    throw py::type_error(std::string(owner) + " must be an exact tuple of ranked boxes");
  const py::tuple rows = py::reinterpret_borrow<py::tuple>(value);
  std::vector<Box<Dim>> result;
  result.reserve(rows.size());
  for (const py::handle row_handle : rows) {
    if (!PyTuple_CheckExact(row_handle.ptr()) || py::len(row_handle) != 2)
      throw py::type_error(std::string(owner) + " rows must contain lower and upper tuples");
    const py::tuple row = py::reinterpret_borrow<py::tuple>(row_handle);
    if (!PyTuple_CheckExact(row[0].ptr()) || !PyTuple_CheckExact(row[1].ptr()) ||
        py::len(row[0]) != Dim || py::len(row[1]) != Dim)
      throw py::type_error(std::string(owner) + " bounds must match the native dimension");
    const py::tuple lower = py::reinterpret_borrow<py::tuple>(row[0]);
    const py::tuple upper = py::reinterpret_borrow<py::tuple>(row[1]);
    Box<Dim> box;
    for (int axis = 0; axis < Dim; ++axis) {
      if (!PyLong_CheckExact(lower[axis].ptr()) || PyBool_Check(lower[axis].ptr()) ||
          !PyLong_CheckExact(upper[axis].ptr()) || PyBool_Check(upper[axis].ptr()))
        throw py::type_error(std::string(owner) + " bounds must contain exact integers");
      const int low = py::cast<int>(lower[axis]);
      const std::int64_t high_exclusive = py::cast<std::int64_t>(upper[axis]);
      if (high_exclusive <= low || high_exclusive - 1 > std::numeric_limits<int>::max())
        throw py::value_error(std::string(owner) + " contains an empty or overflowing box");
      box.lo[axis] = low;
      box.hi[axis] = static_cast<int>(high_exclusive - 1);
    }
    result.push_back(box);
  }
  return result;
}

template <int Dim>
py::tuple ranked_amr_patches_to_python(const std::vector<AmrPatch<Dim>>& patches) {
  py::tuple result(patches.size());
  for (std::size_t index = 0; index < patches.size(); ++index) {
    py::tuple lower(Dim), upper(Dim);
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = patches[index].box.lo[axis];
      upper[axis] = patches[index].box.hi[axis] + 1;
    }
    result[index] = py::make_tuple(patches[index].level, std::move(lower), std::move(upper));
  }
  return result;
}

template <int Dim>
std::vector<AmrPatch<Dim>> ranked_amr_patches_from_python(const py::handle& value,
                                                          const char* owner) {
  if (!PyTuple_CheckExact(value.ptr()))
    throw py::type_error(std::string(owner) + " must be an exact tuple of AMR patches");
  const py::tuple rows = py::reinterpret_borrow<py::tuple>(value);
  std::vector<AmrPatch<Dim>> result;
  result.reserve(rows.size());
  for (const py::handle row_handle : rows) {
    if (!PyTuple_CheckExact(row_handle.ptr()) || py::len(row_handle) != 3)
      throw py::type_error(std::string(owner) + " rows must contain level, lower and upper");
    const py::tuple row = py::reinterpret_borrow<py::tuple>(row_handle);
    if (!PyLong_CheckExact(row[0].ptr()) || PyBool_Check(row[0].ptr()))
      throw py::type_error(std::string(owner) + " levels must be exact integers");
    const int level = py::cast<int>(row[0]);
    if (level < 0)
      throw py::value_error(std::string(owner) + " levels must be non-negative");
    const py::tuple bounds = py::make_tuple(py::make_tuple(row[1], row[2]));
    std::vector<Box<Dim>> boxes = ranked_boxes_from_python<Dim>(bounds, owner);
    result.push_back(AmrPatch<Dim>{level, boxes.front()});
  }
  return result;
}

template <int Dim>
std::vector<py::ssize_t> ranked_numpy_shape(const Extent<Dim>& native_shape) {
  std::vector<py::ssize_t> result(static_cast<std::size_t>(Dim));
  for (int numpy_axis = 0; numpy_axis < Dim; ++numpy_axis) {
    const std::int64_t cells = native_shape[Dim - 1 - numpy_axis];
    if (cells < 1 || cells > std::numeric_limits<py::ssize_t>::max())
      throw std::overflow_error(
          "pops (bindings): native spatial extent is outside the NumPy shape range");
    result[static_cast<std::size_t>(numpy_axis)] = static_cast<py::ssize_t>(cells);
  }
  return result;
}

/// Copy one exact ranked, axis-zero-contiguous native field to NumPy. NumPy presents the native
/// axes in reverse order so its final axis remains contiguous; the rank itself comes from the
/// compiled System specialization and is never inferred from the incoming value buffer.
template <int Dim>
py::array_t<double> to_ranked_field(const std::vector<double>& values,
                                    const Extent<Dim>& native_shape) {
  py::array_t<double> result(ranked_numpy_shape(native_shape));
  if (static_cast<std::size_t>(result.size()) != values.size())
    throw std::runtime_error(
        "pops (bindings): field element count does not match the exact native spatial shape");
  std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(double));
  return result;
}

/// Component-major counterpart of to_ranked_field. The leading component axis is retained and the
/// spatial axes alone are reversed for NumPy, yielding (ncomp, nz, ny, nx) for a 3D build.
template <int Dim>
py::array_t<double> to_ranked_state(const std::vector<double>& values, int ncomp,
                                    const Extent<Dim>& native_shape) {
  if (ncomp < 1)
    throw std::invalid_argument("pops (bindings): state component count must be positive");
  std::vector<py::ssize_t> shape = ranked_numpy_shape(native_shape);
  shape.insert(shape.begin(), static_cast<py::ssize_t>(ncomp));
  py::array_t<double> result(shape);
  if (static_cast<std::size_t>(result.size()) != values.size())
    throw std::runtime_error(
        "pops (bindings): state element count does not match components times native shape");
  std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(double));
  return result;
}
template <int Dim>
inline py::tuple output_pieces_to_python(const std::vector<OutputPiece<Dim>>& pieces) {
  py::tuple result(pieces.size());
  for (std::size_t index = 0; index < pieces.size(); ++index) {
    const OutputPiece<Dim>& piece = pieces[index];
    const std::int64_t cells = piece.box.numPts();
    if (piece.level < 0 || piece.ncomp < 1 || cells < 1 ||
        static_cast<std::uint64_t>(cells) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(piece.ncomp) ||
        piece.values.size() !=
            static_cast<std::size_t>(piece.ncomp) * static_cast<std::size_t>(cells))
      throw std::runtime_error("native output piece has an inconsistent compact shape");
    std::vector<py::ssize_t> value_shape(static_cast<std::size_t>(Dim + 1));
    value_shape[0] = static_cast<py::ssize_t>(piece.ncomp);
    py::tuple lower(static_cast<py::ssize_t>(Dim));
    py::tuple upper(static_cast<py::ssize_t>(Dim));
    for (int array_axis = 0; array_axis < Dim; ++array_axis) {
      const int native_axis = Dim - 1 - array_axis;
      value_shape[static_cast<std::size_t>(array_axis + 1)] =
          static_cast<py::ssize_t>(piece.box.length(native_axis));
      lower[static_cast<py::ssize_t>(array_axis)] = piece.box.lo[native_axis];
      upper[static_cast<py::ssize_t>(array_axis)] = piece.box.hi[native_axis] + 1;
    }
    py::array_t<double> values(value_shape);
    std::memcpy(values.mutable_data(), piece.values.data(), piece.values.size() * sizeof(double));
    py::dict row;
    row["lower"] = std::move(lower);
    row["upper"] = std::move(upper);
    row["values"] = std::move(values);
    row["global_box_index"] = piece.global_box_index;
    row["owner_rank"] = piece.owner_rank;
    row["replicated"] = piece.replicated;
    result[index] = std::move(row);
  }
  return result;
}
inline std::vector<double> flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

inline py::dict profile_snapshot_to_dict(
    const pops::runtime::program::Profiler::Snapshot& snapshot) {
  py::list scopes;
  for (const auto& scope : snapshot.scopes) {
    py::dict row;
    row["name"] = scope.name;
    row["count"] = scope.count;
    row["total_s"] = scope.total_s;
    row["mean_s"] = scope.mean_s;
    row["min_s"] = scope.min_s;
    row["max_s"] = scope.max_s;
    scopes.append(row);
  }
  py::list counters;
  for (const auto& counter : snapshot.counters) {
    py::dict row;
    row["name"] = counter.name;
    row["value"] = counter.value;
    counters.append(row);
  }
  py::dict out;
  out["schema_version"] = snapshot.schema_version;
  out["enabled"] = snapshot.enabled;
  out["total_s"] = snapshot.total_s;
  out["scopes"] = scopes;
  out["counters"] = counters;
  return out;
}

inline py::dict numerical_defaults_report_to_dict() {
  py::dict newton;
  newton["max_iters"] = kNewtonDefaultMaxIters;
  newton["rel_tol"] = static_cast<double>(kNewtonDefaultRelTol);
  newton["abs_tol"] = static_cast<double>(kNewtonDefaultAbsTol);
  newton["fd_eps"] = static_cast<double>(kNewtonDefaultFdEps);
  newton["damping"] = static_cast<double>(kNewtonDefaultDamping);
  newton["finite_abs_limit"] = static_cast<double>(kNewtonFiniteAbsLimit);

  py::dict krylov;
  krylov["rel_tol"] = static_cast<double>(kKrylovDefaultRelTol);
  krylov["polar_tensor_max_iters"] = kPolarTensorKrylovDefaultMaxIters;
  krylov["schur_polar_max_iters"] = kSchurKrylovPolarMaxIters;
  krylov["breakdown_tiny"] = static_cast<double>(kKrylovBreakdownTiny);

  py::dict mg;
  mg["rel_tol"] = static_cast<double>(kMGDefaultRelTol);
  mg["max_cycles"] = kMGDefaultMaxCycles;
  mg["abs_tol"] = static_cast<double>(kMGDefaultAbsTol);
  mg["min_coarse"] = kMGDefaultMinCoarse;
  mg["pre_smooth"] = kMGDefaultPreSmooth;
  mg["post_smooth"] = kMGDefaultPostSmooth;
  mg["bottom_sweeps"] = kMGDefaultBottomSweeps;
  mg["coarse_threshold"] = kMGDefaultCoarseThreshold;  // ADC-644: total-cell coarsening ceiling.

  py::dict fac;
  fac["max_iters"] = kFACDefaultMaxIters;
  fac["fine_sweeps"] = kFACDefaultFineSweeps;
  fac["rel_tol"] = static_cast<double>(kFACDefaultRelTol);
  fac["abs_tol"] = static_cast<double>(kFACDefaultAbsTol);
  fac["coarse_rel_tol"] = static_cast<double>(kFACInitialCoarseRelTol);
  fac["coarse_abs_tol"] = static_cast<double>(kFACInitialCoarseAbsTol);
  fac["coarse_cycles"] = kFACInitialCoarseMaxCycles;

  py::dict fft;
  fft["spectral_default"] = kFFTDefaultSpectral;
  fft["zero_mean_gauge"] = kFFTZeroMeanGauge;
  fft["direct_dft_fallback"] = kFFTDirectDftFallback;

  py::dict eb;
  eb["cut_fraction_floor"] = static_cast<double>(kEbCutFractionFloor);
  eb["face_open_eps"] = static_cast<double>(kEbFaceOpenEps);  // ADC-615/618
  eb["kappa_min"] = static_cast<double>(kEbKappaMin);

  py::dict weno;
  weno["epsilon"] = static_cast<double>(kWenoEpsilon);

  py::dict performance;
  performance["cfl_speed_floor"] = static_cast<double>(kCflSpeedFloor);
  performance["adaptive_no_evolving_block_sentinel"] =
      static_cast<double>(kAdaptiveNoEvolvingBlockSentinel);

  py::dict amr;
  amr["max_levels"] = kAmrDefaultMaxLevels;
  amr["refinement_ratio"] = kAmrRefRatio;
  amr["refinement_disabled_threshold"] = static_cast<double>(kAmrRefinementDisabledThreshold);
  amr["phi_refinement_disabled_threshold"] =
      static_cast<double>(kAmrPhiRefinementDisabledThreshold);

  py::dict physical;
  physical["preset"] = "legacy_native_brick_defaults";
  physical["B0"] = static_cast<double>(kPhysicalDefaultB0);
  physical["gamma"] = static_cast<double>(kPhysicalDefaultGamma);
  physical["fluid_state_cs2"] = static_cast<double>(kPhysicalDefaultFluidStateCs2);
  physical["native_brick_isothermal_cs2"] =
      static_cast<double>(kPhysicalDefaultNativeIsothermalCs2);
  physical["vacuum_floor"] = static_cast<double>(kPhysicalDefaultVacuumFloor);
  physical["qom"] = static_cast<double>(kPhysicalDefaultQOverM);
  physical["charge_q"] = static_cast<double>(kPhysicalDefaultChargeQ);
  physical["alpha"] = static_cast<double>(kPhysicalDefaultAlpha);
  physical["n0"] = static_cast<double>(kPhysicalDefaultBackgroundN0);
  physical["gravity_sign"] = static_cast<double>(kPhysicalDefaultGravitySign);
  physical["four_pi_G"] = static_cast<double>(kPhysicalDefaultFourPiG);
  physical["gravity_rho0"] = static_cast<double>(kPhysicalDefaultGravityRho0);
  physical["cs2_note"] =
      "FluidState defaults to 0.5 while the raw native IsothermalFlux brick defaults to 1.0.";

  // ADC-618: hard limits + diagnostics. kMaxRuntimeParams is a fixed-size device carrier bound
  // (native_loader fails fast above it); the DFT-fallback counter records each time the FFT Poisson
  // falls back to the O(n^2) direct DFT on a non-power-of-two grid.
  py::dict runtime;
  runtime["max_runtime_params"] = kMaxRuntimeParams;

  py::dict diagnostics;
  diagnostics["fft_direct_dft_fallback_count"] =
      static_cast<int>(poisson_fft_direct_dft_fallback_count());

  // ADC-618: the CLASSIFICATION fence. EVERY user-visible inline constexpr numeric constant of
  // numerical_defaults.hpp / types.hpp / runtime_params.hpp appears here with an explicit class:
  //   public_knob     -- configurable end to end (a typed descriptor / setter reaches the native use);
  //   internal_default -- a fixed default not (yet) user-configurable, but inspectable;
  //   diagnostic_only  -- a counter / instrumented fact, not a tuning knob;
  //   hard_limit       -- a fixed cap enforced fail-fast (changing it needs a header rebuild).
  // The source-scanning architecture test (tests/python/architecture/test_numeric_constant_fence.py)
  // asserts no constant is missing from this map -> a new user-visible constant cannot ship unclassified.
  py::dict classification;
  auto klass = [&classification](const char* name, const char* cls) { classification[name] = cls; };
  klass("kNewtonDefaultMaxIters", "public_knob");
  klass("kNewtonDefaultRelTol", "public_knob");
  klass("kNewtonDefaultAbsTol", "public_knob");
  klass("kNewtonDefaultFdEps", "public_knob");
  klass("kNewtonDefaultDamping", "public_knob");
  klass("kNewtonFiniteAbsLimit", "internal_default");
  klass("kKrylovDefaultRelTol", "public_knob");
  klass("kPolarTensorKrylovDefaultMaxIters", "public_knob");
  klass("kSchurKrylovPolarMaxIters", "public_knob");
  klass("kKrylovBreakdownTiny", "internal_default");
  klass("kMGDefaultRelTol", "public_knob");
  klass("kMGDefaultMaxCycles", "public_knob");
  klass("kMGDefaultAbsTol", "public_knob");
  klass("kMGDefaultMinCoarse", "public_knob");
  klass("kMGDefaultPreSmooth", "public_knob");
  klass("kMGDefaultPostSmooth", "public_knob");
  klass("kMGDefaultBottomSweeps", "public_knob");
  klass("kMGDefaultCoarseThreshold", "public_knob");
  klass("kFACDefaultMaxIters", "public_knob");
  klass("kFACDefaultFineSweeps", "public_knob");
  klass("kFACDefaultRelTol", "public_knob");
  klass("kFACDefaultAbsTol", "public_knob");
  klass("kFACInitialCoarseRelTol", "public_knob");
  klass("kFACInitialCoarseAbsTol", "public_knob");
  klass("kFACInitialCoarseMaxCycles", "public_knob");
  klass("kFFTDefaultSpectral", "public_knob");
  klass("kFFTZeroMeanGauge", "internal_default");
  klass("kFFTDirectDftFallback", "diagnostic_only");
  klass("kEbCutFractionFloor", "public_knob");
  // The knob is public; each resolved layout still proves that its reconstruction and transfer
  // providers can execute the requested stencil before the value reaches a native kernel.
  klass("kWenoEpsilon", "public_knob");
  klass("kEbFaceOpenEps", "public_knob");
  klass("kEbKappaMin", "public_knob");
  klass("kAmrDefaultMaxLevels", "internal_default");
  klass("kAmrRefinementDisabledThreshold", "internal_default");
  klass("kAmrPhiRefinementDisabledThreshold", "internal_default");
  klass("kAdaptiveNoEvolvingBlockSentinel", "diagnostic_only");
  klass("kAmrClusterMinEfficiency", "public_knob");
  klass("kAmrClusterMinBoxSize", "public_knob");
  klass("kAmrClusterMaxBoxSize", "public_knob");
  klass("kAmrDriftSpeedFloor", "internal_default");
  klass("kPhysicalDefaultB0", "public_knob");
  klass("kPhysicalDefaultGamma", "public_knob");
  klass("kPhysicalDefaultFluidStateCs2", "public_knob");
  klass("kPhysicalDefaultNativeIsothermalCs2", "internal_default");
  klass("kPhysicalDefaultVacuumFloor", "public_knob");
  klass("kPhysicalDefaultQOverM", "public_knob");
  klass("kPhysicalDefaultChargeQ", "public_knob");
  klass("kPhysicalDefaultAlpha", "public_knob");
  klass("kPhysicalDefaultBackgroundN0", "public_knob");
  klass("kPhysicalDefaultGravitySign", "public_knob");
  klass("kPhysicalDefaultFourPiG", "public_knob");
  klass("kPhysicalDefaultGravityRho0", "public_knob");
  klass("kCflSpeedFloor", "public_knob");  // ADC-645: step_cfl(speed_floor=) is wired end to end
  klass("kMaxRuntimeParams", "hard_limit");

  py::dict out;
  out["schema_version"] = 1;
  out["source"] = "pops.runtime.numerical_defaults";
  out["newton"] = newton;
  out["krylov"] = krylov;
  out["mg"] = mg;
  out["fac"] = fac;
  out["fft"] = fft;
  out["eb"] = eb;
  out["weno"] = weno;
  out["performance"] = performance;
  out["amr"] = amr;
  out["physical"] = physical;
  out["runtime"] = runtime;
  out["diagnostics"] = diagnostics;
  out["classification"] = classification;
  return out;
}

inline py::dict fallback_diagnostics_report_to_dict(const FallbackDiagnosticsReport& report) {
  py::list entries;
  std::size_t total_count = 0;
  for (const FallbackDiagnosticEntry& entry : report.entries) {
    py::dict row;
    row["key"] = entry.key;
    row["route"] = entry.route;
    row["cause"] = entry.cause;
    row["policy"] = entry.policy;
    row["default_action"] = entry.default_action;
    row["impact"] = entry.impact;
    row["frequency"] = entry.frequency;
    row["count"] = entry.count;
    row["explicit_opt_in"] = entry.explicit_opt_in;
    row["performance_degraded"] = entry.performance_degraded;
    row["semantics_changed"] = entry.semantics_changed;
    total_count += entry.count;
    entries.append(row);
  }
  py::dict out;
  out["schema_version"] = report.schema_version;
  out["source"] = report.source;
  out["entries"] = entries;
  out["total_count"] = total_count;
  return out;
}

inline py::dict effective_newton_options_to_dict(const EffectiveNewtonOptions& n) {
  py::dict d;
  d["max_iters"] = n.max_iters;
  d["rel_tol"] = n.rel_tol;
  d["abs_tol"] = n.abs_tol;
  d["fd_eps"] = n.fd_eps;
  d["damping"] = n.damping;
  d["diagnostics"] = n.diagnostics;
  d["non_default"] = n.non_default;
  return d;
}

inline py::dict effective_block_options_to_dict(const EffectiveBlockOptions& b) {
  py::dict physical;
  physical["gamma"] = b.gamma;
  physical["B0"] = b.B0;
  physical["cs2"] = b.cs2;
  physical["vacuum_floor"] = b.vacuum_floor;
  physical["qom"] = b.qom;
  physical["q"] = b.q;
  physical["alpha"] = b.alpha;
  physical["n0"] = b.n0;
  physical["sign"] = b.sign;
  physical["four_pi_G"] = b.four_pi_G;
  physical["rho0"] = b.rho0;

  py::dict d;
  d["name"] = b.name;
  d["route"] = b.route;
  d["compiled"] = b.compiled;
  d["transport"] = b.transport;
  d["source"] = b.source;
  d["elliptic"] = b.elliptic;
  d["limiter"] = b.limiter;
  d["riemann"] = b.riemann;
  d["recon"] = b.recon;
  d["time"] = b.time;
  d["time_method"] = b.time_method;
  d["substeps"] = b.substeps;
  d["stride"] = b.stride;
  d["evolve"] = b.evolve;
  d["ncomp"] = b.ncomp;
  d["n_ghost"] = b.n_ghost;
  d["conservative_vars"] = py::cast(b.conservative_vars);
  d["primitive_vars"] = py::cast(b.primitive_vars);
  d["implicit_vars"] = py::cast(b.implicit_vars);
  d["implicit_roles"] = py::cast(b.implicit_roles);
  d["newton"] = effective_newton_options_to_dict(b.newton);
  d["positivity_floor"] = b.positivity_floor;
  d["wave_speed_cache"] = b.wave_speed_cache;
  d["weno_epsilon"] = b.weno_epsilon;
  d["physical"] = physical;
  return d;
}

inline py::dict effective_poisson_options_to_dict(const EffectivePoissonOptions& p) {
  py::dict d;
  d["rhs"] = p.rhs;
  d["solver"] = p.solver;
  d["bc"] = p.bc;
  d["wall"] = p.wall;
  d["wall_radius"] = p.wall_radius;
  d["epsilon"] = p.epsilon;
  d["rel_tol"] = p.rel_tol;  // ADC-613: effective GeometricMG V-cycle knobs
  d["abs_tol"] = p.abs_tol;
  d["max_cycles"] = p.max_cycles;
  d["min_coarse"] = p.min_coarse;
  d["pre_smooth"] = p.pre_smooth;
  d["post_smooth"] = p.post_smooth;
  d["bottom_sweeps"] = p.bottom_sweeps;
  d["coarse_threshold"] = p.coarse_threshold;  // ADC-644: total-cell coarsening ceiling.
  d["smoother"] = p.smoother;
  d["coarse"] = p.coarse;
  d["has_epsilon_field"] = p.has_epsilon_field;
  d["has_anisotropic_epsilon"] = p.has_anisotropic_epsilon;
  d["has_reaction_field"] = p.has_reaction_field;
  return d;
}

inline py::dict effective_eb_options_to_dict(const EffectiveEbOptions& e) {
  py::dict d;
  d["enabled"] = e.enabled;
  d["geometry_mode"] = e.geometry_mode;
  d["kappa_min"] = e.kappa_min;
  d["face_open_eps"] = e.face_open_eps;
  d["cut_theta_min"] = e.cut_theta_min;
  return d;
}

inline py::dict effective_refinement_options_to_dict(const EffectiveRefinementOptions& r) {
  py::dict d;
  if (r.scalar_threshold_available) {
    d["threshold"] = r.threshold;
    d["variable"] = r.variable;
    d["role"] = r.role;
  }
  d["disabled"] = r.disabled;
  d["disabled_policy"] = r.disabled_policy;
  py::dict tagging;
  tagging["provider_identity"] = r.tagging_provider_identity;
  tagging["authority"] = r.tagging_authority;
  tagging["execution_provider_identity"] = r.tagging_execution_provider_identity;
  d["tagging_provider"] = std::move(tagging);
  d["phi_grad_threshold"] = r.phi_grad_threshold;
  d["phi_refinement_enabled"] = r.phi_refinement_enabled;
  py::dict clustering;
  clustering["provider_identity"] = r.clustering_provider_identity;
  clustering["authority"] = r.clustering_authority;
  d["clustering_provider"] = std::move(clustering);
  if (r.clustering_parameters_available) {
    d["cluster_min_efficiency"] = r.cluster_min_efficiency;
    d["cluster_min_box_size"] = r.cluster_min_box_size;
    d["cluster_max_box_size"] = r.cluster_max_box_size;
  }
  return d;
}

inline py::dict effective_options_report_to_dict(const EffectiveOptionsReport& report) {
  py::list blocks;
  for (const auto& b : report.blocks)
    blocks.append(effective_block_options_to_dict(b));
  py::dict d;
  d["schema_version"] = report.schema_version;
  d["runtime"] = report.runtime;
  d["defaults"] = numerical_defaults_report_to_dict();
  d["blocks"] = blocks;
  d["poisson"] = effective_poisson_options_to_dict(report.poisson);
  py::dict topology;
  topology["dimension"] = report.topology.dimension;
  topology["periodicity"] = py::cast(report.topology.periodicity);
  d["topology"] = std::move(topology);
  d["eb"] = effective_eb_options_to_dict(report.eb);  // ADC-615
  if (report.has_amr)
    d["amr"] = effective_refinement_options_to_dict(report.amr_refinement);
  else
    d["amr"] = py::none();
  return d;
}

// Per-area binding registration, each defined in its own TU (init_core.cpp / init_system.cpp /
// init_amr.cpp). bindings.cpp calls them in this order: init_core registers SystemConfig / ModelSpec
// (used by System / AmrSystem signatures) before init_system / init_amr run.
void init_core(py::module_& m);
void init_identity(py::module_& m);
void init_component_loader(py::module_& m);
void init_parallel_hdf5(py::module_& m);
void init_system(py::module_& m);
void init_amr(py::module_& m);
