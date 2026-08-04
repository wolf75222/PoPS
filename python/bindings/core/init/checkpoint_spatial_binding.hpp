#pragma once

#include "../bindings_detail.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/checkpoint/spatial_contract.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::python::detail {

inline void require_checkpoint_spatial_keys(const py::dict& data) {
  constexpr std::array<const char*, 9> keys{
      "schema_version",
      "dimension",
      "shape",
      "lower",
      "upper",
      "periodicity",
      "refinement_ratios",
      "native_layout_identity",
      "identity",
  };
  if (data.size() != static_cast<py::ssize_t>(keys.size()))
    throw py::type_error("checkpoint spatial contract has an unsupported exact schema");
  for (const char* key : keys)
    if (!data.contains(py::str(key)))
      throw py::type_error("checkpoint spatial contract has an unsupported exact schema");
}

inline int checkpoint_spatial_exact_int(const py::handle& value, const char* where) {
  if (!PyLong_CheckExact(value.ptr()))
    throw py::type_error(std::string(where) + " must be an exact integer");
  return py::cast<int>(value);
}

inline std::int64_t checkpoint_spatial_exact_int64(const py::handle& value, const char* where) {
  if (!PyLong_CheckExact(value.ptr()))
    throw py::type_error(std::string(where) + " must be an exact integer");
  return py::cast<std::int64_t>(value);
}

inline double checkpoint_spatial_hex_float(const py::handle& value, const char* where) {
  if (!PyUnicode_CheckExact(value.ptr()))
    throw py::type_error(std::string(where) + " must be a float.hex string");
  const std::string text = py::cast<std::string>(value);
  std::size_t consumed = 0;
  double result = 0.0;
  try {
    result = std::stod(text, &consumed);
  } catch (const std::exception&) {
    throw py::value_error(std::string(where) + " contains invalid float.hex data");
  }
  if (consumed != text.size() || !std::isfinite(result))
    throw py::value_error(std::string(where) + " contains invalid float.hex data");
  return result;
}

inline std::string checkpoint_spatial_exact_string(const py::handle& value, const char* where) {
  if (!PyUnicode_CheckExact(value.ptr()))
    throw py::type_error(std::string(where) + " must be an exact string");
  return py::cast<std::string>(value);
}

inline py::list checkpoint_spatial_exact_list(const py::handle& value, const char* where) {
  if (!PyList_CheckExact(value.ptr()))
    throw py::type_error(std::string(where) + " must be an exact list");
  return py::reinterpret_borrow<py::list>(value);
}

template <int Dim>
std::vector<std::int64_t> prepare_checkpoint_spatial_contract(const py::dict& data) {
  using checkpoint::EncodedSpatialContract;
  using checkpoint::decode_spatial_contract;

  require_checkpoint_spatial_keys(data);
  EncodedSpatialContract encoded;
  encoded.schema_version =
      checkpoint_spatial_exact_int(data["schema_version"], "checkpoint spatial schema_version");
  encoded.dimension =
      checkpoint_spatial_exact_int(data["dimension"], "checkpoint spatial dimension");
  // Refuse an incompatible variant before decoding or allocating any vector payload.
  if (encoded.schema_version != checkpoint::kSpatialContractSchemaVersion)
    throw py::value_error("checkpoint spatial schema version is unsupported");
  if (encoded.dimension != Dim)
    throw py::value_error("checkpoint dimension does not match the loaded native specialization");

  const auto shape = checkpoint_spatial_exact_list(data["shape"], "checkpoint spatial shape");
  const auto lower = checkpoint_spatial_exact_list(data["lower"], "checkpoint spatial lower");
  const auto upper = checkpoint_spatial_exact_list(data["upper"], "checkpoint spatial upper");
  const auto periodicity =
      checkpoint_spatial_exact_list(data["periodicity"], "checkpoint spatial periodicity");
  if (shape.size() != static_cast<py::ssize_t>(Dim) ||
      lower.size() != static_cast<py::ssize_t>(Dim) ||
      upper.size() != static_cast<py::ssize_t>(Dim) ||
      periodicity.size() != static_cast<py::ssize_t>(Dim))
    throw py::value_error("checkpoint spatial vectors must have exactly Dim components");
  encoded.shape.reserve(Dim);
  encoded.lower.reserve(Dim);
  encoded.upper.reserve(Dim);
  encoded.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<py::ssize_t>(axis);
    encoded.shape.push_back(
        checkpoint_spatial_exact_int64(shape[index], "checkpoint spatial shape component"));
    encoded.lower.push_back(
        checkpoint_spatial_hex_float(lower[index], "checkpoint spatial lower component"));
    encoded.upper.push_back(
        checkpoint_spatial_hex_float(upper[index], "checkpoint spatial upper component"));
    const py::handle periodic = periodicity[index];
    if (!PyBool_Check(periodic.ptr()))
      throw py::type_error("checkpoint spatial periodicity components must be exact bool values");
    encoded.periodicity.push_back(periodic.ptr() == Py_True ? 1U : 0U);
  }

  const auto ratios =
      checkpoint_spatial_exact_list(data["refinement_ratios"], "checkpoint refinement ratios");
  encoded.refinement_ratios.reserve(static_cast<std::size_t>(ratios.size()));
  for (const py::handle item : ratios) {
    const auto row = checkpoint_spatial_exact_list(item, "checkpoint refinement-ratio vector");
    if (row.size() != static_cast<py::ssize_t>(Dim))
      throw py::value_error("checkpoint refinement-ratio vectors must have exactly Dim components");
    std::vector<int> values;
    values.reserve(Dim);
    for (int axis = 0; axis < Dim; ++axis)
      values.push_back(checkpoint_spatial_exact_int(row[static_cast<py::ssize_t>(axis)],
                                                    "checkpoint refinement-ratio component"));
    encoded.refinement_ratios.push_back(std::move(values));
  }
  encoded.native_layout_identity = checkpoint_spatial_exact_string(
      data["native_layout_identity"], "checkpoint native layout identity");
  encoded.spatial_identity =
      checkpoint_spatial_exact_string(data["identity"], "checkpoint spatial identity");

  const auto prepared = decode_spatial_contract<Dim>(encoded);
  std::vector<std::int64_t> counts;
  counts.reserve(prepared.refinement_ratios.size() + 1);
  for (std::size_t level = 0; level <= prepared.refinement_ratios.size(); ++level) {
    const auto level_shape = prepared.shape_at_level(level);
    std::int64_t count = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      if (count > std::numeric_limits<std::int64_t>::max() / level_shape[axis])
        throw std::overflow_error("checkpoint spatial cell count exceeds int64_t");
      count *= level_shape[axis];
    }
    counts.push_back(count);
  }
  return counts;
}

}  // namespace pops::python::detail
