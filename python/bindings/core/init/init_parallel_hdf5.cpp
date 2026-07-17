#include "../bindings_detail.hpp"

#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/output/hdf5_collective.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::size_t exact_extent(const py::handle& value, const char* where) {
  if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
    throw py::type_error(std::string(where) + " must contain exact non-negative integers");
  const auto result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() != nullptr)
    throw py::error_already_set();
  if (result > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error(std::string(where) + " exceeds size_t");
  return static_cast<std::size_t>(result);
}

[[nodiscard]] std::vector<std::size_t> exact_shape(const py::handle& value, const char* where) {
  if (!py::isinstance<py::tuple>(value) && !py::isinstance<py::list>(value))
    throw py::type_error(std::string(where) + " must be a tuple/list of extents");
  const auto sequence = py::reinterpret_borrow<py::sequence>(value);
  std::vector<std::size_t> result;
  result.reserve(sequence.size());
  for (const auto item : sequence)
    result.push_back(exact_extent(item, where));
  return result;
}

[[nodiscard]] pops::runtime::output::ArrayView array_view(const py::handle& value,
                                                          std::vector<py::array>& owners,
                                                          const char* where) {
  if (!py::isinstance<py::array>(value))
    throw py::type_error(std::string(where) + " must be an exact NumPy array");
  auto array = py::reinterpret_borrow<py::array>(value);
  if ((array.flags() & py::array::c_style) == 0)
    throw py::value_error(std::string(where) + " must be C-contiguous");
  if (!array.writeable()) {
    // Read-only native snapshot buffers are expected and supported.  This branch documents that
    // the adapter never requests a mutable view; no copy is introduced.
  }
  std::vector<std::size_t> shape;
  shape.reserve(static_cast<std::size_t>(array.ndim()));
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    if (array.shape(axis) <= 0)
      throw py::value_error(std::string(where) + " dimensions must be positive");
    shape.push_back(static_cast<std::size_t>(array.shape(axis)));
  }
  const auto dtype = py::str(array.dtype().attr("str")).cast<std::string>();
  const auto bytes = static_cast<std::size_t>(array.nbytes());
  const void* data = array.data();
  owners.push_back(std::move(array));
  return {dtype, std::move(shape), data, bytes};
}

void require_exact_keys(const py::dict& value, std::initializer_list<const char*> expected,
                        const char* where) {
  if (value.size() != static_cast<py::ssize_t>(expected.size()))
    throw py::value_error(std::string(where) + " keys are not exact");
  for (const auto* key : expected)
    if (!value.contains(key))
      throw py::value_error(std::string(where) + " keys are not exact");
}

[[nodiscard]] std::pair<std::size_t, std::size_t> exact_pair(const py::handle& value,
                                                             const char* where) {
  const auto shape = exact_shape(value, where);
  if (shape.size() != 2)
    throw py::value_error(std::string(where) + " must contain two extents");
  return {shape[0], shape[1]};
}

}  // namespace

void init_parallel_hdf5(py::module_& m) {
  const auto capability = pops::runtime::output::parallel_hdf5_capability();
  m.attr("__has_parallel_hdf5__") = capability.available;
  m.def("_parallel_hdf5_capability", [] {
    const auto value = pops::runtime::output::parallel_hdf5_capability();
    py::dict result;
    result["available"] = value.available;
    result["hdf5_version"] = value.hdf5_version;
    result["reason"] = value.reason;
    result["communicator"] = "MPI_COMM_WORLD";
    result["implementation"] = "C++ HDF5 C API";
    return result;
  });
  m.def(
      "_write_parallel_hdf5",
      [](const pops::WorldCommunicator& world, const py::object& path_value,
         const py::object& manifest_value, const py::object& root_arrays_value,
         const py::object& field_rows_value) {
        std::vector<py::array> owners;
        std::vector<pops::runtime::output::NamedArrayView> arrays;
        std::vector<pops::runtime::output::FieldView> fields;
        std::string path;
        std::string manifest_json;
        std::string local_error;
        try {
          if (!PyUnicode_CheckExact(path_value.ptr()) ||
              !PyUnicode_CheckExact(manifest_value.ptr()))
            throw py::type_error("native HDF5 path and manifest must be exact strings");
          if (!PyDict_CheckExact(root_arrays_value.ptr()) ||
              !PyTuple_CheckExact(field_rows_value.ptr()))
            throw py::type_error("native HDF5 root arrays/fields must be an exact dict and tuple");
          path = path_value.cast<std::string>();
          manifest_json = manifest_value.cast<std::string>();
          const auto root_arrays = py::reinterpret_borrow<py::dict>(root_arrays_value);
          const auto field_rows = py::reinterpret_borrow<py::tuple>(field_rows_value);
          arrays.reserve(root_arrays.size());
          for (const auto item : root_arrays) {
            if (!py::isinstance<py::str>(item.first))
              throw py::type_error("native HDF5 root-array names must be exact strings");
            arrays.push_back({py::cast<std::string>(item.first),
                              array_view(item.second, owners, "native HDF5 root array")});
          }
          std::sort(arrays.begin(), arrays.end(), [](const auto& left, const auto& right) {
            return left.dataset < right.dataset;
          });

          fields.reserve(field_rows.size());
          for (const auto field_item : field_rows) {
            if (!py::isinstance<py::dict>(field_item))
              throw py::type_error("native HDF5 fields must be exact descriptor dicts");
            const auto field = py::reinterpret_borrow<py::dict>(field_item);
            require_exact_keys(field, {"dataset", "dtype", "shape", "pieces"},
                               "native HDF5 field descriptor");
            pops::runtime::output::FieldView result;
            result.dataset = py::cast<std::string>(field["dataset"]);
            result.dtype = py::cast<std::string>(field["dtype"]);
            result.shape = exact_shape(field["shape"], "native HDF5 field shape");
            if (!py::isinstance<py::tuple>(field["pieces"]))
              throw py::type_error("native HDF5 field pieces must be an exact tuple");
            const auto pieces = py::reinterpret_borrow<py::tuple>(field["pieces"]);
            result.pieces.reserve(pieces.size());
            for (const auto piece_item : pieces) {
              if (!py::isinstance<py::dict>(piece_item))
                throw py::type_error("native HDF5 pieces must be exact descriptor dicts");
              const auto piece = py::reinterpret_borrow<py::dict>(piece_item);
              require_exact_keys(piece, {"lower", "upper", "values"},
                                 "native HDF5 piece descriptor");
              const auto [jlo, ilo] = exact_pair(piece["lower"], "native HDF5 piece lower");
              const auto [jhi, ihi] = exact_pair(piece["upper"], "native HDF5 piece upper");
              result.pieces.push_back(
                  {jlo, ilo, jhi, ihi,
                   array_view(piece["values"], owners, "native HDF5 piece values")});
            }
            fields.push_back(std::move(result));
          }
        } catch (const std::exception& exc) {
          local_error = exc.what();
        }
        {
          py::gil_scoped_release release;
          pops::runtime::output::collective_hdf5_input_consensus(world, local_error);
          pops::runtime::output::write_collective_hdf5(world, path, manifest_json, arrays, fields);
        }
      },
      py::arg("world"), py::arg("path"), py::arg("manifest_json"), py::arg("root_arrays"),
      py::arg("fields"),
      "Write one exact artifact through native parallel HDF5 on MPI_COMM_WORLD.");
}
