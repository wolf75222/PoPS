#include "../bindings_detail.hpp"

#include <pops/core/identity/canonical_value.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/config/component_manifest.hpp>

#include <cstdint>
#include <unordered_set>

namespace {

using pops::identity::CanonicalValue;

class ActiveContainer {
 public:
  ActiveContainer(std::unordered_set<PyObject*>& active, PyObject* value) : active_(active), value_(value) {
    if (!active_.insert(value_).second)
      throw py::value_error("canonical identity contains a container cycle");
  }
  ~ActiveContainer() { active_.erase(value_); }
  ActiveContainer(const ActiveContainer&) = delete;
  ActiveContainer& operator=(const ActiveContainer&) = delete;

 private:
  std::unordered_set<PyObject*>& active_;
  PyObject* value_;
};

CanonicalValue from_python(const py::handle& value, std::unordered_set<PyObject*>& active) {
  if (value.is_none())
    return CanonicalValue();
  if (PyBool_Check(value.ptr()))
    return CanonicalValue(value.ptr() == Py_True);
  if (PyLong_Check(value.ptr())) {
    const long long integer = PyLong_AsLongLong(value.ptr());
    if (integer == -1 && PyErr_Occurred())
      throw py::error_already_set();
    return CanonicalValue(static_cast<std::int64_t>(integer));
  }
  if (PyUnicode_Check(value.ptr()))
    return CanonicalValue::text(py::cast<std::string>(value));
  if (PyBytes_Check(value.ptr())) {
    char* data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(value.ptr(), &data, &size) != 0)
      throw py::error_already_set();
    return CanonicalValue::bytes(CanonicalValue::Bytes(
        reinterpret_cast<const std::uint8_t*>(data),
        reinterpret_cast<const std::uint8_t*>(data) + size));
  }
  if (PyDict_Check(value.ptr())) {
    ActiveContainer guard(active, value.ptr());
    CanonicalValue::Map items;
    items.reserve(static_cast<std::size_t>(PyDict_Size(value.ptr())));
    for (auto pair : py::reinterpret_borrow<py::dict>(value)) {
      if (!PyUnicode_Check(pair.first.ptr()))
        throw py::type_error("canonical identity mapping keys must be strings");
      items.emplace_back(py::cast<std::string>(pair.first), from_python(pair.second, active));
    }
    return CanonicalValue::map(std::move(items));
  }
  if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
    ActiveContainer guard(active, value.ptr());
    CanonicalValue::Array items;
    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    items.reserve(static_cast<std::size_t>(sequence.size()));
    for (const py::handle item : sequence)
      items.push_back(from_python(item, active));
    return CanonicalValue::array(std::move(items));
  }
  if (PySet_Check(value.ptr()) || PyFrozenSet_Check(value.ptr())) {
    ActiveContainer guard(active, value.ptr());
    CanonicalValue::Array items;
    const py::set set = py::reinterpret_borrow<py::set>(value);
    items.reserve(static_cast<std::size_t>(set.size()));
    for (const py::handle item : set)
      items.push_back(from_python(item, active));
    return CanonicalValue::set(std::move(items));
  }
  throw py::type_error(
      "canonical identity supports only None, bool, int64, str, bytes, list/tuple, "
      "dict[str, ...], set and frozenset");
}

CanonicalValue from_python(const py::handle& value) {
  std::unordered_set<PyObject*> active;
  return from_python(value, active);
}

[[noreturn]] void raise_component_manifest_error(
    const pops::component_manifest::Error& error) {
  const py::object error_type =
      py::module_::import("pops.model._component_manifest").attr("ComponentManifestError");
  const py::object instance = error_type(error.code(), error.path(), std::string(error.what()));
  PyErr_SetObject(error_type.ptr(), instance.ptr());
  throw py::error_already_set();
}

}  // namespace

void init_identity(py::module_& m) {
  m.def(
      "_identity_canonical_bytes",
      [](const py::handle& value) {
        const auto bytes = pops::identity::canonical_bytes(from_python(value));
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      },
      "Private deterministic CBOR encoder for cross-language identity parity.");
  m.def(
      "_identity_sha256",
      [](const py::handle& value) {
        return pops::identity::sha256_hex(pops::identity::canonical_bytes(from_python(value)));
      },
      "Private SHA-256 of the deterministic identity encoding.");
  m.def(
      "_component_manifest_canonical_bytes",
      [](const py::handle& value) {
        try {
          const auto bytes = pops::component_manifest::canonical_bytes(from_python(value));
          return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        } catch (const pops::component_manifest::Error& error) {
          raise_component_manifest_error(error);
        }
      },
      "Validate and serialize the complete schema-v2 ComponentManifest in native code.");
  m.def(
      "_component_manifest_semantic_bytes",
      [](const py::handle& value) {
        try {
          const auto normalized = pops::component_manifest::normalize(from_python(value));
          const auto bytes = pops::identity::canonical_bytes(normalized.semantic_payload);
          return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        } catch (const pops::component_manifest::Error& error) {
          raise_component_manifest_error(error);
        }
      },
      "Native canonical bytes of the semantic ComponentManifest payload.");
}
