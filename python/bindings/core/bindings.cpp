// pybind11 bindings of the PoPS LIB: compiles the `_pops` module. Exposes the
// runtime composition facade `System` (the tutor's "coupler / system") + its
// config. Python composes WHAT to assemble (model + spatial scheme + temporal
// treatment + per-block substeps, system Poisson); all the cell-by-cell compute
// stays in the compiled lib. The readable sugar (Spatial, Explicit, IMEX,
// System) is added by the Python package pops/__init__.py.
// Built only with -DPOPS_BUILD_PYTHON=ON.
//
// ADC-365: the py::class_/.def surface is split across init_core / init_system / init_amr (each its own
// TU, declared in bindings_detail.hpp) so they compile in parallel with lower peak pybind memory. This
// file is the thin PYBIND11_MODULE that calls them in order (init_core first: it registers SystemConfig
// and ModelSpec, which the System / AmrSystem signatures reference).

#include "bindings_detail.hpp"
#include <pops/runtime/program/step_transaction.hpp>

#include <exception>

namespace {

using NativeStepAttemptRejected = pops::runtime::program::StepAttemptRejected;

// pybind11's ordinary register_exception translator preserves only what().  Attempt rejection is a
// control protocol, not an opaque diagnostic: keep its C++ status and retry disposition structured
// all the way to the Python controller.  The GIL-safe storage follows pybind11's own
// register_exception_impl lifetime model and avoids a process-global borrowed PyObject pointer.
PYBIND11_CONSTINIT py::gil_safe_call_once_and_store<py::exception<NativeStepAttemptRejected>>
    step_attempt_rejected_type;

void translate_step_attempt_rejected(std::exception_ptr error) {
  if (!error)
    return;
  try {
    std::rethrow_exception(error);
  } catch (const NativeStepAttemptRejected& rejected) {
    auto& error_type = step_attempt_rejected_type.get_stored();
    py::object instance = error_type.attr("__call__")(py::str(rejected.what()));
    instance.attr("status") = py::str(pops::solve_status_name(rejected.status()));
    instance.attr("phase") = py::str(rejected.phase());
    instance.attr("detail") = py::str(rejected.detail());
    instance.attr("disposition") =
        py::str(pops::runtime::program::step_attempt_disposition_name(rejected.disposition()));
    instance.attr("reason_code") = py::int_(rejected.reason_code());
    PyErr_SetObject(error_type.ptr(), instance.ptr());
  }
}

void register_step_attempt_rejected(py::module_& module) {
  step_attempt_rejected_type
      .call_once_and_store_result([&] {
        return py::exception<NativeStepAttemptRejected>(module, "StepAttemptRejected",
                                                        PyExc_RuntimeError);
      })
      .get_stored();
  py::register_local_exception_translator(&translate_step_attempt_rejected);
}

}  // namespace

PYBIND11_MODULE(_pops, m) {
  register_step_attempt_rejected(m);
  init_core(m);
  init_identity(m);
  init_component_loader(m);
  init_parallel_hdf5(m);
  init_system(m);
  init_amr(m);
}
