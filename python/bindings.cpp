// Bindings pybind11 de la FACADE compilee (libadc). On expose les solveurs
// CONCRETS et sans template (DiocotronSolver, EulerPoissonSolver) : c'est tout
// l'interet du src/ -> une surface stable et bindable, jamais Coupler<Model,...>.
// Construit seulement avec -DADC_BUILD_PYTHON=ON.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <adc/solver/diocotron_solver.hpp>
#include <adc/solver/euler_poisson_solver.hpp>

#include <cstring>
#include <vector>

namespace py = pybind11;
using namespace adc;

// densite (n*n row-major) -> tableau numpy (ny, nx) (copie).
static py::array_t<double> to_2d(const std::vector<double>& v, int n) {
  py::array_t<double> a({n, n});
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
  return a;
}

PYBIND11_MODULE(adc, m) {
  m.doc() =
      "adc_cpp : solveurs hyperbolique-elliptique (facade compilee libadc). "
      "Le coeur generique reste C++ template ; ici les solveurs concrets.";

  py::class_<DiocotronConfig>(m, "DiocotronConfig")
      .def(py::init<>())
      .def_readwrite("n", &DiocotronConfig::n)
      .def_readwrite("L", &DiocotronConfig::L)
      .def_readwrite("B0", &DiocotronConfig::B0)
      .def_readwrite("n_i0", &DiocotronConfig::n_i0)
      .def_readwrite("alpha", &DiocotronConfig::alpha)
      .def_readwrite("eps", &DiocotronConfig::eps)
      .def_readwrite("poisson_per_stage", &DiocotronConfig::poisson_per_stage);

  py::class_<DiocotronSolver>(m, "DiocotronSolver")
      .def(py::init<const DiocotronConfig&>())
      .def("step", &DiocotronSolver::step, py::arg("dt"))
      .def("mass", &DiocotronSolver::mass)
      .def("time", &DiocotronSolver::time)
      .def("nx", &DiocotronSolver::nx)
      .def("density",
           [](const DiocotronSolver& s) { return to_2d(s.density(), s.nx()); });

  py::class_<EulerPoissonConfig>(m, "EulerPoissonConfig")
      .def(py::init<>())
      .def_readwrite("n", &EulerPoissonConfig::n)
      .def_readwrite("L", &EulerPoissonConfig::L)
      .def_readwrite("gamma", &EulerPoissonConfig::gamma)
      .def_readwrite("four_pi_G", &EulerPoissonConfig::four_pi_G)
      .def_readwrite("rho0", &EulerPoissonConfig::rho0)
      .def_readwrite("p0", &EulerPoissonConfig::p0)
      .def_readwrite("eps", &EulerPoissonConfig::eps)
      .def_readwrite("poisson_per_stage", &EulerPoissonConfig::poisson_per_stage)
      .def_readwrite("use_fft", &EulerPoissonConfig::use_fft);

  py::class_<EulerPoissonSolver>(m, "EulerPoissonSolver")
      .def(py::init<const EulerPoissonConfig&>())
      .def("step", &EulerPoissonSolver::step, py::arg("dt"))
      .def("mass", &EulerPoissonSolver::mass)
      .def("energy", &EulerPoissonSolver::energy)
      .def("total_momentum", &EulerPoissonSolver::total_momentum, py::arg("dir"))
      .def("time", &EulerPoissonSolver::time)
      .def("nx", &EulerPoissonSolver::nx)
      .def("density", [](const EulerPoissonSolver& s) {
        return to_2d(s.density(), s.nx());
      });
}
