#include <pybind11/pybind11.h>

#include "orbit/trajectories.hpp"
#include "orbit/kepler.hpp"
#include "light_curve/fluxes.hpp"

namespace py = pybind11;


PYBIND11_MODULE(bindings, m) {

    m.def("orbit", &orbital_trajectories,
          py::arg("t0") = py::none(),
          py::arg("period") = py::none(),
          py::arg("a") = py::none(),
          py::arg("inc") = py::none(),
          py::arg("ecc") = py::none(),
          py::arg("omega") = py::none(),
          py::arg("times") = py::none(),
          py::arg("ds") = py::none(),
          py::arg("nus") = py::none(),
          py::arg("ds_grad") = py::none(),
          py::arg("nus_grad") = py::none(),
          py::arg("require_gradients") = false);

    m.def("light_curve", &light_curve_fluxes);

    m.def("test_solve_kepler", &solve_kepler);

}
