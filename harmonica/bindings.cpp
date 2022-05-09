#include <pybind11/pybind11.h>

#include "orbit/trajectories.hpp"
#include "light_curve/fluxes.hpp"

namespace py = pybind11;


void compute_orbital_separation_and_angles(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  py::array_t<double, py::array::c_style> times,
  py::array_t<double, py::array::c_style> ds,
  py::array_t<double, py::array::c_style> nus,
  py::array_t<double, py::array::c_style> ds_grad,
  py::array_t<double, py::array::c_style> nus_grad,
  bool require_gradients) {

  // Unpack python arrays.
  auto times_ = times.unchecked<1>();
  auto ds_ = ds.mutable_unchecked<1>();
  auto nus_ = nus.mutable_unchecked<1>();
  auto ds_grad_ = ds_grad.mutable_unchecked<2>();
  auto nus_grad_ = nus_grad.mutable_unchecked<2>();

  // Compute orbital trajectories.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega);
  if (ecc == 0.) {
    // Circular case.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {
      orbital.compute_circular_orbit(times_(i), ds_(i), nus_(i),
                                     ds_grad_(i, 0), ds_grad_(i, 1),
                                     ds_grad_(i, 2), ds_grad_(i, 3),
                                     nus_grad_(i, 0), nus_grad_(i, 1),
                                     nus_grad_(i, 2), nus_grad_(i, 3),
                                     require_gradients);
    }
  } else {
    // Eccentric case.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {
      orbital.compute_eccentric_orbit(times_(i), ds_(i), nus_(i),
                                      ds_grad_(i, 0), ds_grad_(i, 1),
                                      ds_grad_(i, 2), ds_grad_(i, 3),
                                      ds_grad_(i, 4), ds_grad_(i, 5),
                                      nus_grad_(i, 0), nus_grad_(i, 1),
                                      nus_grad_(i, 2), nus_grad_(i, 3),
                                      nus_grad_(i, 4), nus_grad_(i, 5),
                                      require_gradients);
    }
  }
}


void compute_harmonica_light_curve(
  bool require_gradients) {



}


PYBIND11_MODULE(bindings, m) {

    m.def("orbit", &compute_orbital_separation_and_angles,
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

    m.def("light_curve", &compute_harmonica_light_curve,
      py::arg("require_gradients") = false);

}
