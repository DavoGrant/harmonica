#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bindings.hpp"
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

  int n_times = times_.shape(0);
  int n_partials = ds_grad_.shape(1);
  double* ds_partials[n_partials];
  double* nus_partials[n_partials];

  // Compute orbital trajectories.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega,
                            require_gradients);
  if (ecc == 0.) {

    // Circular case.
    for (int i = 0; i < n_times; i++) {
      for (int j = 0; j < n_partials; j++) {
        ds_partials[j] = &ds_grad_(i, j);
        nus_partials[j] = &nus_grad_(i, j);
      }
      orbital.compute_circular_orbit(times_(i), ds_(i), nus_(i),
                                     ds_partials, nus_partials);
    }
  } else {

    // Eccentric case.
    for (int i = 0; i < n_times; i++) {
      for (int j = 0; j < n_partials; j++) {
        ds_partials[j] = &ds_grad_(i, j);
        nus_partials[j] = &nus_grad_(i, j);
      }
      orbital.compute_eccentric_orbit(times_(i), ds_(i), nus_(i),
                                      ds_partials, nus_partials);
    }
  }
}


void compute_harmonica_light_curve(
  int ld_law,
  py::array_t<double, py::array::c_style> us,
  py::array_t<double, py::array::c_style> rs,
  py::array_t<double, py::array::c_style> ds,
  py::array_t<double, py::array::c_style> nus,
  py::array_t<double, py::array::c_style> fs,
  py::array_t<double, py::array::c_style> ds_grad,
  py::array_t<double, py::array::c_style> nus_grad,
  py::array_t<double, py::array::c_style> fs_grad,
  bool require_gradients) {

  // Unpack python arrays.
  auto ds_ = ds.unchecked<1>();
  auto nus_ = nus.unchecked<1>();
  auto ds_grad_ = ds_grad.unchecked<2>();
  auto nus_grad_ = nus_grad.unchecked<2>();

  auto fs_ = fs.mutable_unchecked<1>();
  auto fs_grad_ = fs_grad.mutable_unchecked<2>();

  int n_positions = ds_.shape(0);
  int n_dnu_partials = ds_grad_.shape(1);
  int n_f_partials = fs_grad_.shape(1);
  const double* ds_partials[n_dnu_partials];
  const double* nus_partials[n_dnu_partials];
  double* fs_partials[n_f_partials];

  // Compute transit light curve.
  Fluxes flux(ld_law, us, rs, require_gradients);
  for (int i = 0; i < n_positions; i++) {
    for (int j = 0; j < n_dnu_partials; j++) {
      ds_partials[j] = &ds_grad_(i, j);
      nus_partials[j] = &nus_grad_(i, j);
    }
    for (int j = 0; j < n_f_partials; j++) {
      fs_partials[j] = &fs_grad_(i, j);
    }
    flux.transit_flux(ds_(i), nus_(i), fs_(i),
                      ds_partials, nus_partials, fs_partials);
  }

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
      py::arg("ld_law") = py::none(),
      py::arg("us") = py::none(),
      py::arg("rs") = py::none(),
      py::arg("ds") = py::none(),
      py::arg("nus") = py::none(),
      py::arg("fs") = py::none(),
      py::arg("ds_grad") = py::none(),
      py::arg("nus_grad") = py::none(),
      py::arg("fs_grad") = py::none(),
      py::arg("require_gradients") = false);

}
