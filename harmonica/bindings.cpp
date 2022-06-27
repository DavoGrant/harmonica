#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bindings.hpp"
#include "orbit/trajectories.hpp"
#include "light_curve/fluxes.hpp"

namespace py = pybind11;


void compute_harmonica_light_curve(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  int ld_law,
  py::array_t<double, py::array::c_style> us_py,
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> fs_py,
  int pnl_c, int pnl_e) {

  // Unpack python arrays.
  auto us_py_ = us_py.unchecked<1>();
  int n_us = us_py_.shape(0);
  double us[n_us];
  for (int i = 0; i < n_us; ++i) {
    us[i] = us_py_(i);
  }
  auto rs_py_ = rs_py.unchecked<1>();
  int n_rs = rs_py_.shape(0);
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = rs_py_(i);
  }
  auto times_py_ = times_py.unchecked<1>();
  int n_times = times_py_.shape(0);
  auto fs_py_ = fs_py.mutable_unchecked<1>();

  // Iterate times.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega, false);
  Fluxes flux(ld_law, us, n_rs, rs, pnl_c, pnl_e, false);
  for (int i = 0; i < n_times; i++) {

    // Compute orbital trajectories.
    double d, z, nu;
    if (ecc == 0.) {
      // Circular case.
      orbital.compute_circular_orbit(times_py_(i), d, z, nu, NULL, NULL);
    } else {
      // Eccentric case.
      orbital.compute_eccentric_orbit(times_py_(i), d, z, nu, NULL, NULL);
    }

    // Compute transit flux.
    flux.transit_flux(d, z, nu, fs_py_(i), NULL);
  }
}


void compute_transmission_string(
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> thetas_py,
  py::array_t<double, py::array::c_style> transmission_string_py) {

  // Unpack python arrays.
  auto rs_py_ = rs_py.unchecked<1>();
  int n_rs = rs_py_.shape(0);
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = rs_py_(i);
  }
  auto thetas_py_ = thetas_py.mutable_unchecked<1>();
  auto transmission_string_py_ = transmission_string_py.mutable_unchecked<1>();

  // Compute transmission string.
  Fluxes flux(0, NULL, n_rs, rs, 0, 0, false);
  for (int i = 0; i < thetas_py_.shape(0); i++) {
    transmission_string_py_(i) = flux.rp_theta(thetas_py_(i));
  }
}


const void jax_quad_ld_light_curve(void* out_tuple, const void** in) {

  // Unpack input meta.
  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  // Unpack input data structures.
  // The params are const through time so only require first index.
  double* times = (double*) in[2];
  double t0 = ((double*) in[3])[0];
  double period = ((double*) in[4])[0];
  double a = ((double*) in[5])[0];
  double inc = ((double*) in[6])[0];
  double ecc = ((double*) in[7])[0];
  double omega = ((double*) in[8])[0];
  double us[2];
  for (int i = 0; i < 2; ++i) {
    us[i] = ((double*) in[i + 9])[0];
  }
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = ((double*) in[i + 11])[0];
  }

  // Unpack output data structures.
  int n_x_derivatives = 6;
  int n_y_derivatives = 4 + n_rs;
  int n_z_derivatives = 8 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  // Iterate times.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega, true);
  Fluxes flux(0, us, n_rs, rs, 50, 500, true);
  for (int i = 0; i < n_times; i++) {

    // Compute orbit and derivatives wrt x={t0, p, a, i, e, w}.
    double d, z, nu;
    double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
    orbital.compute_eccentric_orbit(times[i], d, z, nu, dd_dx, dnu_dx);

    // Compute flux and derivatives wrt y={d, nu, {us}, {rs}}.
    double df_dy[n_y_derivatives];
    flux.transit_flux(d, z, nu, f[i], df_dy);

    // Compute total derivatives wrt z={t0, p, a, i, e, w, {us}, {rs}}.
    int idx_ravel = i * n_z_derivatives;
    df_dz[idx_ravel] = df_dy[0] * dd_dx[0] + df_dy[1] * dd_dx[0];
    df_dz[idx_ravel + 1] = df_dy[0] * dd_dx[1] + df_dy[1] * dd_dx[1];
    df_dz[idx_ravel + 2] = df_dy[0] * dd_dx[2] + df_dy[1] * dd_dx[2];
    df_dz[idx_ravel + 3] = df_dy[0] * dd_dx[3] + df_dy[1] * dd_dx[3];
    df_dz[idx_ravel + 4] = df_dy[0] * dd_dx[4] + df_dy[1] * dd_dx[4];
    df_dz[idx_ravel + 5] = df_dy[0] * dd_dx[5] + df_dy[1] * dd_dx[5];
    df_dz[idx_ravel + 6] = df_dy[2];
    df_dz[idx_ravel + 7] = df_dy[3];
    for (int j = 0; j < n_rs + 1; j++) {
      df_dz[idx_ravel + 8 + j] = df_dy[4 + j];
    }
  }
}


const void jax_nl_ld_light_curve(void* out_tuple, const void** in) {

  // todo: similar to above but with additional ld terms.

}


template <typename T>
py::capsule encapsulate(T* fn) {
  // JAX callables must be wrapped in this py::capsule.
  return py::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}


py::dict jax_registrations() {
  // Dictionary of JAX callables.
  py::dict dict;
  dict["jax_quad_ld_lc"] = encapsulate(jax_quad_ld_light_curve);
  dict["jax_nl_ld_lc"] = encapsulate(jax_nl_ld_light_curve);
  return dict;
}


PYBIND11_MODULE(bindings, m) {

  m.def("light_curve", &compute_harmonica_light_curve,
    py::arg("t0") = py::none(),
    py::arg("period") = py::none(),
    py::arg("a") = py::none(),
    py::arg("inc") = py::none(),
    py::arg("ecc") = py::none(),
    py::arg("omega") = py::none(),
    py::arg("ld_law") = py::none(),
    py::arg("us") = py::none(),
    py::arg("rs") = py::none(),
    py::arg("times") = py::none(),
    py::arg("fs") = py::none(),
    py::arg("pnl_c") = 50,
    py::arg("pnl_e") = 500);

  m.def("transmission_string", &compute_transmission_string,
    py::arg("rs") = py::none(),
    py::arg("thetas") = py::none(),
    py::arg("transmission_string") = py::none());

  m.def("jax_registrations", &jax_registrations);

}
