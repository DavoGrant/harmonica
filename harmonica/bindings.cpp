#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bindings.hpp"
#include "constants/constants.hpp"
#include "orbit/trajectories.hpp"
#include "orbit/gradients.hpp"
#include "light_curve/fluxes.hpp"
#include "light_curve/gradients.hpp"

namespace py = pybind11;


void compute_orbit_trajectories(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_ds_py,
  py::array_t<double, py::array::c_style> out_zs_py,
  py::array_t<double, py::array::c_style> out_nus_py) {

  // Unpack python arrays.
  auto times_py_ = times_py.unchecked<1>();
  int n_times = times_py_.shape(0);
  auto out_ds_py_ = out_ds_py.mutable_unchecked<1>();
  auto out_zs_py_ = out_zs_py.mutable_unchecked<1>();
  auto out_nus_py_ = out_nus_py.mutable_unchecked<1>();

  // Iterate times.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega);
  for (int i = 0; i < n_times; i++) {

    // Compute orbital trajectories.
    if (ecc == 0.) {
      // Circular case.
      orbital.compute_circular_orbit(times_py_(i), out_ds_py_(i),
                                      out_zs_py_(i), out_nus_py_(i));
    } else {
      // Eccentric case.
      orbital.compute_eccentric_orbit(times_py_(i), out_ds_py_(i),
                                       out_zs_py_(i), out_nus_py_(i));
    }
  }
}


void compute_harmonica_light_curve(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  int ld_law,
  py::array_t<double, py::array::c_style> us_py,
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_fs_py,
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
  auto out_fs_py_ = out_fs_py.mutable_unchecked<1>();

  // Iterate times.
  OrbitTrajectories orbital(t0, period, a, inc, ecc, omega);
  Fluxes flux(ld_law, us, n_rs, rs, pnl_c, pnl_e);
  for (int i = 0; i < n_times; i++) {

    // Compute orbital trajectories.
    double d, z, nu;
    if (ecc == 0.) {
      // Circular case.
      orbital.compute_circular_orbit(times_py_(i), d, z, nu);
    } else {
      // Eccentric case.
      orbital.compute_eccentric_orbit(times_py_(i), d, z, nu);
    }

    // Compute transit flux.
    flux.transit_flux(d, z, nu, out_fs_py_(i));
  }
}


void compute_transmission_string(
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> thetas_py,
  py::array_t<double, py::array::c_style> out_transmission_string_py) {

  // Unpack python arrays.
  auto rs_py_ = rs_py.unchecked<1>();
  int n_rs = rs_py_.shape(0);
  double us[2] = {0., 0.};
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = rs_py_(i);
  }
  auto thetas_py_ = thetas_py.mutable_unchecked<1>();
  auto out_transmission_string_py_ =
    out_transmission_string_py.mutable_unchecked<1>();

  // Compute transmission string.
  Fluxes flux(0, us, n_rs, rs, 0, 0);
  for (int i = 0; i < thetas_py_.shape(0); i++) {
    out_transmission_string_py_(i) = flux.rp_theta(thetas_py_(i));
  }
}


const void jax_light_curve_quad_ld(void* out_tuple, const void** in) {

  // Unpack input meta.
  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  // Unpack input data structures. The params are const
  // through time so only require first index.
  double* times = (double*) in[2];
  double t0 = ((double*) in[3])[0];
  double period = ((double*) in[4])[0];
  double a = ((double*) in[5])[0];
  double inc = ((double*) in[6])[0];
  double ecc = ((double*) in[7])[0];
  double omega = ((double*) in[8])[0];
  double us[2] {((double*) in[9])[0], ((double*) in[10])[0]};
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = ((double*) in[i + 11])[0];
  }

  // Unpack output data structures.
  int n_x_derivatives = 6;
  int n_y_derivatives = 2 + 2 + n_rs;
  int n_z_derivatives = 6 + 2 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  // Iterate times.
  OrbitDerivatives orbital(t0, period, a, inc, ecc, omega);
  FluxDerivatives flux(limb_darkening::quadratic, us, n_rs, rs, 20, 50);
  for (int i = 0; i < n_times; i++) {

    // Compute orbit and derivatives wrt x={t0, p, a, i, e, w}.
    double d, z, nu;
    double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
    orbital.compute_eccentric_orbit_and_derivatives(
      times[i], d, z, nu, dd_dx, dnu_dx);

    // Compute flux and derivatives wrt y={d, nu, {us}, {rs}}.
    double df_dy[n_y_derivatives];
    flux.transit_flux_and_derivatives(d, z, nu, f[i], df_dy);

    // Compute total derivatives wrt z={t0, p, a, i, e, w, {us}, {rs}}.
    int idx_ravel = i * n_z_derivatives;
    for (int j = 0; j < n_z_derivatives; j++) {
      if (j < 6) {
        df_dz[idx_ravel + j] = df_dy[0] * dd_dx[j] + df_dy[1] * dnu_dx[j];
      } else {
        df_dz[idx_ravel + j] = df_dy[j - 4];
      }
    }
  }
}


const void jax_light_curve_nonlinear_ld(void* out_tuple, const void** in) {

  // Unpack input meta.
  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  // Unpack input data structures. The params are const
  // through time so only require first index.
  double* times = (double*) in[2];
  double t0 = ((double*) in[3])[0];
  double period = ((double*) in[4])[0];
  double a = ((double*) in[5])[0];
  double inc = ((double*) in[6])[0];
  double ecc = ((double*) in[7])[0];
  double omega = ((double*) in[8])[0];
  double us[4] {((double*) in[9])[0], ((double*) in[10])[0],
                ((double*) in[11])[0], ((double*) in[12])[0]};
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = ((double*) in[i + 13])[0];
  }

  // Unpack output data structures.
  int n_x_derivatives = 6;
  int n_y_derivatives = 2 + 4 + n_rs;
  int n_z_derivatives = 6 + 4 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  // Iterate times.
  OrbitDerivatives orbital(t0, period, a, inc, ecc, omega);
  FluxDerivatives flux(limb_darkening::non_linear, us, n_rs, rs, 20, 50);
  for (int i = 0; i < n_times; i++) {

    // Compute orbit and derivatives wrt x={t0, p, a, i, e, w}.
    double d, z, nu;
    double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
    orbital.compute_eccentric_orbit_and_derivatives(
      times[i], d, z, nu, dd_dx, dnu_dx);

    // Compute flux and derivatives wrt y={d, nu, {us}, {rs}}.
    double df_dy[n_y_derivatives];
    flux.transit_flux_and_derivatives(d, z, nu, f[i], df_dy);

    // Compute total derivatives wrt z={t0, p, a, i, e, w, {us}, {rs}}.
    int idx_ravel = i * n_z_derivatives;
    for (int j = 0; j < n_z_derivatives; j++) {
      if (j < 6) {
        df_dz[idx_ravel + j] = df_dy[0] * dd_dx[j] + df_dy[1] * dnu_dx[j];
      } else {
        df_dz[idx_ravel + j] = df_dy[j - 4];
      }
    }
  }
}


template <typename T>
py::capsule encapsulate(T* fn) {
  // JAX callables must be wrapped in this py::capsule.
  return py::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}


py::dict jax_registrations() {
  // Dictionary of JAX callables.
  py::dict dict;
  dict["jax_light_curve_quad_ld"] = encapsulate(
    jax_light_curve_quad_ld);
  dict["jax_light_curve_nonlinear_ld"] = encapsulate(
    jax_light_curve_nonlinear_ld);
  return dict;
}


PYBIND11_MODULE(bindings, m) {

  m.def("orbit", &compute_orbit_trajectories,
    py::arg("t0") = py::none(),
    py::arg("period") = py::none(),
    py::arg("a") = py::none(),
    py::arg("inc") = py::none(),
    py::arg("ecc") = py::none(),
    py::arg("omega") = py::none(),
    py::arg("times") = py::none(),
    py::arg("ds") = py::none(),
    py::arg("zs") = py::none(),
    py::arg("nus") = py::none());

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
