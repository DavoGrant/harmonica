#include <cmath>
#include <tuple>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "trajectories.hpp"
#include "kepler.hpp"
#include "derivatives/orbit_derivatives.hpp"
#include "../constants/constants.hpp"

namespace py = pybind11;


void orbital_trajectories(double t0, double period, double a,
                          double inc, double ecc, double omega,
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

  const double n = fractions::twopi / period;
  const double sin_i = std::sin(inc);
  const double cos_i = std::cos(inc);

  if (ecc == 0.) {

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      const double tp = t0 - fractions::pi_d_2 / n;

      // Compute mean anomaly.
      const double M = (times_(i) - tp) * n;

      // Compute sine and cosine of the true anomaly.
      const double sin_M = std::sin(M);
      const double cos_M = std::cos(M);

      // Compute location of planet centre relative to stellar centre.
      const double x = a * cos_M;
      const double y = a * cos_i * sin_M;

      // Compute angle between x-axis and planet velocity.
      const double atan_mcsM = std::atan(-cos_M / sin_M);
      const double psi = cos_i * atan_mcsM;

      // Compute separation distance between planet and stellar centres.
      const double d_squared = x * x + y * y;
      ds_(i) = std::sqrt(d_squared);

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      if (require_gradients == true) {
        // Optionally compute derivatives.
        orbital_derivatives_circular(t0, period, a, sin_i, cos_i, times_(i),
                                     n, sin_M, cos_M, x, y, atan_mcsM,
                                     ds_(i), d_squared,
                                     ds_grad_(i, 0), ds_grad_(i, 1),
                                     ds_grad_(i, 2), ds_grad_(i, 3),
                                     nus_grad_(i, 0), nus_grad_(i, 1),
                                     nus_grad_(i, 2), nus_grad_(i, 3));
      }
    }

  } else {

    const double sin_omega = std::sin(omega);
    const double cos_omega = std::cos(omega);

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      const double some = std::sqrt(1. - ecc);
      const double sope = std::sqrt(1. + ecc);
      const double E0 = 2. * std::atan2(some * cos_omega,
                                        sope * (1 + sin_omega));
      const double M0 = E0 - ecc * std::sin(E0);
      const double tp = t0 - M0 / n;

      // Compute mean anomaly.
      const double M = (times_(i) - tp) * n;

      // Compute sine and cosine of the true anomaly.
      std::tuple<double, double> sin_cos_f = solve_kepler(M, ecc);

      // Compute location of planet centre relative to stellar centre.
      const double omes = 1. - ecc * ecc;
      const double ope_cosf = 1. + ecc * std::get<1>(sin_cos_f);
      const double r = a * omes / ope_cosf;
      const double sin_fpw = std::get<1>(sin_cos_f) * sin_omega
                             + std::get<0>(sin_cos_f) * cos_omega;
      const double cos_fpw = std::get<1>(sin_cos_f) * cos_omega
                             - std::get<0>(sin_cos_f) * sin_omega;
      const double x = r * cos_fpw;
      const double y = r * cos_i * sin_fpw;

      // Compute angle between x-axis and planet velocity.
      const double atan_mcs_fpw = std::atan(-cos_fpw / sin_fpw);
      const double psi = cos_i * atan_mcs_fpw;

      // Compute separation distance between planet and stellar centres.
      const double d_squared = x * x + y * y;
      ds_(i) = std::sqrt(d_squared);

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      if (require_gradients == true) {
        // Optionally compute derivatives.
        orbital_derivatives(t0, period, a, sin_i, cos_i, ecc, sin_omega,
                            cos_omega, times_(i), E0, n,
                            std::get<0>(sin_cos_f), std::get<1>(sin_cos_f),
                            x, y, sin_fpw, cos_fpw, atan_mcs_fpw, r, some,
                            sope, omes, ope_cosf, ds_(i), d_squared,
                            ds_grad_(i, 0), ds_grad_(i, 1), ds_grad_(i, 2),
                            ds_grad_(i, 3), ds_grad_(i, 4), ds_grad_(i, 5),
                            nus_grad_(i, 0), nus_grad_(i, 1), nus_grad_(i, 2),
                            nus_grad_(i, 3), nus_grad_(i, 4), nus_grad_(i, 5));
      }
    }
  }
}
