#include <cmath>
#include <tuple>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "trajectories.hpp"
#include "kepler.hpp"
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
  if (require_gradients == true) {
    auto ds_grad_ = ds_grad.mutable_unchecked<2>();
    auto nus_grad_ = nus_grad.mutable_unchecked<2>();
  }

  const double n = fractions::twopi / period;
  const double cos_i = std::cos(inc);

  if (ecc == 0.) {

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      double tp = t0 - fractions::pi_d_2 / n;

      // Compute mean anomaly.
      double M = (times_(i) - tp) * n;

      // Compute sine and cosine of the true anomaly.
      std::tuple<double, double> sin_cos_f = std::make_tuple(
          std::sin(M), std::cos(M));

      // Compute location of planet centre relative to stellar centre.
      double x = a * std::get<1>(sin_cos_f);
      double y = a * cos_i * std::get<0>(sin_cos_f);

      // Compute angle between x-axis and planet velocity.
      double psi = cos_i * std::atan(-std::get<1>(sin_cos_f)
                                     / std::get<0>(sin_cos_f));

      // Compute separation distance between planet and stellar centres.
      ds_(i) = std::sqrt(x * x + y * y);

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      // Optionally compute derivatives.

    }

  } else {

    const double sin_w = std::sin(omega);
    const double cos_w = std::cos(omega);

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      double E0 = 2. * std::atan2(std::sqrt(1. - ecc) * cos_w,
                                  std::sqrt(1. + ecc) * (1 + sin_w));
      double M0 = E0 - ecc * std::sin(E0);
      double tp = t0 - M0 / n;

      // Compute mean anomaly.
      double M = (times_(i) - tp) * n;

      // Compute sine and cosine of the true anomaly.
      std::tuple<double, double> sin_cos_f = solve_kepler(M, ecc);

      // Compute location of planet centre relative to stellar centre.
      double r = a * (1 - ecc * ecc) / (1 + ecc * std::get<1>(sin_cos_f));
      const double sin_fpw = std::get<1>(sin_cos_f) * sin_w
                             + std::get<0>(sin_cos_f) * cos_w;
      const double cos_fpw = std::get<1>(sin_cos_f) * cos_w
                             - std::get<0>(sin_cos_f) * sin_w;
      double x = r * cos_fpw;
      double y = r * cos_i * sin_fpw;

      // Compute angle between x-axis and planet velocity.
      double psi = cos_i * std::atan(-cos_fpw / sin_fpw);

      // Compute separation distance between planet and stellar centres.
      ds_(i) = std::sqrt(x * x + y * y);

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      // Optionally compute derivatives.

    }
  }
}
