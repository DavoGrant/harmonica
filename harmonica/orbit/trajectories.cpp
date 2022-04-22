#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "trajectories.hpp"
#include "kepler.hpp"


void orbital_trajectories(
  double t0, double period, double a, double inc, double ecc, double omega,
  pybind11::array_t<double, pybind11::array::c_style> times,
  pybind11::array_t<double, pybind11::array::c_style> ds,
  pybind11::array_t<double, pybind11::array::c_style> nus,
  pybind11::array_t<double, pybind11::array::c_style> ds_grad,
  pybind11::array_t<double, pybind11::array::c_style> nus_grad,
  bool require_gradients) {

  // Unpack numpy arrays.
  auto times_ = times.unchecked<1>();
  auto ds_ = ds.mutable_unchecked<1>();
  auto nus_ = nus.mutable_unchecked<1>();
  if (require_gradients == true) {
    auto ds_grad_ = ds_grad.mutable_unchecked<2>();
    auto nus_grad_ = nus_grad.mutable_unchecked<2>();
  }

  // todo: cache certain algebra
  // todo: ecc==0. simplify.

  // Compute time of periastron.

  // Iterate evaluation times.

      // Compute mean anomaly.

      // Compute true anomaly.

      // Compute separation of planet centre from stellar centre.

      // Compute angle between planet velocity and stellar centre.

      // Compute derivatives.





//  auto times_uc = times.unchecked<1>();
//  auto ds_uc = ds.mutable_unchecked<1>();
//  ds_uc(1) = 1000.;
//  std::cout << times_uc(1) << "\n";
//  std::cout << ds_uc(1) << "\n";
//
//  const double M = i;
//  const double e = j;
//
//  std::tuple<double, double> sin_cos_ta;
//  for (i = 0; i < 1000000; ++i) {
//      sin_cos_ta = solve_kepler(M, e);
//  }
//
//  std::cout << M << "\n";
//  std::cout << e << "\n";
//  std::cout << std::get<0>(sin_cos_ta) << "\n";
//  std::cout << std::get<1>(sin_cos_ta) << "\n";

}
