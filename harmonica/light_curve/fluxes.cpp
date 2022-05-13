#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fluxes.hpp"
#include "../constants/constants.hpp"


Fluxes::Fluxes(int ld_law,
               py::array_t<double, py::array::c_style> us,
               py::array_t<double, py::array::c_style> rs,
               bool require_gradients) {

  if (ld_law == 0) {
    // Quadratic limb darkening law.
    B.resize(3, 3);
    B << -1., -1., -1.,
         0., 1., 2.,
         0., 0., -1.;
  } else if (ld_law == 1) {
    // Non-linear limb darkening law.
    B.resize(5, 5);
    B << -1., -1., -1., -1., -1.,
         0., 1., 0., 0., 0.,
         0., 0., 1., 0., 0.,
         0., 0., 0., 1., 0.,
         0., 0., 0., 0., 1.;
  }
}


void Fluxes::transit_light_curve() {

    std::cout << B << std::endl;

}
