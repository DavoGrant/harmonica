#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>

#include "fluxes.hpp"
#include "../constants/constants.hpp"


Fluxes::Fluxes(int ld_law,
               py::array_t<double, py::array::c_style> us,
               py::array_t<double, py::array::c_style> rs,
               py::array_t<double, py::array::c_style> ds,
               py::array_t<double, py::array::c_style> nus,
               py::array_t<double, py::array::c_style> fs,
               py::array_t<double, py::array::c_style> ds_grad,
               py::array_t<double, py::array::c_style> nus_grad,
               py::array_t<double, py::array::c_style> fs_grad,
               bool require_gradients) {

  Eigen::Matrix<double, 3, 3> B_1 {{-1., -1., -1.},
                                   {0., 1., 2.},
                                   {0., 0., -1.}};

  Eigen::Matrix<double, 5, 5> B_2 {{-1., -1., -1., -1., -1.},
                                   {0., 1., 0., 0., 0.},
                                   {0., 0., 1., 0., 0.},
                                   {0., 0., 0., 1., 0.},
                                   {0., 0., 0., 0., 1.}};

  std::cout << "bool:\n" << require_gradients << std::endl;
  std::cout << "Here is B_1:\n" << B_1 << std::endl;
  std::cout << "Here is B_2:\n" << B_2 << std::endl;

}
