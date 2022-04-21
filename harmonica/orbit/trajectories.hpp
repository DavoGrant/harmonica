#ifndef TRAJECTORIES_HPP
#define TRAJECTORIES_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


void orbital_trajectories(double i, double j,
                          bool require_gradients,
                          pybind11::array_t<double, pybind11::array::c_style> np_in,
                          pybind11::array_t<double, pybind11::array::c_style> np_out);

#endif
