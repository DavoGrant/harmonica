#ifndef LIGHT_CURVE_HPP
#define LIGHT_CURVE_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

void light_curve(int i, int j, bool require_gradients,
                 pybind11::array_t<double, pybind11::array::c_style> np_in,
                 pybind11::array_t<double, pybind11::array::c_style> np_out);

#endif
