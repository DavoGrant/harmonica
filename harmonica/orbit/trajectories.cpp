#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "trajectories.hpp"
#include "kepler.hpp"


void orbital_trajectories(double i, double j,
                          bool require_gradients,
                          pybind11::array_t<double, pybind11::array::c_style> np_in,
                          pybind11::array_t<double, pybind11::array::c_style> np_out) {

    auto r1 = np_in.unchecked<1>();
    auto r2 = np_out.mutable_unchecked<1>();
    r2(1) = 1000.;
    std::cout << r1(1) << "\n";
    std::cout << r2(1) << "\n";

    const double M = i;
    const double e = j;

    std::tuple<double, double> sin_cos_ta;
    for (i = 0; i < 1000000; ++i) {
        sin_cos_ta = solve_kepler(M, e);
    }

    std::cout << M << "\n";
    std::cout << e << "\n";
    std::cout << std::get<0>(sin_cos_ta) << "\n";
    std::cout << std::get<1>(sin_cos_ta) << "\n";

}
