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
    double sinf = 0.1;
    double cosf = 0.1;

    exoplanet::kepler::solve_kepler(M, e, &sinf, &cosf);

    std::cout << M << "\n";
    std::cout << e << "\n";
    std::cout << sinf << "\n";
    std::cout << cosf << "\n";

}
