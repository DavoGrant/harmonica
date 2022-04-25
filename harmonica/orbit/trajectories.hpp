#ifndef TRAJECTORIES_HPP
#define TRAJECTORIES_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * Compute orbital trajectories, and optionally derivatives, for
 * a given set of orbital parameters and evaluation times.
 *
 * @param t0 time of transit centre [days].
 * @param period orbital period [days].
 * @param a semi-major axis [stellar radii].
 * @param inc orbital inclination [radians].
 * @param b impact parameter [stellar radii].
 * @param ecc eccentricity [].
 * @param omega argument of periastron [radians].
 * @param times array of model evaluation times [days].
 * @param ds empty array of planet-star centre separations [stellar radii].
 * @param nus empty array of planet velocity-star centre angles [radians].
 * @param grad empty array of derivatives [tbd].
 * @return void.
 */
void orbital_trajectories(double t0, double period, double a,
                          double inc, double ecc, double omega,
                          py::array_t<double, py::array::c_style> times,
                          py::array_t<double, py::array::c_style> ds,
                          py::array_t<double, py::array::c_style> nus,
                          py::array_t<double, py::array::c_style> ds_grad,
                          py::array_t<double, py::array::c_style> nus_grad,
                          bool require_gradients);


#endif
