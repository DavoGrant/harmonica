#ifndef BINDINGS_HPP
#define BINDINGS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * Compute orbital separation and angles for circular or eccentric
 * orbits of a planet-star system. The separation distance, d, is
 * between the planet and stellar centres and the angle, nu, is
 * between the planet's velocity and the stellar centre. Both
 * quantities are computed in the plane of the sky. Optionally, the
 * partial derivatives dd/dz and dnu/dz, for z in the
 * set {t0, p, a, i, e, w} may be computed.
 *
 * @param t0 time of transit centre [days].
 * @param period orbital period [days].
 * @param a semi-major axis [stellar radii].
 * @param inc orbital inclination [radians].
 * @param ecc eccentricity [].
 * @param omega argument of periastron [radians].
 * @param times array of model evaluation times [days].
 * @param ds empty array of planet-star centre separations [stellar radii].
 * @param nus empty array of planet velocity-star centre angles [radians].
 * @param ds_grad empty array of derivatives dd/dx x={t0, p, a, i, e, w}.
 * @param nus_grad empty array of derivatives dnu/dx x={t0, p, a, i, e, w}.
 * @param require_gradients derivatives switch.
 * @return void.
 */
void compute_orbital_separation_and_angles(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  py::array_t<double, py::array::c_style> times,
  py::array_t<double, py::array::c_style> ds,
  py::array_t<double, py::array::c_style> nus,
  py::array_t<double, py::array::c_style> ds_grad,
  py::array_t<double, py::array::c_style> nus_grad,
  bool require_gradients)


/**
 * Compute light curve for a given transmission string.
 *
 * ... WIP ...
 *
 * @param require_gradients derivatives switch.
 * @return void.
 */
void compute_orbital_separation_and_angles(
  bool require_gradients)


#endif
