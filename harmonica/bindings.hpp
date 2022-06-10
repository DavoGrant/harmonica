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
 * @param ds_grad empty derivatives array dd/dz z={t0, p, a, i, e, w}.
 * @param nus_grad empty derivatives array dnu/dz z={t0, p, a, i, e, w}.
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
  bool require_gradients);


/**
 * Compute a normalised transit light curve for a given transmission string,
 * defined by an array of harmonic coefficients rs, and stellar limb
 * darkening law, defined by an array of coefficients us.
 *
 * @param ld_law limb darkening law, 0=quadratic, 1=non-linear.
 * @param us array of stellar limb darkening coefficients [].
 * @param rs array of planet radius harmonic coefficients [stellar radii].
 * @param ds array of planet-star centre separations [stellar radii].
 * @param nus array of planet velocity-star centre angles [radians].
 * @param fs empty array of normalised light curve fluxes [].
 * @param ds_grad (empty) derivatives array dd/dz z={t0, p, a, i, e, w}.
 * @param nus_grad (empty) derivatives array dnu/dz z={t0, p, a, i, e, w}.
 * @param fs_grad empty derivatives array dfs/dy y={t0, p, a, i, e, w,
                                                    {us}, {rs}}.
 * @param pnl_c N_l precision for planet inside stellar disc.
 * @param pnl_e N_l precision for planet intersecting stellar disc.
 * @param require_gradients derivatives switch.
 * @return void.
 */
void compute_harmonica_light_curve(
  int ld_law,
  py::array_t<double, py::array::c_style> us,
  py::array_t<double, py::array::c_style> rs,
  py::array_t<double, py::array::c_style> ds,
  py::array_t<double, py::array::c_style> nus,
  py::array_t<double, py::array::c_style> fs,
  py::array_t<double, py::array::c_style> ds_grad,
  py::array_t<double, py::array::c_style> nus_grad,
  py::array_t<double, py::array::c_style> fs_grad,
  int pnl_c, int pnl_e, bool require_gradients);


/**
 * Compute the transmission string defined by an array of harmonic
 * coefficients, rs, around an array of angles, thetas, in the terminator
 * plane.
 *
 * @param rs array of planet radius harmonic coefficients [stellar radii].
 * @param thetas array of angles in the terminator plane [radians].
 * @param transmission_string empty array of planet radii [stellar radii].
 * @return void.
 */
void compute_transmission_string(
  py::array_t<double, py::array::c_style> rs,
  py::array_t<double, py::array::c_style> thetas,
  py::array_t<double, py::array::c_style> transmission_string);


#endif
