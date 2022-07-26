#ifndef BINDINGS_HPP
#define BINDINGS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * Compute orbit trajectories for a binary system.
 *
 * @param t0 time of transit centre [days].
 * @param period orbital period [days].
 * @param a semi-major axis [stellar radii].
 * @param inc orbital inclination [radians].
 * @param ecc eccentricity [].
 * @param omega argument of periastron [radians].
 * @param times_py array of model evaluation times [days].
 * @param out_ds_py planet-star centre separations [stellar radii].
 * @param out_zs_py distances from sky-plane, z < 0 planet behind [stellar radii].
 * @param out_nus_py planet velocity-star centre angles [radians].
 * @return void.
 */
void compute_orbit_trajectories(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_ds_py,
  py::array_t<double, py::array::c_style> out_zs_py,
  py::array_t<double, py::array::c_style> out_nus_py);


/**
 * Compute a normalised transit light curve for a given transmission
 * string, defined by an array of harmonic coefficients rs, and stellar
 * limb darkening law, defined by an array of coefficients us.
 *
 * @param t0 time of transit centre [days].
 * @param period orbital period [days].
 * @param a semi-major axis [stellar radii].
 * @param inc orbital inclination [radians].
 * @param ecc eccentricity [].
 * @param omega argument of periastron [radians].
 * @param ld_law limb darkening law, 0=quadratic, 1=non-linear.
 * @param us_py array of stellar limb darkening coefficients [].
 * @param rs_py array of planet radius harmonic coefficients [stellar radii].
 * @param times_py array of model evaluation times [days].
 * @param out_fs_py array of normalised light curve fluxes [].
 * @param pnl_c N_l precision for planet inside stellar disc.
 * @param pnl_e N_l precision for planet intersecting stellar disc.
 * @return void.
 */
void compute_harmonica_light_curve(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  int ld_law,
  py::array_t<double, py::array::c_style> us_py,
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_fs_py,
  int pnl_c, int pnl_e);


/**
 * Compute the transmission string defined by an array of harmonic
 * coefficients, rs, around an array of angles, thetas, in the terminator
 * plane.
 *
 * @param rs_py array of planet radius harmonic coefficients [stellar radii].
 * @param thetas_py array of angles in the terminator plane [radians].
 * @param out_transmission_string_py array of planet radii [stellar radii].
 * @return void.
 */
void compute_transmission_string(
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> thetas_py,
  py::array_t<double, py::array::c_style> transmission_string_py);


/**
 * JAX custom XLA call. Transit light curve computation and derivatives
 * for a quadratically limb-darkened star. This code follows the structure
 * required for creating custom calls. The derivatives computed are df/dz
 * where z={t0, p, a, i, e, w, {us}, {rs}}.
 *
 * @param out_tuple tuple(f, df_dz) where f is an array allocated for
 *        the light curve flux values, and df_dz is a ravelled array
 *        allocated for the flux derivatives. This pre-ravelled array
 *        has dimensions [..., 6 + n_us + n_rs] where we have 6 orbital,
 *        n_us=2 limb-darkening, and n_rs transmission string parameters.
 * @param in list(t0, p, a, i, e, w, u1, u2, r0, r1, r2,..) where the
 *        number of r params is not known until runtime. NB. these params
 *        are arrays, with values at each time, but as they are const
 *        through time we only require first index. This may change in
 *        future releases.
 * @return void.
 */
const void jax_light_curve_quad_ld(void* out_tuple, const void** in);


/**
 * JAX custom XLA call. Transit light curve computation and derivatives
 * for a non-linear limb-darkened star. This code follows the structure
 * required for creating custom calls. The derivatives computed are df/dz
 * where z={t0, p, a, i, e, w, {us}, {rs}}.
 *
 * @param out_tuple tuple(f, df_dz) where f is an array allocated for
 *        the light curve flux values, and df_dz is a ravelled array
 *        allocated for the flux derivatives. This pre-ravelled array
 *        has dimensions [..., 6 + n_us + n_rs] where we have 6 orbital,
 *        n_us=4 limb-darkening, and n_rs transmission string parameters.
 * @param in list(t0, p, a, i, e, w, u1, u2, u3, u4, r0, r1, r2,..) where
 *        the number of r params is not known until runtime. NB. these
 *        params are arrays, with values at each time, but as they are
 *        const through time we only require first index. This may change
 *        in future releases.
 * @return void.
 */
const void jax_light_curve_nonlinear_ld(void* out_tuple, const void** in);


#endif
