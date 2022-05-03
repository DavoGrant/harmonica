#ifndef ORBIT_DERIVATIVES_HPP
#define ORBIT_DERIVATIVES_HPP


/**
 * Compute orbital derivatives, dd/dz and dnu/dz, for z in
 * the set {t0, p, a, i} for circular trajectories, e=0. Here,
 * d is the separation distance between planet and stellar centres
 * and nu is the angle between planet velocity and stellar centre.
 *
 * @param t0 time of transit centre [days].
 * @param period orbital period [days].
 * @param a semi-major axis [stellar radii].
 * @param sin_i sine of orbital inclination [].
 * @param cos_i cosine of orbital inclination [].
 * @param time of model evaluation [days].
 * @param n 2pi/period [1/days].
 * @param sin_M sine of mean anomaly [].
 * @param cos_M cosine of mean anomaly [].
 * @param x planet x position relative to stellar centre [stellar radii].
 * @param y planet y position relative to stellar centre [stellar radii].
 * @param atan_mcsM arctangent(-cos_M/sin_M) [radians].
 * @param d planet-star centre separation [stellar radii].
 * @param d_squared planet-star centre separation squared [stellar radii^2].
 * @param dd_dt0 derivative [stellar radii/days].
 * @param dd_dp derivative [stellar radii/days].
 * @param dd_da derivative [].
 * @param dd_dinc derivative [stellar radii/radians].
 * @param dnu_dt0 derivative [radians/days].
 * @param dnu_dp derivative [radians/days].
 * @param dnu_da derivative [radians/stellar radii].
 * @param dnu_dinc derivative [].
 * @return void.
 */
void orbital_derivatives_circular(const double &t0, const double &period,
                                  const double &a, const double &sin_i,
                                  const double &cos_i, const double &time,
                                  const double &n, const double &sin_M,
                                  const double &cos_M, const double &x,
                                  const double &y, const double &atan_mcsM,
                                  const double &d, const double &d_squared,
                                  double &dd_dt0, double &dd_dp,
                                  double &dd_da, double &dd_dinc,
                                  double &dnu_dt0, double &dnu_dp,
                                  double &dnu_da, double &dnu_dinc);


#endif
