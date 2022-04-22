#ifndef KEPLER_HPP
#define KEPLER_HPP

#include <tuple>


/**
 * Solve Kepler's equation for the sine and cosine of the true
 * anomaly, via the eccentric anomaly.
 *
 * @param M mean anomaly.
 * @param ecc eccentricity.
 * @return sine and cosine of the true anomaly.
 */
std::tuple<double, double> solve_kepler(const double M, const double ecc);


/**
 * Eccentric anomaly guess to Kepler's equation when in
 * the singular corner of parameter space.
 *
 * @param M mean anomaly, close to zero.
 * @param ecc eccentricity, close to one.
 * @return eccentric anomaly guess.
 */
double eccentric_anomaly_guess(const double M, const double ecc);


/**
 * Compute the sine and cosine of the eccentric anomaly. The method
 * follows Raposo-Pulido+ (2017) and the implementation is adapted
 * from exoplanet-core (Dan Foreman-Mackey), originally written by
 * Timothy Brandt. Accuracy of ~1e-15 in E-ecc*sin(E)-M.
 *
 * @param M mean anomaly.
 * @param ecc eccentricity.
 * @return sine and cosine of the eccentric anomaly.
 */
std::tuple<double, double> compute_eccentric_anomaly(
  const double M, const double ecc);


/**
 * Compute the sine and cosine of the true anomaly.
 *
 * @param ecc eccentricity.
 * @param sine and cosine of the eccentric anomaly.
 * @return sine and cosine of the true anomaly.
 */
std::tuple<double, double> compute_true_anomaly(
  const double ecc, std::tuple<double, double> sin_cos_ea);


#endif
