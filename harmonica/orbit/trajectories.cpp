#include <cmath>

#include "trajectories.hpp"
#include "kepler.hpp"
#include "../constants/constants.hpp"


OrbitTrajectories::OrbitTrajectories(double t0, double period, double a,
                                     double inc, double ecc, double omega) {

  // Orbital parameters.
  m_t0 = t0;
  m_period = period;
  m_n = fractions::twopi / m_period;
  m_a = a;
  m_inc = inc;
  m_sin_inc = std::sin(inc);
  m_cos_inc = std::cos(inc);
  m_ecc = ecc;
  if (ecc == 0.) {
    m_omega = 0.;
    m_sin_omega = 0.;
    m_cos_omega = 1.;
  } else {
    m_omega = omega;
    m_sin_omega = std::sin(omega);
    m_cos_omega = std::cos(omega);
  }
}


void OrbitTrajectories::compute_circular_orbit(
  const double time, double& out_d, double& out_z, double& out_nu) {

  // Compute time of periastron.
  const double tp = m_t0 - fractions::pi_d_2 / m_n;

  // Compute mean anomaly.
  const double M = (time - tp) * m_n;

  // Compute sine and cosine of the true anomaly.
  m_sin_M = std::sin(M);
  m_cos_M = std::cos(M);

  // Compute location of planet centre relative to stellar centre.
  m_x = m_a * m_cos_M;
  m_y = m_a *m_cos_inc * m_sin_M;
  out_z = m_a * m_sin_inc * m_sin_M;

  // Compute angle between x-axis and planet velocity.
  m_atan_mcsM = std::atan(-m_cos_M / m_sin_M);
  const double psi =m_cos_inc * m_atan_mcsM;

  // Compute separation distance between planet and stellar centres.
  m_d_squared = m_x * m_x + m_y * m_y;
  out_d = std::sqrt(m_d_squared);

  // Compute angle between planet velocity and stellar centre.
  out_nu = std::atan2(m_y, m_x) - psi;
}


void OrbitTrajectories::compute_eccentric_orbit(
  const double time, double& out_d, double& out_z, double& out_nu) {

  // Compute time of periastron.
  m_some = std::sqrt(1. - m_ecc);
  m_sope = std::sqrt(1. + m_ecc);
  m_E0 = 2. * std::atan2(m_some * m_cos_omega, m_sope * (1 + m_sin_omega));
  const double M0 = m_E0 - m_ecc * std::sin(m_E0);
  const double tp = m_t0 - M0 / m_n;

  // Compute mean anomaly.
  const double M = (time - tp) * m_n;

  // Compute sine and cosine of the true anomaly.
  TrueAnomaly true_anom = solve_kepler(M, m_ecc);
  m_sin_f = true_anom.sinf;
  m_cos_f = true_anom.cosf;

  // Compute location of planet centre relative to stellar centre.
  m_omes = 1. - m_ecc * m_ecc;
  m_ope_cosf = 1. + m_ecc * m_cos_f;
  m_r = m_a * m_omes / m_ope_cosf;
  m_sin_fpw = m_cos_f * m_sin_omega + m_sin_f * m_cos_omega;
  m_cos_fpw = m_cos_f * m_cos_omega - m_sin_f * m_sin_omega;
  m_x = m_r * m_cos_fpw;
  m_y = m_r * m_cos_inc * m_sin_fpw;
  out_z = m_r * m_sin_inc * m_sin_fpw;

  // Compute angle between x-axis and planet velocity.
  m_atan_mcs_fpw = std::atan(-m_cos_fpw / m_sin_fpw);
  const double psi = m_cos_inc * m_atan_mcs_fpw;

  // Compute separation distance between planet and stellar centres.
  m_d_squared = m_x * m_x + m_y * m_y;
  out_d = std::sqrt(m_d_squared);

  // Compute angle between planet velocity and stellar centre.
  out_nu = std::atan2(m_y, m_x) - psi;
}
