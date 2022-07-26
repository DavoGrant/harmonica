#include <cmath>

#include "gradients.hpp"
#include "kepler.hpp"
#include "../constants/constants.hpp"


OrbitDerivatives::OrbitDerivatives(double t0, double period, double a,
                                   double inc, double ecc, double omega)
    : OrbitTrajectories(t0, period, a, inc, ecc, omega) {}


void OrbitDerivatives::compute_circular_orbit_and_derivatives(
  const double time, double& out_d, double& out_z, double& out_nu,
  double out_dd_dz[], double out_dnu_dz[]) {

  this->compute_circular_orbit(time, out_d, out_z, out_nu);

  // Compute d partial derivatives: first branches of the tree.
  const double dd_dx = m_x / out_d;
  const double dd_dy = m_y / out_d;

  // Compute d partial derivatives: second branches of the tree.
  const double dx_da = m_cos_M;
  const double dx_dM = -m_a* m_sin_M;
  const double dy_da = m_sin_M * m_cos_inc;
  const double dy_dM = m_a* m_cos_M * m_cos_inc;
  const double dy_dinc = -m_a* m_sin_M * m_sin_inc;

  // Compute d partial derivatives: third branches of the tree.
  const double dM_dt0 = -m_n;
  const double dM_dp = (m_t0 - time) * fractions::twopi
                       / (m_period * m_period);

  // Compute nu partial derivatives: first branches of the tree.
  const double dnu_dx = -m_y / m_d_squared;
  const double dnu_dy = m_x / m_d_squared;
  const double dnu_dpsi = -1.;

  // Compute nu partial derivatives: second branches of the tree.
  const double dpsi_dM = m_cos_inc;
  const double dpsi_dinc = -m_sin_inc * m_atan_mcsM;

  // Compute dd_dt0, dd_dp, dd_da, and dd_dinc via the chain rule.
  out_dd_dz[0] = dd_dx * dx_dM * dM_dt0 + dd_dy * dy_dM * dM_dt0;
  out_dd_dz[1] = dd_dx * dx_dM * dM_dp + dd_dy * dy_dM * dM_dp;
  out_dd_dz[2] = dd_dx * dx_da + dd_dy * dy_da;
  out_dd_dz[3] = dd_dy * dy_dinc;

  // Compute dnu_dt0, dnu_dp, dnu_da, and dnu_dinc via the chain rule.
  out_dnu_dz[0] = dnu_dx * dx_dM * dM_dt0 + dnu_dy * dy_dM * dM_dt0
                  + dnu_dpsi * dpsi_dM * dM_dt0;
  out_dnu_dz[1] = dnu_dx * dx_dM * dM_dp + dnu_dy * dy_dM * dM_dp
                  + dnu_dpsi * dpsi_dM * dM_dp;
  out_dnu_dz[2] = 0.;
  out_dnu_dz[3] = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
}


void OrbitDerivatives::compute_eccentric_orbit_and_derivatives(
  const double time, double& out_d, double& out_z, double& out_nu,
  double out_dd_dz[], double out_dnu_dz[]) {

  this->compute_eccentric_orbit(time, out_d, out_z, out_nu);

  // Compute d partial derivatives: first branches of the tree.
  const double dd_dx = m_x / out_d;
  const double dd_dy = m_y / out_d;

  // Compute d partial derivatives: second branches of the tree.
  const double dx_dr = m_cos_fpw;
  const double dx_dsinf = -m_r * m_sin_omega;
  const double dx_dcosf = m_r * m_cos_omega;
  const double dx_domega = -m_r * m_sin_fpw;
  const double dy_dr = m_cos_inc * m_sin_fpw;
  const double dy_dsinf = m_r * m_cos_inc * m_cos_omega;
  const double dy_dcosf = m_r * m_cos_inc * m_sin_omega;
  const double dy_domega = m_r * m_cos_inc * m_cos_fpw;
  const double dy_dinc = -m_r *m_sin_inc * m_sin_fpw;

  // Compute d partial derivatives: third branches of the tree.
  const double ope_cosfs = m_ope_cosf * m_ope_cosf;
  const double omes_ptot = std::pow(m_omes, fractions::three_halves);
  const double dr_da = m_omes / m_ope_cosf;
  const double dr_de = -m_a* (m_cos_f * (1. + m_ecc * m_ecc) + 2. * m_ecc)
                       / ope_cosfs;
  const double dr_dcosf = -m_a* m_ecc * m_omes / ope_cosfs;
  const double dsinf_dM = m_cos_f * ope_cosfs / omes_ptot;
  const double dsinf_de = m_cos_f * m_sin_f * (1. + m_ope_cosf) / m_omes;
  const double dcosf_dM = -m_sin_f * ope_cosfs / omes_ptot;
  const double dcosf_de = -m_sin_f * m_sin_f * (1. + m_ope_cosf) / m_omes;

  // Compute d partial derivatives: fourth/fifth branches of the tree.
  const double alpha = m_some * m_cos_omega / (m_sope * (1. + m_sin_omega));
  const double toaspo = 2. / (alpha * alpha + 1);
  const double omecosE0_toaspo = (1. - m_ecc * std::cos(m_E0)) * toaspo;
  const double dM_dt0 = -m_n;
  const double dM_dp = (m_t0 - time) * fractions::twopi
               / (m_period * m_period);
  const double dM_de = -std::sin(m_E0) + omecosE0_toaspo * -m_cos_omega
                       / ((1. + m_sin_omega) * m_some
                       * std::pow(m_sope, 3.));
  const double dM_domega = omecosE0_toaspo * -m_some
                           / ((1. + m_sin_omega) * m_sope);

  // Compute nu partial derivatives: first branches of the tree.
  const double dnu_dx = -m_y / m_d_squared;
  const double dnu_dy = m_x / m_d_squared;
  const double dnu_dpsi = -1.;

  // Compute nu partial derivatives: second branches of the tree.
  const double dpsi_dsinf = m_cos_inc * m_cos_f;
  const double dpsi_dcosf = -m_cos_inc * m_sin_f;
  const double dpsi_domega = m_cos_inc;
  const double dpsi_dinc = -m_sin_inc * m_atan_mcs_fpw;

  // Compute dd_dt0, dd_dp, dd_da, dd_dinc, dd_de, and dd_dw
  // via the chain rule. Probs autograd next time eh.
  out_dd_dz[0] = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                      + dx_dsinf * dsinf_dM * dM_dt0
                      + dx_dcosf * dcosf_dM * dM_dt0)
             + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                        + dy_dsinf * dsinf_dM * dM_dt0
                        + dy_dcosf * dcosf_dM * dM_dt0);
  out_dd_dz[1] = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                      + dx_dsinf * dsinf_dM * dM_dp
                      + dx_dcosf * dcosf_dM * dM_dp)
             + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                        + dy_dsinf * dsinf_dM * dM_dp
                        + dy_dcosf * dcosf_dM * dM_dp);
  out_dd_dz[2] = dd_dx * dx_dr * dr_da + dd_dy * dy_dr * dr_da;
  out_dd_dz[3] = dd_dy * dy_dinc;
  out_dd_dz[4] = dd_dx * (dx_dr * (dr_de
                               + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                      + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                      + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
             + dd_dy * (dy_dr * (dr_de
                                 + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                        + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                        + dy_dcosf * (dcosf_de + dcosf_dM * dM_de));
  out_dd_dz[5] = dd_dx * (dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                      + dx_dsinf * dsinf_dM * dM_domega
                      + dx_dcosf * dcosf_dM * dM_domega)
             + dd_dy * (dy_domega
                        + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                        + dy_dsinf * dsinf_dM * dM_domega
                        + dy_dcosf * dcosf_dM * dM_domega);

  // Compute dnu_dt0, dnu_dp, dnu_da, dnu_dinc, dnu_de, and dnu_dw
  // via the chain rule.
  out_dnu_dz[0] = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                        + dx_dsinf * dsinf_dM * dM_dt0
                        + dx_dcosf * dcosf_dM * dM_dt0)
              + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                          + dy_dsinf * dsinf_dM * dM_dt0
                          + dy_dcosf * dcosf_dM * dM_dt0)
              + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dt0
                            + dpsi_dcosf * dcosf_dM * dM_dt0);
  out_dnu_dz[1] = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                        + dx_dsinf * dsinf_dM * dM_dp
                        + dx_dcosf * dcosf_dM * dM_dp)
              + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                          + dy_dsinf * dsinf_dM * dM_dp
                          + dy_dcosf * dcosf_dM * dM_dp)
              + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dp
                            + dpsi_dcosf * dcosf_dM * dM_dp);
  out_dnu_dz[2] = dnu_dx * dx_dr * dr_da + dnu_dy * dy_dr * dr_da;
  out_dnu_dz[3] = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
  out_dnu_dz[4] = dnu_dx * (dx_dr * (dr_de
                                 + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                        + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                        + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
              + dnu_dy * (dy_dr * (dr_de
                                   + dr_dcosf * (dcosf_de
                                                 + dcosf_dM * dM_de))
                          + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                          + dy_dcosf * (dcosf_de + dcosf_dM * dM_de))
              + dnu_dpsi * (dpsi_dsinf * (dsinf_de + dsinf_dM * dM_de)
                            + dpsi_dcosf * (dcosf_de + dcosf_dM * dM_de));
  out_dnu_dz[5] = dnu_dx * (dx_domega
                        + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                        + dx_dsinf * dsinf_dM * dM_domega
                        + dx_dcosf * dcosf_dM * dM_domega)
              + dnu_dy * (dy_domega
                          + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                          + dy_dsinf * dsinf_dM * dM_domega
                          + dy_dcosf * dcosf_dM * dM_domega)
              + dnu_dpsi * (dpsi_domega
                            + dpsi_dsinf * dsinf_dM * dM_domega
                            + dpsi_dcosf * dcosf_dM * dM_domega);
}
