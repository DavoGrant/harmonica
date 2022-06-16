#include <cmath>
#include <tuple>
#include <iostream>

#include "trajectories.hpp"
#include "kepler.hpp"
#include "../constants/constants.hpp"


OrbitTrajectories::OrbitTrajectories(double t0, double period, double a,
                                     double inc, double ecc, double omega,
                                     bool require_gradients) {

  // Orbital parameters.
  _t0 = t0;
  _period = period;
  _n = fractions::twopi / period;
  _a = a;
  _inc = inc;
  _sin_inc = std::sin(inc);
  _cos_inc = std::cos(inc);
  _ecc = ecc;
  if (ecc != 0.) {
    _omega = omega;
    _sin_omega = std::sin(omega);
    _cos_omega = std::cos(omega);
  }
  _require_gradients = require_gradients;
}


void OrbitTrajectories::compute_circular_orbit(
  const double &time, double &d, double &nu,
  double* dd_dz[], double* dnu_dz[]) {

  // Compute time of periastron.
  const double tp = _t0 - fractions::pi_d_2 / _n;

  // Compute mean anomaly.
  const double M = (time - tp) * _n;

  // Compute sine and cosine of the true anomaly.
  const double sin_M = std::sin(M);
  const double cos_M = std::cos(M);

  // Compute location of planet centre relative to stellar centre.
  const double x = _a * cos_M;
  const double y = _a *_cos_inc * sin_M;
  const double z = _a * _sin_inc * sin_M;
  if (z < 0.) {
    // Planet behind star, no transit, and no need for derivatives.
    d = intersections::behind;
    return;
  }

  // Compute angle between x-axis and planet velocity.
  const double atan_mcsM = std::atan(-cos_M / sin_M);
  const double psi =_cos_inc * atan_mcsM;

  // Compute separation distance between planet and stellar centres.
  const double d_squared = x * x + y * y;
  d = std::sqrt(d_squared);

  // Compute angle between planet velocity and stellar centre.
  nu = std::atan2(y, x) - psi;

  // Optionally compute derivatives.
  if (_require_gradients == true) {

    // Compute d partial derivatives: first branches of the tree.
    const double dd_dx = x / d;
    const double dd_dy = y / d;

    // Compute d partial derivatives: second branches of the tree.
    const double dx_da = cos_M;
    const double dx_dM = -_a * sin_M;
    const double dy_da = sin_M *_cos_inc;
    const double dy_dM = _a * cos_M *_cos_inc;
    const double dy_dinc = -_a * sin_M *_sin_inc;

    // Compute d partial derivatives: third branches of the tree.
    const double dM_dt0 = -_n;
    const double dM_dp = (_t0 - time) * fractions::twopi
                         / (_period * _period);

    // Compute nu partial derivatives: first branches of the tree.
    const double dnu_dx = -y / d_squared;
    const double dnu_dy = x / d_squared;
    const double dnu_dpsi = -1.;

    // Compute nu partial derivatives: second branches of the tree.
    const double dpsi_dM =_cos_inc;
    const double dpsi_dinc = -_sin_inc * atan_mcsM;

    // Compute dd_dt0, dd_dp, dd_da, and dd_dinc via the chain rule.
    *dd_dz[0] = dd_dx * dx_dM * dM_dt0 + dd_dy * dy_dM * dM_dt0;
    *dd_dz[1] = dd_dx * dx_dM * dM_dp + dd_dy * dy_dM * dM_dp;
    *dd_dz[2] = dd_dx * dx_da + dd_dy * dy_da;
    *dd_dz[3] = dd_dy * dy_dinc;

    // Compute dnu_dt0, dnu_dp, dnu_da, and dnu_dinc via the chain rule.
    *dnu_dz[0] = dnu_dx * dx_dM * dM_dt0 + dnu_dy * dy_dM * dM_dt0
                 + dnu_dpsi * dpsi_dM * dM_dt0;
    *dnu_dz[1] = dnu_dx * dx_dM * dM_dp + dnu_dy * dy_dM * dM_dp
                 + dnu_dpsi * dpsi_dM * dM_dp;
    *dnu_dz[2] = 0.;
    *dnu_dz[3] = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
  }
}


void OrbitTrajectories::compute_eccentric_orbit(
  const double &time, double &d, double &nu,
  double* dd_dz[], double* dnu_dz[]) {

  // Compute time of periastron.
  const double some = std::sqrt(1. - _ecc);
  const double sope = std::sqrt(1. + _ecc);
  const double E0 = 2. * std::atan2(some * _cos_omega,
                                    sope * (1 + _sin_omega));
  const double M0 = E0 - _ecc * std::sin(E0);
  const double tp = _t0 - M0 / _n;

  // Compute mean anomaly.
  const double M = (time - tp) * _n;

  // Compute sine and cosine of the true anomaly.
  std::tuple<double, double> sin_cos_f = solve_kepler(M, _ecc);
  const double sin_f = std::get<0>(sin_cos_f);
  const double cos_f = std::get<1>(sin_cos_f);

  // Compute location of planet centre relative to stellar centre.
  const double omes = 1. - _ecc * _ecc;
  const double ope_cosf = 1. + _ecc * std::get<1>(sin_cos_f);
  const double r = _a * omes / ope_cosf;
  const double sin_fpw = cos_f * _sin_omega + sin_f * _cos_omega;
  const double cos_fpw = cos_f * _cos_omega - sin_f * _sin_omega;
  const double x = r * cos_fpw;
  const double y = r * _cos_inc * sin_fpw;
  const double z = r * _sin_inc * sin_fpw;
  if (z < 0.) {
    // Planet behind star, no transit, and no need for derivatives.
    d = intersections::behind;
    return;
  }

  // Compute angle between x-axis and planet velocity.
  const double atan_mcs_fpw = std::atan(-cos_fpw / sin_fpw);
  const double psi =_cos_inc * atan_mcs_fpw;

  // Compute separation distance between planet and stellar centres.
  const double d_squared = x * x + y * y;
  d = std::sqrt(d_squared);

  // Compute angle between planet velocity and stellar centre.
  nu = std::atan2(y, x) - psi;

  // Optionally compute derivatives.
  if (_require_gradients == true) {

      // Compute d partial derivatives: first branches of the tree.
      const double dd_dx = x / d;
      const double dd_dy = y / d;

      // Compute d partial derivatives: second branches of the tree.
      const double dx_dr = cos_fpw;
      const double dx_dsinf = -r * _sin_omega;
      const double dx_dcosf = r * _cos_omega;
      const double dx_domega = -r * sin_fpw;
      const double dy_dr =_cos_inc * sin_fpw;
      const double dy_dsinf = r *_cos_inc * _cos_omega;
      const double dy_dcosf = r *_cos_inc * _sin_omega;
      const double dy_domega = r *_cos_inc * cos_fpw;
      const double dy_dinc = -r *_sin_inc * sin_fpw;

      // Compute d partial derivatives: third branches of the tree.
      const double ope_cosfs = ope_cosf * ope_cosf;
      const double omes_ptot = std::pow(omes, fractions::three_halves);
      const double dr_da = omes / ope_cosf;
      const double dr_de = -_a * (cos_f * (1. + _ecc * _ecc) + 2. * _ecc)
                           / ope_cosfs;
      const double dr_dcosf = -_a * _ecc * omes / ope_cosfs;
      const double dsinf_dM = cos_f * ope_cosfs / omes_ptot;
      const double dsinf_de = cos_f * sin_f * (1. + ope_cosf) / omes;
      const double dcosf_dM = -sin_f * ope_cosfs / omes_ptot;
      const double dcosf_de = -sin_f * sin_f * (1. + ope_cosf) / omes;

      // Compute d partial derivatives: fourth/fifth branches of the tree.
      const double alpha = some * _cos_omega / (sope * (1. + _sin_omega));
      const double toaspo = 2. / (alpha * alpha + 1);
      const double omecosE0_toaspo = (1. - _ecc * std::cos(E0)) * toaspo;
      const double dM_dt0 = -_n;
      const double dM_dp = (_t0 - time) * fractions::twopi
                   / (_period * _period);
      const double dM_de = -std::sin(E0) + omecosE0_toaspo * -_cos_omega
                           / ((1. + _sin_omega) * some
                           * std::pow(sope, 3.));
      const double dM_domega = omecosE0_toaspo * -some
                               / ((1. + _sin_omega) * sope);

      // Compute nu partial derivatives: first branches of the tree.
      const double dnu_dx = -y / d_squared;
      const double dnu_dy = x / d_squared;
      const double dnu_dpsi = -1.;

      // Compute nu partial derivatives: second branches of the tree.
      const double dpsi_dsinf =_cos_inc * cos_f;
      const double dpsi_dcosf = -_cos_inc * sin_f;
      const double dpsi_domega =_cos_inc;
      const double dpsi_dinc = -_sin_inc * atan_mcs_fpw;

      // Compute dd_dt0, dd_dp, dd_da, dd_dinc, dd_de, and dd_dw
      // via the chain rule. Probs autograd next time eh.
      *dd_dz[0] = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                           + dx_dsinf * dsinf_dM * dM_dt0
                           + dx_dcosf * dcosf_dM * dM_dt0)
                  + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                             + dy_dsinf * dsinf_dM * dM_dt0
                             + dy_dcosf * dcosf_dM * dM_dt0);
      *dd_dz[1] = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                           + dx_dsinf * dsinf_dM * dM_dp
                           + dx_dcosf * dcosf_dM * dM_dp)
                  + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                             + dy_dsinf * dsinf_dM * dM_dp
                             + dy_dcosf * dcosf_dM * dM_dp);
      *dd_dz[2] = dd_dx * dx_dr * dr_da + dd_dy * dy_dr * dr_da;
      *dd_dz[3] = dd_dy * dy_dinc;
      *dd_dz[4] = dd_dx * (dx_dr * (dr_de
                                    + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                           + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                           + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
                  + dd_dy * (dy_dr * (dr_de
                                      + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                             + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                             + dy_dcosf * (dcosf_de + dcosf_dM * dM_de));
      *dd_dz[5] = dd_dx * (dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                           + dx_dsinf * dsinf_dM * dM_domega
                           + dx_dcosf * dcosf_dM * dM_domega)
                  + dd_dy * (dy_domega
                             + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                             + dy_dsinf * dsinf_dM * dM_domega
                             + dy_dcosf * dcosf_dM * dM_domega);

      // Compute dnu_dt0, dnu_dp, dnu_da, dnu_dinc, dnu_de, and dnu_dw
      // via the chain rule.
      *dnu_dz[0] = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                             + dx_dsinf * dsinf_dM * dM_dt0
                             + dx_dcosf * dcosf_dM * dM_dt0)
                   + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                               + dy_dsinf * dsinf_dM * dM_dt0
                               + dy_dcosf * dcosf_dM * dM_dt0)
                   + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dt0
                                 + dpsi_dcosf * dcosf_dM * dM_dt0);
      *dnu_dz[1] = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                             + dx_dsinf * dsinf_dM * dM_dp
                             + dx_dcosf * dcosf_dM * dM_dp)
                   + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                               + dy_dsinf * dsinf_dM * dM_dp
                               + dy_dcosf * dcosf_dM * dM_dp)
                   + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dp
                                 + dpsi_dcosf * dcosf_dM * dM_dp);
      *dnu_dz[2] = dnu_dx * dx_dr * dr_da + dnu_dy * dy_dr * dr_da;
      *dnu_dz[3] = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
      *dnu_dz[4] = dnu_dx * (dx_dr * (dr_de
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
      *dnu_dz[5] = dnu_dx * (dx_domega
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
}
