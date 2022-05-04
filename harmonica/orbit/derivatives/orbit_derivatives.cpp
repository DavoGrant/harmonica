#include <cmath>

#include "orbit_derivatives.hpp"
#include "../../constants/constants.hpp"


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
                                  double &dnu_da, double &dnu_dinc) {

  // Compute d partial derivatives: first branches of the tree.
  const double dd_dx = x / d;
  const double dd_dy = y / d;

  // Compute d partial derivatives: second branches of the tree.
  const double dx_da = cos_M;
  const double dx_dM = -a * sin_M;
  const double dy_da = sin_M * cos_i;
  const double dy_dM = a * cos_M * cos_i;
  const double dy_dinc = -a * sin_M * sin_i;

  // Compute d partial derivatives: third branches of the tree.
  const double dM_dt0 = -n;
  const double dM_dp = (t0 - time) * fractions::twopi / (period * period);

  // Compute nu partial derivatives: first branches of the tree.
  const double dnu_dx = -y / d_squared;
  const double dnu_dy = x / d_squared;
  const double dnu_dpsi = -1.;

  // Compute nu partial derivatives: second branches of the tree.
  const double dpsi_dM = cos_i;
  const double dpsi_dinc = -sin_i * atan_mcsM;

  // Compute dd_dt0, dd_dp, dd_da, and dd_dinc via the chain rule.
  dd_dt0 = dd_dx * dx_dM * dM_dt0 + dd_dy * dy_dM * dM_dt0;
  dd_dp = dd_dx * dx_dM * dM_dp + dd_dy * dy_dM * dM_dp;
  dd_da = dd_dx * dx_da + dd_dy * dy_da;
  dd_dinc = dd_dy * dy_dinc;

  // Compute dnu_dt0, dnu_dp, dnu_da, and dnu_dinc via the chain rule.
  dnu_dt0 = dnu_dx * dx_dM * dM_dt0 + dnu_dy * dy_dM * dM_dt0
            + dnu_dpsi * dpsi_dM * dM_dt0;
  dnu_dp = dnu_dx * dx_dM * dM_dp + dnu_dy * dy_dM * dM_dp
           + dnu_dpsi * dpsi_dM * dM_dp;
  dnu_da = 0.;
  dnu_dinc = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
}


void orbital_derivatives(const double &t0, const double &period,
                         const double &a, const double &sin_i,
                         const double &cos_i, const double &ecc,
                         const double &sin_omega, const double &cos_omega,
                         const double &time, const double E0,
                         const double &n, const double &sin_f,
                         const double &cos_f, const double &x,
                         const double &y, const double &sin_fpw,
                         const double &cos_fpw,
                         const double &atan_mcs_fpw, const double &r,
                         const double some, const double sope,
                         const double omes, const double ope_cosf,
                         const double &d, const double &d_squared,
                         double &dd_dt0, double &dd_dp,
                         double &dd_da, double &dd_dinc,
                         double &dd_de, double &dd_domega,
                         double &dnu_dt0, double &dnu_dp,
                         double &dnu_da, double &dnu_dinc,
                         double &dnu_de, double &dnu_domega) {

  // Compute d partial derivatives: first branches of the tree.
  const double dd_dx = x / d;
  const double dd_dy = y / d;

  // Compute d partial derivatives: second branches of the tree.
  const double dx_dr = cos_fpw;
  const double dx_dsinf = -r * sin_omega;
  const double dx_dcosf = r * cos_omega;
  const double dx_domega = -r * sin_fpw;
  const double dy_dr = cos_i * sin_fpw;
  const double dy_dsinf = r * cos_i * cos_omega;
  const double dy_dcosf = r * cos_i * sin_omega;
  const double dy_domega = r * cos_i * cos_fpw;
  const double dy_dinc = -r * sin_i * sin_fpw;

  // Compute d partial derivatives: third branches of the tree.
  const double ope_cosfs = ope_cosf * ope_cosf;
  const double omes_ptot = std::pow(omes, fractions::three_halves);
  const double dr_da = omes / ope_cosf;
  const double dr_de = -a * (cos_f * (1. + ecc * ecc) + 2. * ecc) / ope_cosfs;
  const double dr_dcosf = -a * ecc * omes / ope_cosfs;
  const double dsinf_dM = cos_f * ope_cosfs / omes_ptot;
  const double dsinf_de = cos_f * sin_f * (1. + ope_cosf) / omes;
  const double dcosf_dM = -sin_f * ope_cosfs / omes_ptot;
  const double dcosf_de = -sin_f * sin_f * (1. + ope_cosf) / omes;

  // Compute d partial derivatives: fourth/fifth branches of the tree.
  const double alpha = some * cos_omega / (sope * (1. + sin_omega));
  const double toaspo = 2. / (alpha * alpha + 1);
  const double omecosE0_toaspo = (1. - ecc * std::cos(E0)) * toaspo;
  const double dM_dt0 = -n;
  const double dM_dp = (t0 - time) * fractions::twopi / (period * period);
  const double dM_de = -std::sin(E0) + omecosE0_toaspo * -cos_omega
                       / ((1. + sin_omega) * some * std::pow(sope, 3.));
  const double dM_domega = omecosE0_toaspo * -some / ((1. + sin_omega) * sope);

  // Compute nu partial derivatives: first branches of the tree.
  const double dnu_dx = -y / d_squared;
  const double dnu_dy = x / d_squared;
  const double dnu_dpsi = -1.;

  // Compute nu partial derivatives: second branches of the tree.
  const double dpsi_dsinf = cos_i * cos_f;
  const double dpsi_dcosf = -cos_i * sin_f;
  const double dpsi_domega = cos_i;
  const double dpsi_dinc = -sin_i * atan_mcs_fpw;

  // Compute dd_dt0, dd_dp, dd_da, dd_dinc, dd_de, and dd_dw via
  // the chain rule.
  dd_dt0 = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                    + dx_dsinf * dsinf_dM * dM_dt0
                    + dx_dcosf * dcosf_dM * dM_dt0)
           + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                      + dy_dsinf * dsinf_dM * dM_dt0
                      + dy_dcosf * dcosf_dM * dM_dt0);
  dd_dp = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                   + dx_dsinf * dsinf_dM * dM_dp
                   + dx_dcosf * dcosf_dM * dM_dp)
          + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                     + dy_dsinf * dsinf_dM * dM_dp
                     + dy_dcosf * dcosf_dM * dM_dp);
  dd_da = dd_dx * dx_dr * dr_da + dd_dy * dy_dr * dr_da;
  dd_dinc = dd_dy * dy_dinc;
  dd_de = dd_dx * (dx_dr * (dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                   + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                   + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
          + dd_dy * (dy_dr * (dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                     + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                     + dy_dcosf * (dcosf_de + dcosf_dM * dM_de));
  dd_domega = dd_dx * (dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                       + dx_dsinf * dsinf_dM * dM_domega
                       + dx_dcosf * dcosf_dM * dM_domega)
              + dd_dy * (dy_domega + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                         + dy_dsinf * dsinf_dM * dM_domega
                         + dy_dcosf * dcosf_dM * dM_domega);

  // Compute dnu_dt0, dnu_dp, dnu_da, dnu_dinc, dnu_de, and dnu_dw
  // via the chain rule.
  dnu_dt0 = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                      + dx_dsinf * dsinf_dM * dM_dt0
                      + dx_dcosf * dcosf_dM * dM_dt0)
            + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                        + dy_dsinf * dsinf_dM * dM_dt0
                        + dy_dcosf * dcosf_dM * dM_dt0)
            + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dt0
                          + dpsi_dcosf * dcosf_dM * dM_dt0);
  dnu_dp = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                     + dx_dsinf * dsinf_dM * dM_dp
                     + dx_dcosf * dcosf_dM * dM_dp)
           + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                       + dy_dsinf * dsinf_dM * dM_dp
                       + dy_dcosf * dcosf_dM * dM_dp)
           + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dp
                         + dpsi_dcosf * dcosf_dM * dM_dp);
  dnu_da = dnu_dx * dx_dr * dr_da + dnu_dy * dy_dr * dr_da;
  dnu_dinc = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
  dnu_de = dnu_dx * (dx_dr * (dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                     + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                     + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
           + dnu_dy * (dy_dr * (dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                       + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                       + dy_dcosf * (dcosf_de + dcosf_dM * dM_de))
           + dnu_dpsi * (dpsi_dsinf * (dsinf_de + dsinf_dM * dM_de)
                         + dpsi_dcosf * (dcosf_de + dcosf_dM * dM_de));
  dnu_domega = dnu_dx * (dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                         + dx_dsinf * dsinf_dM * dM_domega
                         + dx_dcosf * dcosf_dM * dM_domega)
               + dnu_dy * (dy_domega + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                           + dy_dsinf * dsinf_dM * dM_domega
                           + dy_dcosf * dcosf_dM * dM_domega)
               + dnu_dpsi * (dpsi_domega + dpsi_dsinf * dsinf_dM * dM_domega
                             + dpsi_dcosf * dcosf_dM * dM_domega);
}
