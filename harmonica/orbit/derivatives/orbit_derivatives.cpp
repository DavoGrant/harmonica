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

  // Compute partial derivatives: first branches of the trees.
  double dd_dx = x / d;
  double dd_dy = y / d;
  double dnu_dx = -y / d_squared;
  double dnu_dy = x / d_squared;
  double dnu_dpsi = -1.;

  // Compute partial derivatives: second branches of the trees.
  double dx_da = cos_M;
  double dx_dM = -a * sin_M;
  double dy_da = sin_M * cos_i;
  double dy_dM = a * cos_M * cos_i;
  double dy_dinc = -a * sin_M * sin_i;
  double dpsi_dM = cos_i;
  double dpsi_dinc = -sin_i * atan_mcsM;

  // Compute partial derivatives: third branches of the trees.
  double dM_dp = (t0 - time) * fractions::twopi / (period * period);
  double dM_dt0 = -n;

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
