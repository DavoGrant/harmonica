#include <cmath>
#include <tuple>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "trajectories.hpp"
#include "kepler.hpp"
#include "../constants/constants.hpp"

namespace py = pybind11;


void orbital_trajectories(double t0, double period, double a,
                          double inc, double ecc, double omega,
                          py::array_t<double, py::array::c_style> times,
                          py::array_t<double, py::array::c_style> ds,
                          py::array_t<double, py::array::c_style> nus,
                          py::array_t<double, py::array::c_style> ds_grad,
                          py::array_t<double, py::array::c_style> nus_grad,
                          bool require_gradients) {

  // Unpack python arrays.
  auto times_ = times.unchecked<1>();
  auto ds_ = ds.mutable_unchecked<1>();
  auto nus_ = nus.mutable_unchecked<1>();
  auto ds_grad_ = ds_grad.mutable_unchecked<2>();
  auto nus_grad_ = nus_grad.mutable_unchecked<2>();

  const double n = fractions::twopi / period;
  const double sin_i = std::sin(inc);
  const double cos_i = std::cos(inc);

  if (ecc == 0.) {

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      const double tp = t0 - fractions::pi_d_2 / n;

      // Compute mean anomaly.
      const double time = times_(i);
      const double M = (time - tp) * n;

      // Compute sine and cosine of the true anomaly.
      const double sin_M = std::sin(M);
      const double cos_M = std::cos(M);

      // Compute location of planet centre relative to stellar centre.
      const double x = a * cos_M;
      const double y = a * cos_i * sin_M;

      // Compute angle between x-axis and planet velocity.
      const double atan_mcsM = std::atan(-cos_M / sin_M);
      const double psi = cos_i * atan_mcsM;

      // Compute separation distance between planet and stellar centres.
      const double d_squared = x * x + y * y;
      const double d = std::sqrt(d_squared);
      ds_(i) = d;

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      // Optionally compute derivatives.
      if (require_gradients == true) {

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
          const double dM_dp = (t0 - time) * fractions::twopi
                               / (period * period);

          // Compute nu partial derivatives: first branches of the tree.
          const double dnu_dx = -y / d_squared;
          const double dnu_dy = x / d_squared;
          const double dnu_dpsi = -1.;

          // Compute nu partial derivatives: second branches of the tree.
          const double dpsi_dM = cos_i;
          const double dpsi_dinc = -sin_i * atan_mcsM;

          // Compute dd_dt0, dd_dp, dd_da, and dd_dinc via the chain rule.
          ds_grad_(i, 0) = dd_dx * dx_dM * dM_dt0 + dd_dy * dy_dM * dM_dt0;
          ds_grad_(i, 1) = dd_dx * dx_dM * dM_dp + dd_dy * dy_dM * dM_dp;
          ds_grad_(i, 2) = dd_dx * dx_da + dd_dy * dy_da;
          ds_grad_(i, 3) = dd_dy * dy_dinc;

          // Compute dnu_dt0, dnu_dp, dnu_da, and dnu_dinc via the chain rule.
          nus_grad_(i, 0) = dnu_dx * dx_dM * dM_dt0 + dnu_dy * dy_dM * dM_dt0
                            + dnu_dpsi * dpsi_dM * dM_dt0;
          nus_grad_(i, 1) = dnu_dx * dx_dM * dM_dp + dnu_dy * dy_dM * dM_dp
                            + dnu_dpsi * dpsi_dM * dM_dp;
          nus_grad_(i, 2) = 0.;
          nus_grad_(i, 3) = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
      }
    }

  } else {

    const double sin_omega = std::sin(omega);
    const double cos_omega = std::cos(omega);

    // Iterate evaluation times.
    for (py::ssize_t i = 0; i < times_.shape(0); i++) {

      // Compute time of periastron.
      const double some = std::sqrt(1. - ecc);
      const double sope = std::sqrt(1. + ecc);
      const double E0 = 2. * std::atan2(some * cos_omega,
                                        sope * (1 + sin_omega));
      const double M0 = E0 - ecc * std::sin(E0);
      const double tp = t0 - M0 / n;

      // Compute mean anomaly.
      const double time = times_(i);
      const double M = (time - tp) * n;

      // Compute sine and cosine of the true anomaly.
      std::tuple<double, double> sin_cos_f = solve_kepler(M, ecc);
      const double sin_f = std::get<0>(sin_cos_f);
      const double cos_f = std::get<1>(sin_cos_f);

      // Compute location of planet centre relative to stellar centre.
      const double omes = 1. - ecc * ecc;
      const double ope_cosf = 1. + ecc * std::get<1>(sin_cos_f);
      const double r = a * omes / ope_cosf;
      const double sin_fpw = cos_f * sin_omega + sin_f * cos_omega;
      const double cos_fpw = cos_f * cos_omega - sin_f * sin_omega;
      const double x = r * cos_fpw;
      const double y = r * cos_i * sin_fpw;

      // Compute angle between x-axis and planet velocity.
      const double atan_mcs_fpw = std::atan(-cos_fpw / sin_fpw);
      const double psi = cos_i * atan_mcs_fpw;

      // Compute separation distance between planet and stellar centres.
      const double d_squared = x * x + y * y;
      const double d = std::sqrt(d_squared);
      ds_(i) = d;

      // Compute angle between planet velocity and stellar centre.
      nus_(i) = std::atan2(y, x) - psi;

      // Optionally compute derivatives dd/dz and dnu/dz, for z in
      // the set {t0, p, a, i, e, w} for eccentric trajectories.
      if (require_gradients == true) {

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
          const double dr_de = -a * (cos_f * (1. + ecc * ecc) + 2. * ecc)
                               / ope_cosfs;
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
          const double dM_dp = (t0 - time) * fractions::twopi
                       / (period * period);
          const double dM_de = -std::sin(E0) + omecosE0_toaspo * -cos_omega
                               / ((1. + sin_omega) * some
                               * std::pow(sope, 3.));
          const double dM_domega = omecosE0_toaspo * -some
                                   / ((1. + sin_omega) * sope);

          // Compute nu partial derivatives: first branches of the tree.
          const double dnu_dx = -y / d_squared;
          const double dnu_dy = x / d_squared;
          const double dnu_dpsi = -1.;

          // Compute nu partial derivatives: second branches of the tree.
          const double dpsi_dsinf = cos_i * cos_f;
          const double dpsi_dcosf = -cos_i * sin_f;
          const double dpsi_domega = cos_i;
          const double dpsi_dinc = -sin_i * atan_mcs_fpw;

          // Compute dd_dt0, dd_dp, dd_da, dd_dinc, dd_de, and dd_dw
          // via the chain rule. Probs autograd next time eh.
          ds_grad_(i, 0) = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                                    + dx_dsinf * dsinf_dM * dM_dt0
                                    + dx_dcosf * dcosf_dM * dM_dt0)
                           + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                                      + dy_dsinf * dsinf_dM * dM_dt0
                                      + dy_dcosf * dcosf_dM * dM_dt0);
          ds_grad_(i, 1) = dd_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                                    + dx_dsinf * dsinf_dM * dM_dp
                                    + dx_dcosf * dcosf_dM * dM_dp)
                           + dd_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                                      + dy_dsinf * dsinf_dM * dM_dp
                                      + dy_dcosf * dcosf_dM * dM_dp);
          ds_grad_(i, 2) = dd_dx * dx_dr * dr_da + dd_dy * dy_dr * dr_da;
          ds_grad_(i, 3) = dd_dy * dy_dinc;
          ds_grad_(i, 4) = dd_dx * (dx_dr * (
              dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
            + dd_dy * (dy_dr * (dr_de + dr_dcosf * (dcosf_de
                                                    + dcosf_dM * dM_de))
                       + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                       + dy_dcosf * (dcosf_de + dcosf_dM * dM_de));
          ds_grad_(i, 5) = dd_dx * (
              dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
                + dx_dsinf * dsinf_dM * dM_domega
                + dx_dcosf * dcosf_dM * dM_domega)
            + dd_dy * (dy_domega + dy_dr * dr_dcosf * dcosf_dM * dM_domega
                       + dy_dsinf * dsinf_dM * dM_domega
                       + dy_dcosf * dcosf_dM * dM_domega);

          // Compute dnu_dt0, dnu_dp, dnu_da, dnu_dinc, dnu_de, and dnu_dw
          // via the chain rule.
          nus_grad_(i, 0) = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dt0
                                      + dx_dsinf * dsinf_dM * dM_dt0
                                      + dx_dcosf * dcosf_dM * dM_dt0)
                            + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dt0
                                        + dy_dsinf * dsinf_dM * dM_dt0
                                        + dy_dcosf * dcosf_dM * dM_dt0)
                            + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dt0
                                          + dpsi_dcosf * dcosf_dM * dM_dt0);
          nus_grad_(i, 1) = dnu_dx * (dx_dr * dr_dcosf * dcosf_dM * dM_dp
                                      + dx_dsinf * dsinf_dM * dM_dp
                                      + dx_dcosf * dcosf_dM * dM_dp)
                            + dnu_dy * (dy_dr * dr_dcosf * dcosf_dM * dM_dp
                                        + dy_dsinf * dsinf_dM * dM_dp
                                        + dy_dcosf * dcosf_dM * dM_dp)
                            + dnu_dpsi * (dpsi_dsinf * dsinf_dM * dM_dp
                                          + dpsi_dcosf * dcosf_dM * dM_dp);
          nus_grad_(i, 2) = dnu_dx * dx_dr * dr_da + dnu_dy * dy_dr * dr_da;
          nus_grad_(i, 3) = dnu_dy * dy_dinc + dnu_dpsi * dpsi_dinc;
          nus_grad_(i, 4) = dnu_dx * (dx_dr * (
                dr_de + dr_dcosf * (dcosf_de + dcosf_dM * dM_de))
                + dx_dsinf * (dsinf_de + dsinf_dM * dM_de)
                + dx_dcosf * (dcosf_de + dcosf_dM * dM_de))
            + dnu_dy * (dy_dr * (dr_de + dr_dcosf * (dcosf_de
                                                     + dcosf_dM * dM_de))
                        + dy_dsinf * (dsinf_de + dsinf_dM * dM_de)
                        + dy_dcosf * (dcosf_de + dcosf_dM * dM_de))
            + dnu_dpsi * (dpsi_dsinf * (dsinf_de + dsinf_dM * dM_de)
                          + dpsi_dcosf * (dcosf_de + dcosf_dM * dM_de));
          nus_grad_(i, 5) = dnu_dx * (
              dx_domega + dx_dr * dr_dcosf * dcosf_dM * dM_domega
              + dx_dsinf * dsinf_dM * dM_domega
              + dx_dcosf * dcosf_dM * dM_domega)
            + dnu_dy * (
              dy_domega + dy_dr * dr_dcosf * dcosf_dM * dM_domega
              + dy_dsinf * dsinf_dM * dM_domega
              + dy_dcosf * dcosf_dM * dM_domega)
            + dnu_dpsi * (
              dpsi_domega + dpsi_dsinf * dsinf_dM * dM_domega
              + dpsi_dcosf * dcosf_dM * dM_domega);
      }
    }
  }
}
