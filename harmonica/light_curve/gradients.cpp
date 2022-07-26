#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "gradients.hpp"
#include "../constants/constants.hpp"

using namespace std::complex_literals;


FluxDerivatives::FluxDerivatives(
    int ld_law, double us[], int n_rs, double rs[],
    int pnl_c, int pnl_e) : Fluxes(ld_law, us, n_rs, rs, pnl_c, pnl_e) {

  // Pre-compute some derivative terms.
  m_df_dalpha = -1.;
  if (m_ld_law == limb_darkening::quadratic) {
    m_dI0_du1 = 1. / (fractions::threepi * m_I_0_bts);
    m_dI0_du2 = 1. / (fractions::sixpi * m_I_0_bts);

    m_dalpha_ds0 = m_I_0 * (1. - us[0] - us[1]);
    m_dalpha_ds1 = m_I_0 * (us[0] + 2. * us[1]);
    m_dalpha_ds2 = -m_I_0 * us[1];

  } else {
    m_dI0_du1 = 1. / (fractions::fivepi * m_I_0_bts);
    m_dI0_du2 = 1. / (fractions::threepi * m_I_0_bts);
    m_dI0_du3 = 1. / (fractions::sevenpi_d_3 * m_I_0_bts);
    m_dI0_du4 = 1. / (fractions::twopi * m_I_0_bts);

    m_dalpha_ds0 = m_I_0 * (1. - us[0] - us[1] - us[2] - us[3]);
    m_dalpha_ds12 = m_I_0 * us[0];
    m_dalpha_ds1 = m_I_0 * us[1];
    m_dalpha_ds32 = m_I_0 * us[2];
    m_dalpha_ds2 = m_I_0 * us[3];
  }

  m_df_dcs.resize(m_n_rs);
  m_ds0_dcs.resize(m_n_rs);
  m_ds12_dcs.resize(m_n_rs);
  m_ds1_dcs.resize(m_n_rs);
  m_ds32_dcs.resize(m_n_rs);
  m_ds2_dcs.resize(m_n_rs);

  m_dc0_da0 = 1.;
  m_dcplus_dan = 1. / 2.;
  m_dcminus_dan = 1. / 2.;
  m_dcplus_dbn = -1.i / 2.;
  m_dcminus_dbn = 1.i / 2.;

  m_zeroes_c_conv_c.resize(m_len_c_conv_c);
  m_zeroes_c_conv_c.setZero();

  // Pre-build nested vector structures.
  m_els.resize(m_n_rs);
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    m_dthetas_dcs.push_back({});

    m_els(n + m_N_c).resize(m_n_rs);
    m_els(n + m_N_c).setZero();
    m_els(n + m_N_c)(n + m_N_c) = 1.;
  }

  if (m_N_c != 0) {
    // Pre-build the derivatives of intersection eqn companion matrix.
    m_dC_dd.resize(m_C_shape, m_C_shape);
    m_dC_dd.setZero();
    m_dC_dnu.resize(m_C_shape, m_C_shape);
    m_dC_dnu.setZero();
    m_dC_dcs.resize(m_n_rs);
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      m_dC_dcs(npN_c).resize(m_C_shape, m_C_shape);
      m_dC_dcs(npN_c).setZero();
    }
  }
}


void FluxDerivatives::transit_flux_and_derivatives(
    const double d, const double z, const double nu,
    double& out_f, double out_df_dz[]) {

  this->reset_derivatives();
  this->compute_solution_vector(d, z, nu);

  // Compute transit flux: alpha=I0sTp.
  if (m_ld_law == limb_darkening::quadratic) {
    m_alpha = m_I_0 * (m_s0 * m_p(0) + m_s1 * m_p(1) + m_s2 * m_p(2));
  } else {
    m_alpha = m_I_0 * (m_s0 * m_p(0) + m_s12 * m_p(1) + m_s1 * m_p(2)
                   + m_s32 * m_p(3) + m_s2 * m_p(4));
  }
  out_f = 1. - m_alpha;
  this->f_derivatives(out_df_dz);
}


void FluxDerivatives::find_intersections_theta(
    const double d, const double nu) {

  // Check cases where no obvious intersections, avoiding eigenvalue runtime.
  if (this->no_obvious_intersections(d, nu)) {
    m_dthetas_dd = {0., 0.};
    m_dthetas_dnu = {0., 0.};
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      m_dthetas_dcs[n + m_N_c] = {0., 0.};
    }
    return;
  }

  if (m_N_c != 0) {
    // Update intersection companion matrix for current position.
    Eigen::Matrix<std::complex<double>, EigD, EigD> C = m_C0;
    std::complex<double> moo_denom =
      intersection_polynomial_coefficient_moo_denom(m_C_shape);
    for (int j = 1; j < m_C_shape + 1; j++) {
      C(j - 1, m_C_shape - 1) +=
        this->intersection_polynomial_coefficients_h_j_update(j - 1);
      C(j - 1, m_C_shape - 1) *= moo_denom;
    }

    // Get the intersection companion matrix roots.
    m_theta = this->compute_real_theta_roots(C, m_C_shape);

  } else {
    // Transmission string is a circle.
    double _c0c0 = m_c(m_N_c).real() * m_c(m_N_c).real();
    double acos_arg = (_c0c0 + m_dd - 1.) / (2. * m_c(m_N_c).real() * d);
    double acos_intersect = std::acos(acos_arg);
    m_theta = {nu - acos_intersect, nu + acos_intersect};

    double dtheta_dacos_arg = 1. / std::sqrt(1. - acos_arg * acos_arg);
    double dacos_arg_dd = (-_c0c0 + m_dd - 1.) / (2. * m_c(m_N_c).real() * m_dd);
    double dacos_arg_dc0 = (_c0c0 - m_dd - 1.) / (2. * _c0c0 * d);
    double dtheta_dd = dtheta_dacos_arg * dacos_arg_dd;
    double dtheta_c0 = dtheta_dacos_arg * dacos_arg_dc0;
    m_dthetas_dd = {dtheta_dd, -dtheta_dd};
    m_dthetas_dnu = {1., 1.};
    m_dthetas_dcs = {{dtheta_c0, -dtheta_c0}};
  }

  if (m_theta.size() == 0) {
    // No roots, check which trivial case this configuration corresponds to.
    if (this->trivial_configuration(d, nu)) {
      m_dthetas_dd = {0., 0.};
      m_dthetas_dnu = {0., 0.};
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        m_dthetas_dcs[n + m_N_c] = {0., 0.};
      }
      return;
    }
  }

  // Sort thetas and corresponding theta derivatives.
  std::vector<int> indices(m_theta.size());
  std::vector<double> theta_unsorted = m_theta;
  std::vector<double> m_dthetas_dd_unsorted = m_dthetas_dd;
  std::vector<double> m_dthetas_dnu_unsorted = m_dthetas_dnu;
  std::vector<std::vector<std::complex<double>>> m_dthetas_dcs_unsorted = m_dthetas_dcs;
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](double A, double B)->bool {
    return theta_unsorted[A] < theta_unsorted[B];
  });
  for (int j = 0; j < m_theta.size(); j++) {
    m_theta[j] = theta_unsorted[indices[j]];
    m_dthetas_dd[j] = m_dthetas_dd_unsorted[indices[j]];
    m_dthetas_dnu[j] = m_dthetas_dnu_unsorted[indices[j]];
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      m_dthetas_dcs[npN_c][j] = m_dthetas_dcs_unsorted[npN_c][indices[j]];
    }
  }

  // Duplicate first derivatives at end of vector.
  m_dthetas_dd.push_back(m_dthetas_dd[0]);
  m_dthetas_dnu.push_back(m_dthetas_dnu[0]);
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    int npN_c = n + m_N_c;
    m_dthetas_dcs[npN_c].push_back(m_dthetas_dcs[npN_c][0]);
  }

  // Ensure theta vector spans a closed loop, 2pi in total.
  // Thus, duplicate first intersection + 2pi at the end.
  m_theta.push_back(m_theta[0] + fractions::twopi);

  // Characterise theta pairs.
  this->characterise_intersection_pairs(d, nu);
}


void FluxDerivatives::s_star(int _j, int theta_type_j, double _theta_j,
                             double _theta_j_p1, const double d,
                             const double nu) {
  double phi_j;
  double phi_j_p1;

  double dsn_phi_j_dd = 0., dsn_phi_j_p1_dd = 0.,
         dsn_phi_j_dnu = 0., dsn_phi_j_p1_dnu = 0.,
         dsn_phi_j_theta_j_dx = 0., dsn_phi_j_p1_theta_j_p1_dx = 0.,
         dsn_phi_j_rp_theta_j_dx = 0., dsn_phi_j_p1_rp_theta_j_p1_dx = 0.;
  std::complex<double> dsn_phi_j_rp_dcs[m_n_rs], dsn_phi_j_p1_rp_dcs[m_n_rs];

  // Check if entire star -pi to pi.
  if (theta_type_j == intersections::entire_star) {
    phi_j = -fractions::pi;
    phi_j_p1 = fractions::pi;
  } else {
    // Convert theta_j to phi_j (stellar centred frame).
    double rp_theta_j = this->rp_theta(_theta_j);
    double sin_thetajmnu = std::sin(_theta_j - nu);
    double cos_thetajmnu = std::cos(_theta_j - nu);
    phi_j = std::atan2(-rp_theta_j * sin_thetajmnu,
                       -rp_theta_j * cos_thetajmnu + d);

    // Convert theta_j_p1 to phi_j_p1.
    double rp_theta_j_p1 = this->rp_theta(_theta_j_p1);
    double sin_thetajp1mnu = std::sin(_theta_j_p1 - nu);
    double cos_thetajp1mnu = std::cos(_theta_j_p1 - nu);
    phi_j_p1 = std::atan2(
      -rp_theta_j_p1 * sin_thetajp1mnu,
      -rp_theta_j_p1 * cos_thetajp1mnu + d);

    double drpjs = rp_theta_j * rp_theta_j;
    double drpjp1s = rp_theta_j_p1 * rp_theta_j_p1;
    double drp_dtheta_j = this->drp_dtheta(_theta_j);
    double drp_dtheta_j_p1 = this->drp_dtheta(_theta_j_p1);
    double dphi_j_denom = m_dd - 2. * d * rp_theta_j
                          * cos_thetajmnu + drpjs;
    double dphi_j_p1_denom = m_dd - 2. * d * rp_theta_j_p1
                             * cos_thetajp1mnu + drpjp1s;

    double dphi_j_dd = (rp_theta_j * sin_thetajmnu) / dphi_j_denom;
    double dphi_j_p1_dd = (rp_theta_j_p1 * sin_thetajp1mnu)
                               / dphi_j_p1_denom;
    double dphi_j_dnu = (d * rp_theta_j * cos_thetajmnu - drpjs)
                         / dphi_j_denom;
    double dphi_j_p1_dnu = (d * rp_theta_j_p1 * cos_thetajp1mnu
                                - drpjp1s) / dphi_j_p1_denom;

    double dphi_j_dtheta_j = (drpjs - d * rp_theta_j * cos_thetajmnu)
                              / dphi_j_denom;
    double dphi_j_p1_dtheta_j = (drpjp1s - d * rp_theta_j_p1
                                     * cos_thetajp1mnu)
                                     / dphi_j_p1_denom;
    double dphi_j_drp = (-d * sin_thetajmnu) / dphi_j_denom;
    double dphi_j_p1_drp = (-d * sin_thetajp1mnu) / dphi_j_p1_denom;

    dsn_phi_j_dd = dphi_j_dd;
    dsn_phi_j_p1_dd = dphi_j_p1_dd;
    dsn_phi_j_dnu = dphi_j_dnu;
    dsn_phi_j_p1_dnu = dphi_j_p1_dnu;
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      std::complex<double> drpj_dcn = std::exp((1. * n) * 1.i * _theta_j);
      std::complex<double> drpjp1_dcn = std::exp((1. * n) * 1.i * _theta_j_p1);
      dsn_phi_j_rp_dcs[npN_c] += dphi_j_drp * drpj_dcn;
      dsn_phi_j_p1_rp_dcs[npN_c] += dphi_j_p1_drp * drpjp1_dcn;
    }

    dsn_phi_j_theta_j_dx = dphi_j_dtheta_j;
    dsn_phi_j_p1_theta_j_p1_dx = dphi_j_p1_dtheta_j;
    dsn_phi_j_rp_theta_j_dx = dphi_j_drp * drp_dtheta_j;
    dsn_phi_j_p1_rp_theta_j_p1_dx = dphi_j_p1_drp * drp_dtheta_j_p1;
  }

  // Evaluate line integral anticlockwise.
  double phi_diff = phi_j_p1 - phi_j;
    if (m_ld_law == limb_darkening::quadratic) {
    // Limb-darkening terms n=0,1,2.
    m_s0 += fractions::one_half * phi_diff;
    m_s1 += fractions::one_third * phi_diff;
    m_s2 += fractions::one_quarter * phi_diff;

  } else {
    // Limb-darkening terms n=0,1/2,1,3/2,2.
    m_s0 += fractions::one_half * phi_diff;
    m_s12 += fractions::two_fifths * phi_diff;
    m_s1 += fractions::one_third * phi_diff;
    m_s32 += fractions::two_sevenths * phi_diff;
    m_s2 += fractions::one_quarter * phi_diff;
  }

  double dsn_dd = -dsn_phi_j_dd + dsn_phi_j_p1_dd
    - (dsn_phi_j_theta_j_dx + dsn_phi_j_rp_theta_j_dx) * m_dthetas_dd[_j]
    + (dsn_phi_j_p1_theta_j_p1_dx
       + dsn_phi_j_p1_rp_theta_j_p1_dx) * m_dthetas_dd[_j + 1];

  double dsn_dnu = -dsn_phi_j_dnu + dsn_phi_j_p1_dnu
    - (dsn_phi_j_theta_j_dx + dsn_phi_j_rp_theta_j_dx) * m_dthetas_dnu[_j]
    + (dsn_phi_j_p1_theta_j_p1_dx
       + dsn_phi_j_p1_rp_theta_j_p1_dx) * m_dthetas_dnu[_j + 1];

  std::complex<double> dsn_dcs[m_n_rs];
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    int npN_c = n + m_N_c;
    dsn_dcs[npN_c] = -dsn_phi_j_rp_dcs[npN_c] + dsn_phi_j_p1_rp_dcs[npN_c]
      - (dsn_phi_j_theta_j_dx + dsn_phi_j_rp_theta_j_dx)
      * m_dthetas_dcs[npN_c][_j]
      + (dsn_phi_j_p1_theta_j_p1_dx
         + dsn_phi_j_p1_rp_theta_j_p1_dx) * m_dthetas_dcs[npN_c][_j + 1];
  }

  if (m_ld_law == limb_darkening::quadratic) {

    m_ds0_dd += fractions::one_half * dsn_dd;
    m_ds1_dd += fractions::one_third * dsn_dd;
    m_ds2_dd += fractions::one_quarter * dsn_dd;

    m_ds0_dnu += fractions::one_half * dsn_dnu;
    m_ds1_dnu += fractions::one_third * dsn_dnu;
    m_ds2_dnu += fractions::one_quarter * dsn_dnu;

    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      m_ds0_dcs(npN_c) += fractions::one_half * dsn_dcs[npN_c];
      m_ds1_dcs(npN_c) += fractions::one_third * dsn_dcs[npN_c];
      m_ds2_dcs(npN_c) += fractions::one_quarter * dsn_dcs[npN_c];
    }

  } else {

    m_ds0_dd += fractions::one_half * dsn_dd;
    m_ds12_dd += fractions::two_fifths * dsn_dd;
    m_ds1_dd += fractions::one_third * dsn_dd;
    m_ds32_dd += fractions::two_sevenths * dsn_dd;
    m_ds2_dd += fractions::one_quarter * dsn_dd;

    m_ds0_dnu += fractions::one_half * dsn_dnu;
    m_ds12_dnu += fractions::two_fifths * dsn_dnu;
    m_ds1_dnu += fractions::one_third * dsn_dnu;
    m_ds32_dnu += fractions::two_sevenths * dsn_dnu;
    m_ds2_dnu += fractions::one_quarter * dsn_dnu;

    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      m_ds0_dcs(npN_c) += fractions::one_half * dsn_dcs[npN_c];
      m_ds12_dcs(npN_c) += fractions::two_fifths * dsn_dcs[npN_c];
      m_ds1_dcs(npN_c) += fractions::one_third * dsn_dcs[npN_c];
      m_ds32_dcs(npN_c) += fractions::two_sevenths * dsn_dcs[npN_c];
      m_ds2_dcs(npN_c) += fractions::one_quarter * dsn_dcs[npN_c];
    }
  }
}


void FluxDerivatives::f_derivatives(double df_dz[]) {

  double dalpha_dI0 = m_alpha / m_I_0;
  if (m_ld_law == limb_darkening::quadratic) {

    // df_dd, df_dnu.
    df_dz[0] = m_df_dalpha * (m_dalpha_ds0 * m_ds0_dd
                            + m_dalpha_ds1 * m_ds1_dd
                            + m_dalpha_ds2 * m_ds2_dd);
    df_dz[1] = m_df_dalpha * (m_dalpha_ds0 * m_ds0_dnu
                            + m_dalpha_ds1 * m_ds1_dnu
                            + m_dalpha_ds2 * m_ds2_dnu);

    // df_du1, df_du2.
    double dalpha_du1 = m_I_0 * (m_s1 - m_s0);
    double dalpha_du2 = m_I_0 * (2. * m_s1 - m_s0 - m_s2);
    df_dz[2] = m_df_dalpha * (dalpha_dI0 * m_dI0_du1 + dalpha_du1);
    df_dz[3] = m_df_dalpha * (dalpha_dI0 * m_dI0_du2 + dalpha_du2);

    // df_drs.
    m_df_dcs = m_df_dalpha * (m_dalpha_ds0 * m_ds0_dcs
                              + m_dalpha_ds1 * m_ds1_dcs
                              + m_dalpha_ds2 * m_ds2_dcs);
    for (int n = 0; n < m_N_c + 1; n++) {
      if (n == 0){
        df_dz[4] = (m_df_dcs(m_N_c) * m_dc0_da0).real();
      } else {
        df_dz[3 + 2 * n] = (m_df_dcs(n + m_N_c) * m_dcplus_dan
                            + m_df_dcs(-n + m_N_c) * m_dcminus_dan).real();
        df_dz[4 + 2 * n] = (m_df_dcs(n + m_N_c) * m_dcplus_dbn
                            + m_df_dcs(-n + m_N_c) * m_dcminus_dbn).real();
      }
    }
  } else {

    // df_dd, df_dnu.
    df_dz[0] = m_df_dalpha * (
      m_dalpha_ds0 * m_ds0_dd + m_dalpha_ds12 * m_ds12_dd
      + m_dalpha_ds1 * m_ds1_dd + m_dalpha_ds32 * m_ds32_dd
      + m_dalpha_ds2 * m_ds2_dd);
    df_dz[1] = m_df_dalpha * (
      m_dalpha_ds0 * m_ds0_dnu + m_dalpha_ds12 * m_ds12_dnu
      + m_dalpha_ds1 * m_ds1_dnu + m_dalpha_ds32 * m_ds32_dnu
      + m_dalpha_ds2 * m_ds2_dnu);

    // df_du1, df_du2, df_du3, df_du4.
    double dalpha_du1 = m_I_0 * (m_s12 - m_s0);
    double dalpha_du2 = m_I_0 * (m_s1 - m_s0);
    double dalpha_du3 = m_I_0 * (m_s32 - m_s0);
    double dalpha_du4 = m_I_0 * (m_s2 - m_s0);
    df_dz[2] = m_df_dalpha * (dalpha_dI0 * m_dI0_du1 + dalpha_du1);
    df_dz[3] = m_df_dalpha * (dalpha_dI0 * m_dI0_du2 + dalpha_du2);
    df_dz[4] = m_df_dalpha * (dalpha_dI0 * m_dI0_du3 + dalpha_du3);
    df_dz[5] = m_df_dalpha * (dalpha_dI0 * m_dI0_du4 + dalpha_du4);

    // df_drs.
    m_df_dcs = m_df_dalpha * (
      m_dalpha_ds0 * m_ds0_dcs + m_dalpha_ds12 * m_ds12_dcs
      + m_dalpha_ds1 * m_ds1_dcs + m_dalpha_ds32 * m_ds32_dcs
      + m_dalpha_ds2 * m_ds2_dcs);
    for (int n = 0; n < m_N_c + 1; n++) {
      if (n == 0){
        df_dz[6] = (m_df_dcs(m_N_c) * m_dc0_da0).real();
      } else {
        df_dz[5 + 2 * n] = (m_df_dcs(n + m_N_c) * m_dcplus_dan
                            + m_df_dcs(-n + m_N_c) * m_dcminus_dan).real();
        df_dz[6 + 2 * n] = (m_df_dcs(n + m_N_c) * m_dcplus_dbn
                            + m_df_dcs(-n + m_N_c) * m_dcminus_dbn).real();
      }
    }
  }
}


void FluxDerivatives::reset_derivatives() {
  // Reset derivatives.
  m_dthetas_dd.clear();
  m_dthetas_dnu.clear();
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    m_dthetas_dcs[n + m_N_c].clear();
  }
  m_ds0_dd = 0., m_ds12_dd = 0., m_ds1_dd = 0., m_ds32_dd = 0., m_ds2_dd = 0.;
  m_ds0_dnu = 0., m_ds12_dnu = 0., m_ds1_dnu = 0., m_ds32_dnu = 0.,
  m_ds2_dnu = 0.;

  m_ds0_dcs.setZero(), m_ds12_dcs.setZero(), m_ds1_dcs.setZero(),
  m_ds32_dcs.setZero(), m_ds2_dcs.setZero();
}


std::complex<double> FluxDerivatives::dh_j_dd(
  int j) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> _dh_j_dd = 0.;
  if (m_N_c - 1 <= j && j < m_N_c + 1) {
    _dh_j_dd -= m_expinu * m_c(j + 1 - m_N_c);
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    _dh_j_dd -= m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dd -= m_expminu * m_c(j - 1 - m_N_c);
  } else if (j == 2 * m_N_c) {
    _dh_j_dd -= m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dd -= m_expminu * m_c(j - 1 - m_N_c);
    _dh_j_dd += m_td;
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    _dh_j_dd -= m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dd -= m_expminu * m_c(j - 1 - m_N_c);
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    _dh_j_dd -= m_expminu * m_c(j - 1 - m_N_c);
  }
  return _dh_j_dd;
}


std::complex<double> FluxDerivatives::dh_j_dnu(
  int j) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> _dh_j_dnu = 0.;
  if (m_N_c - 1 <= j && j < m_N_c + 1) {
    _dh_j_dnu -= 1.i * m_expinu * m_c(j + 1 - m_N_c);
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    _dh_j_dnu -= 1.i * m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dnu += 1.i * m_expminu * m_c(j - 1 - m_N_c);
  } else if (j == 2 * m_N_c) {
    _dh_j_dnu -= 1.i * m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dnu += 1.i * m_expminu * m_c(j - 1 - m_N_c);
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    _dh_j_dnu -= 1.i * m_expinu * m_c(j + 1 - m_N_c);
    _dh_j_dnu += 1.i * m_expminu * m_c(j - 1 - m_N_c);
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    _dh_j_dnu += 1.i * m_expminu * m_c(j - 1 - m_N_c);
  }
  return _dh_j_dnu;
}


std::complex<double> FluxDerivatives::dh_j_dcn(
  int j, int _n) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> _dh_j_dcn = 0.;
  if (0 <= j && j < m_N_c - 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
  } else if (m_N_c - 1 <= j && j < m_N_c + 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
    if (_n == j + 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expinu;
    }
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
    if (_n == j + 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expinu;
    }
    if (_n == j - 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expminu;
    }
  } else if (j == 2 * m_N_c) {
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
    if (_n == j + 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expinu;
    }
    if (_n == j - 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expminu;
    }
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
    if (_n == j + 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expinu;
    }
    if (_n == j - 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expminu;
    }
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
    if (_n == j - 1 - 2 * m_N_c) {
      _dh_j_dcn -= m_d_expminu;
    }
  } else if (3 * m_N_c + 2 <= j && j < 4 * m_N_c + 1) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      if (_n == n) {
        _dh_j_dcn += m_c(j - n - m_N_c);
      }
      if (_n == j - n - 2 * m_N_c) {
        _dh_j_dcn += m_c(n + m_N_c);
      }
    }
  }
  return _dh_j_dcn;
}


std::vector<double> FluxDerivatives::compute_real_theta_roots(
  const Eigen::Matrix<std::complex<double>, EigD, EigD>&
    companion_matrix, int shape) {

  // Solve eigenvalues.
  Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, EigD, EigD>> ces;
  ces.compute(companion_matrix, true);
  Eigen::Vector<std::complex<double>, EigD> e_vals = ces.eigenvalues();

  // dC_dd, dC_dnu, m_dC_dcs.
  std::complex<double> h_4Nc = this->h_j(m_C_shape);
  std::complex<double> h_4Ncs = h_4Nc * h_4Nc;
  for (int j = 1; j < m_C_shape + 1; j++) {
    std::complex<double> h_jm1 = this->h_j(j - 1);
    m_dC_dd(j - 1, m_C_shape - 1) = (h_jm1 * this->dh_j_dd(m_C_shape)
                                 - this->dh_j_dd(j - 1) * h_4Nc) / h_4Ncs;
    m_dC_dnu(j - 1, m_C_shape - 1) = (h_jm1 * this->dh_j_dnu(m_C_shape)
                                  - this->dh_j_dnu(j - 1) * h_4Nc) / h_4Ncs;
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      m_dC_dcs(n + m_N_c)(j - 1, m_C_shape - 1) = (
        h_jm1 * this->dh_j_dcn(m_C_shape, n)
        - this->dh_j_dcn(j - 1, n) * h_4Nc) / h_4Ncs;
    }
  }

  // And use eigenvectors for derivatives.
  Eigen::Matrix<std::complex<double>, EigD, EigD> e_vecs = ces.eigenvectors();
  Eigen::Matrix<std::complex<double>, EigD, EigD> e_vecs_inv = e_vecs.inverse();
  Eigen::Matrix<std::complex<double>, EigD, EigD>
    dlambda_dd_full = e_vecs_inv * m_dC_dd * e_vecs;
  Eigen::Matrix<std::complex<double>, EigD, EigD>
    dlambda_dnu_full = e_vecs_inv * m_dC_dnu * e_vecs;
  Eigen::Vector<Eigen::Matrix<std::complex<double>, EigD, EigD>, EigD>
    dlambda_dcs_full;
  dlambda_dcs_full.resize(m_n_rs);
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    int npN_c = n + m_N_c;
    dlambda_dcs_full(npN_c) = e_vecs_inv * m_dC_dcs(npN_c) * e_vecs;
  }

  // Select real thetas only, and corresponding derivatives.
  std::vector<double> _theta;
  for (int j = 0; j < shape; j++) {
    double e_vals_abs = std::abs(e_vals(j));
    if (tolerance::unit_circle_lo < e_vals_abs
        && e_vals_abs < tolerance::unit_circle_hi) {
      _theta.push_back(std::arg(e_vals(j)));

      // dtheta_dd, dtheta_dnu, dtheta_dcs.
      std::complex<double> dtheta_dlambda = -1.i / e_vals(j);
      m_dthetas_dd.push_back(
        (dtheta_dlambda * dlambda_dd_full(j, j)).real());
      m_dthetas_dnu.push_back(
        (dtheta_dlambda * dlambda_dnu_full(j, j)).real());
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        m_dthetas_dcs[npN_c].push_back(
          dtheta_dlambda * dlambda_dcs_full(npN_c)(j, j));
      }
    }
  }
  return _theta;
}


void FluxDerivatives::analytic_even_terms(
    int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
    const double d, const double nu) {

  // Build and convolve beta_sin, beta_cos vectors.
  double _theta_diff = _theta_j_p1 - _theta_j;
  double sin_nu = std::sin(nu);
  double cos_nu = std::cos(nu);

  Eigen::Vector<std::complex<double>, 3> beta_sin = m_beta_sin0;
  beta_sin(0) *= sin_nu - cos_nu * 1.i;
  beta_sin(2) *= sin_nu + cos_nu * 1.i;

  Eigen::Vector<std::complex<double>, 3> beta_cos = m_beta_cos0;
  beta_cos(0) *= cos_nu + sin_nu * 1.i;
  beta_cos(2) *= cos_nu - sin_nu * 1.i;

  Eigen::Vector<std::complex<double>, EigD>
    beta_cos_conv_c = complex_convolve(
      beta_cos, m_c, 3, m_n_rs, m_len_beta_conv_c);

  Eigen::Vector<std::complex<double>, EigD>
    beta_sin_conv_Delta_ew_c = complex_convolve(
      beta_sin, m_Delta_ew_c, 3, m_n_rs, m_len_beta_conv_c);

  // Generate q0, equivalent to q_rhs for n=2.
  Eigen::Vector<std::complex<double>, EigD>
    q0 = complex_ca_vector_addition(
      m_c_conv_c, -d * (beta_cos_conv_c + beta_sin_conv_Delta_ew_c),
      m_len_c_conv_c, m_len_beta_conv_c);

  // Generate q lhs.
  Eigen::Vector<std::complex<double>, EigD>
    q_lhs = complex_ca_vector_addition(
      -m_c_conv_c, m_td * beta_cos_conv_c,
      m_len_c_conv_c, m_len_beta_conv_c);
  q_lhs(m_mid_q_lhs) += m_omdd;
  q_lhs(m_mid_q_lhs) += 1.;

  // Generate q2.
  Eigen::Vector<std::complex<double>, EigD>
    q2 = complex_convolve(q_lhs, q0, m_len_q_rhs, m_len_q_rhs, m_len_q);

  Eigen::Vector<std::complex<double>, EigD>
    dq0_dd, dq2_dd, dq0_dnu, dq2_dnu;
  Eigen::Vector<Eigen::Vector<std::complex<double>, EigD>, EigD>
    dq0_dcs, dq2_dcs;
  std::complex<double> ds0_q0_dd = 0., ds2_q2_dd = 0.,
                       ds0_q0_dnu = 0., ds2_q2_dnu = 0.,
                       ds0_theta_j_dx = 0., ds0_theta_j_p1_dx = 0.,
                       ds2_theta_j_dx = 0., ds2_theta_j_p1_dx = 0.;
  std::complex<double> ds0_q0_dcs[m_n_rs], ds2_q2_dcs[m_n_rs];

  Eigen::Vector<std::complex<double>, 3> beta_sin_prime = m_beta_sin0;
  beta_sin_prime(0) *= cos_nu + sin_nu * 1.i;
  beta_sin_prime(2) *= cos_nu - sin_nu * 1.i;

  Eigen::Vector<std::complex<double>, 3> beta_cos_prime = m_beta_cos0;
  beta_cos_prime(0) *= -sin_nu + cos_nu * 1.i;
  beta_cos_prime(2) *= -sin_nu - cos_nu * 1.i;

  Eigen::Vector<std::complex<double>, EigD>
    beta_cos_prime_conv_c = complex_convolve(
      beta_cos_prime, m_c, 3, m_n_rs, m_len_beta_conv_c);

  Eigen::Vector<std::complex<double>, EigD>
    beta_sin_prime_conv_Delta_ew_c = complex_convolve(
      beta_sin_prime, m_Delta_ew_c, 3, m_n_rs, m_len_beta_conv_c);

  dq0_dd = complex_ca_vector_addition(
      m_zeroes_c_conv_c, -(beta_cos_conv_c + beta_sin_conv_Delta_ew_c),
      m_len_c_conv_c, m_len_beta_conv_c);
  dq0_dnu = complex_ca_vector_addition(
      m_zeroes_c_conv_c, -d * (beta_cos_prime_conv_c
                              + beta_sin_prime_conv_Delta_ew_c),
      m_len_c_conv_c, m_len_beta_conv_c);

  Eigen::Vector<std::complex<double>, EigD>
    dq_lhs_dd = complex_ca_vector_addition(
      -m_zeroes_c_conv_c, 2. * beta_cos_conv_c,
      m_len_c_conv_c, m_len_beta_conv_c);
  dq_lhs_dd(m_mid_q_lhs) += -2 * d;
  Eigen::Vector<std::complex<double>, EigD>
    dq_lhs_dnu = complex_ca_vector_addition(
      -m_zeroes_c_conv_c, 2. * d * beta_cos_prime_conv_c,
      m_len_c_conv_c, m_len_beta_conv_c);

  dq2_dd = complex_ca_vector_addition(
      complex_convolve(dq_lhs_dd, q0, m_len_q_rhs, m_len_q_rhs, m_len_q),
      complex_convolve(q_lhs, dq0_dd, m_len_q_rhs, m_len_q_rhs, m_len_q),
      m_len_q, m_len_q);
  dq2_dnu = complex_ca_vector_addition(
      complex_convolve(dq_lhs_dnu, q0, m_len_q_rhs, m_len_q_rhs, m_len_q),
      complex_convolve(q_lhs, dq0_dnu, m_len_q_rhs, m_len_q_rhs, m_len_q),
      m_len_q, m_len_q);

  dq0_dcs.resize(m_n_rs);
  dq2_dcs.resize(m_n_rs);
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    int npN_c = n + m_N_c;
    Eigen::Vector<std::complex<double>, EigD>
      c_conv_el = complex_convolve(
        m_c, m_els(npN_c), m_n_rs, m_n_rs, m_len_c_conv_c);

    Eigen::Vector<std::complex<double>, EigD>
      beta_cos_conv_el = complex_convolve(
        beta_cos, m_els(npN_c), 3, m_n_rs, m_len_beta_conv_c);

    Eigen::Vector<std::complex<double>, EigD>
      beta_sin_conv_el = complex_convolve(
        beta_sin, m_els(npN_c), 3, m_n_rs, m_len_beta_conv_c);

    dq0_dcs(npN_c) = complex_ca_vector_addition(
      2. * c_conv_el, -d * (beta_cos_conv_el
                            + 1.i * (1. * n) * beta_sin_conv_el),
      m_len_c_conv_c, m_len_beta_conv_c);

    Eigen::Vector<std::complex<double>, EigD>
      dq_lhs_dcn = complex_ca_vector_addition(
        -2. * c_conv_el, 2. * d * beta_cos_conv_el,
        m_len_c_conv_c, m_len_beta_conv_c);

    dq2_dcs(npN_c) = complex_ca_vector_addition(
        complex_convolve(dq_lhs_dcn, q0, m_len_q_rhs, m_len_q_rhs, m_len_q),
        complex_convolve(q_lhs, dq0_dcs(npN_c), m_len_q_rhs,
                         m_len_q_rhs, m_len_q),
        m_len_q, m_len_q);
  }

  // Limb-darkening constant term n=0, analytic line integral.
  std::complex<double> s0_planet = 0.;
  for (int m = -m_N_q0; m < m_N_q0 + 1; m++) {
    int mpN_q0 = m + m_N_q0;
    std::complex<double> eim_theta_j = std::exp(
      (1. * m) * 1.i * _theta_j);
    std::complex<double> eim_theta_j_p1 = std::exp(
      (1. * m) * 1.i * _theta_j_p1);
    if (m == 0) {
      s0_planet += q0(mpN_q0) * _theta_diff;

      ds0_q0_dd += dq0_dd(mpN_q0) * _theta_diff;
      ds0_q0_dnu += dq0_dnu(mpN_q0) * _theta_diff;
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        ds0_q0_dcs[npN_c] += dq0_dcs(npN_c)(mpN_q0) * _theta_diff;
      }
    } else {
      s0_planet += q0(mpN_q0) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);

      ds0_q0_dd += dq0_dd(mpN_q0) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
      ds0_q0_dnu += dq0_dnu(mpN_q0) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        ds0_q0_dcs[npN_c] += dq0_dcs(npN_c)(mpN_q0) / (1.i * (1. * m))
                             * (eim_theta_j_p1 - eim_theta_j);
      }
    }
    if (theta_type_j == intersections::planet) {
      ds0_theta_j_dx += q0(mpN_q0) * -eim_theta_j;
      ds0_theta_j_p1_dx += q0(mpN_q0) * eim_theta_j_p1;
    }
  }

  // Limb-darkening even term n=2, analytic line integral.
  std::complex<double> s2_planet = 0.;
  for (int m = -m_N_q2; m < m_N_q2 + 1; m++) {
    int mpN_q2 = m + m_N_q2;
    std::complex<double> eim_theta_j = std::exp(
      (1. * m) * 1.i * _theta_j);
    std::complex<double> eim_theta_j_p1 = std::exp(
      (1. * m) * 1.i * _theta_j_p1);
    if (m == 0) {
      s2_planet += q2(mpN_q2) * _theta_diff;

      ds2_q2_dd += dq2_dd(mpN_q2) * _theta_diff;
      ds2_q2_dnu += dq2_dnu(mpN_q2) * _theta_diff;
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        ds2_q2_dcs[npN_c] += dq2_dcs(npN_c)(mpN_q2) * _theta_diff;
      }
    } else {
      s2_planet += q2(mpN_q2) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);

      ds2_q2_dd += dq2_dd(mpN_q2) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
      ds2_q2_dnu += dq2_dnu(mpN_q2) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        ds2_q2_dcs[npN_c] += dq2_dcs(npN_c)(mpN_q2) / (1.i * (1. * m))
                             * (eim_theta_j_p1 - eim_theta_j);
      }
    }
    if (theta_type_j == intersections::planet) {
      ds2_theta_j_dx += q2(mpN_q2) * -eim_theta_j;
      ds2_theta_j_p1_dx += q2(mpN_q2) * eim_theta_j_p1;
    }
  }
  m_s0 += fractions::one_half * s0_planet.real();
  m_s2 += fractions::one_quarter * s2_planet.real();

  m_ds0_dd += fractions::one_half * (
    ds0_q0_dd + ds0_theta_j_dx * m_dthetas_dd[_j]
    + ds0_theta_j_p1_dx * m_dthetas_dd[_j + 1]).real();
  m_ds2_dd += fractions::one_quarter * (
    ds2_q2_dd + ds2_theta_j_dx * m_dthetas_dd[_j]
    + ds2_theta_j_p1_dx * m_dthetas_dd[_j + 1]).real();

  m_ds0_dnu += fractions::one_half * (
    ds0_q0_dnu + ds0_theta_j_dx * m_dthetas_dnu[_j]
    + ds0_theta_j_p1_dx * m_dthetas_dnu[_j + 1]).real();
  m_ds2_dnu += fractions::one_quarter * (
    ds2_q2_dnu + ds2_theta_j_dx * m_dthetas_dnu[_j]
    + ds2_theta_j_p1_dx * m_dthetas_dnu[_j + 1]).real();

  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    int npN_c = n + m_N_c;
    m_ds0_dcs(npN_c) += fractions::one_half * (
      ds0_q0_dcs[npN_c] + ds0_theta_j_dx * m_dthetas_dcs[npN_c][_j]
      + ds0_theta_j_p1_dx * m_dthetas_dcs[npN_c][_j + 1]);
    m_ds2_dcs(npN_c) += fractions::one_quarter * (
      ds2_q2_dcs[npN_c] + ds2_theta_j_dx * m_dthetas_dcs[npN_c][_j]
      + ds2_theta_j_p1_dx * m_dthetas_dcs[npN_c][_j + 1]);
  }
}


void FluxDerivatives::numerical_odd_terms(
    int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
    const double d, const double nu) {

  double s1_planet = 0.;
  double half_theta_range = (_theta_j_p1 - _theta_j) / 2.;

  double ds1_zeta_zp_dd = 0., ds1_eta_dd = 0.,
         ds1_theta_j_dd = 0., ds1_theta_j_p1_dd = 0.,
         ds1_zeta_zp_dnu = 0., ds1_eta_dnu = 0.,
         ds1_theta_j_dnu = 0., ds1_theta_j_p1_dnu = 0.,
         ds1_zeta_zp_t_theta_j_dx = 0., ds1_zeta_zp_t_theta_j_p1_dx = 0.,
         ds1_zeta_zp_r_t_theta_j_dx = 0., ds1_zeta_zp_r_t_theta_j_p1_dx = 0.,
         ds1_eta_r_t_theta_j_dx = 0., ds1_eta_r_t_theta_j_p1_dx = 0.,
         ds1_eta_t_theta_j_dx = 0., ds1_eta_t_theta_j_p1_dx = 0.,
         ds1_eta_rdash_t_theta_j_dx = 0., ds1_eta_rdash_t_theta_j_p1_dx = 0.;
  std::complex<double> ds1_zeta_zp_rp_dcs[m_n_rs], ds1_eta_rp_dcs[m_n_rs],
                       ds1_eta_rpdash_dcs[m_n_rs];

  if (m_ld_law == limb_darkening::quadratic) {
    // Limb-darkening half-integer and odd terms n=1, using
    // Gauss-legendre quad.
    for (int k = 0; k < m_N_l; k++) {

      // Rescale legendre root.
      double t_k = half_theta_range * (m_l_roots[k] + 1.) + _theta_j;

      // Evaluate integrand at t_k.
      double rp_tk = this->rp_theta(t_k);
      double rp_tks = rp_tk * rp_tk;
      double drp_dtk = this->drp_dtheta(t_k);
      double sintkmnu = std::sin(t_k - nu);
      double costkmnu = std::cos(t_k - nu);
      double d_rp_costkmnu = d * rp_tk * costkmnu;
      double d_drpdtheta_sintkmnu = d * drp_dtk * sintkmnu;
      double zp_tks = m_omdd - rp_tks + 2. * d_rp_costkmnu;
      double zp_tk = std::sqrt(zp_tks);
      double zeta = (1. - zp_tks * zp_tk) / (3. * (1. - zp_tks));
      double eta = rp_tks - d_rp_costkmnu - d_drpdtheta_sintkmnu;
      s1_planet += zeta * eta * m_l_weights[k];

      double opzp = 1 + zp_tk;
      double ds1_dzeta = eta * m_l_weights[k];
      double dzeta_dzp = fractions::one_third * (1. - 1. / (opzp * opzp));
      double dzp_dd = (rp_tk * costkmnu - d) / zp_tk;
      double dzp_dnu = d * rp_tk * sintkmnu / zp_tk;
      double dzp_drp = (d * costkmnu - rp_tk) / zp_tk;

      double ds1_deta = zeta * m_l_weights[k];
      double deta_dd = -rp_tk * costkmnu - drp_dtk * sintkmnu;
      double deta_dnu = -d * rp_tk * sintkmnu + d * drp_dtk * costkmnu;
      double deta_drp = 2. * rp_tk - d * costkmnu;
      double deta_drdash = -d * sintkmnu;

      ds1_zeta_zp_dd += ds1_dzeta * dzeta_dzp * dzp_dd;
      ds1_eta_dd += ds1_deta * deta_dd;

      ds1_zeta_zp_dnu += ds1_dzeta * dzeta_dzp * dzp_dnu;
      ds1_eta_dnu += ds1_deta * deta_dnu;

      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        std::complex<double> drp_dcn = std::exp((1. * n) * 1.i * t_k);
        std::complex<double> drpdash_dcn = (1. * n) * 1.i
                                            * std::exp((1. * n) * 1.i * t_k);
        ds1_zeta_zp_rp_dcs[npN_c] += ds1_dzeta * dzeta_dzp * dzp_drp * drp_dcn;
        ds1_eta_rp_dcs[npN_c] += ds1_deta * deta_drp * drp_dcn;
        ds1_eta_rpdash_dcs[npN_c] += ds1_deta * deta_drdash * drpdash_dcn;
      }

      if (theta_type_j == intersections::planet) {

        double dtk_dtheta_j = 1. - fractions::one_half * (m_l_roots[k] + 1.);
        double dtk_dtheta_j_p1 = fractions::one_half * (m_l_roots[k] + 1.);

        double dzp_dtk = -d * rp_tk * sintkmnu / zp_tk;
        double ds1_zeta_dzp = ds1_dzeta * dzeta_dzp;
        double ds1_zeta_zp_dtk = ds1_zeta_dzp * dzp_dtk;
        double ds1_zeta_zp_r_dtk = ds1_zeta_dzp * dzp_drp * drp_dtk;

        ds1_zeta_zp_t_theta_j_dx += ds1_zeta_zp_dtk * dtk_dtheta_j;
        ds1_zeta_zp_t_theta_j_p1_dx += ds1_zeta_zp_dtk * dtk_dtheta_j_p1;
        ds1_zeta_zp_r_t_theta_j_dx += ds1_zeta_zp_r_dtk * dtk_dtheta_j;
        ds1_zeta_zp_r_t_theta_j_p1_dx += ds1_zeta_zp_r_dtk * dtk_dtheta_j_p1;

        double drdash_dtk = this->d2rp_dtheta2(t_k);
        double deta_dtk = d * (rp_tk * sintkmnu - drp_dtk * costkmnu);
        double ds1_eta_dtk = ds1_deta * deta_dtk;
        double ds1_eta_rp_dtk = ds1_deta * deta_drp * drp_dtk;
        double ds1_eta_rdash_dtk = ds1_deta * deta_drdash * drdash_dtk;

        ds1_eta_t_theta_j_dx += ds1_eta_dtk * dtk_dtheta_j;
        ds1_eta_t_theta_j_p1_dx += ds1_eta_dtk * dtk_dtheta_j_p1;
        ds1_eta_r_t_theta_j_dx += ds1_eta_rp_dtk * dtk_dtheta_j;
        ds1_eta_r_t_theta_j_p1_dx += ds1_eta_rp_dtk * dtk_dtheta_j_p1;
        ds1_eta_rdash_t_theta_j_dx += ds1_eta_rdash_dtk * dtk_dtheta_j;
        ds1_eta_rdash_t_theta_j_p1_dx += ds1_eta_rdash_dtk * dtk_dtheta_j_p1;
      }
    }
    m_s1 += half_theta_range * s1_planet;

    ds1_theta_j_dd += -fractions::one_half * s1_planet * m_dthetas_dd[_j];
    ds1_theta_j_p1_dd += fractions::one_half * s1_planet
                         * m_dthetas_dd[_j + 1];

    ds1_theta_j_dnu += -fractions::one_half * s1_planet * m_dthetas_dnu[_j];
    ds1_theta_j_p1_dnu += fractions::one_half * s1_planet
                          * m_dthetas_dnu[_j + 1];

    m_ds1_dd += ds1_theta_j_dd + ds1_theta_j_p1_dd
              + half_theta_range * (
      ds1_zeta_zp_dd + ds1_eta_dd
      + m_dthetas_dd[_j] * (ds1_zeta_zp_t_theta_j_dx
                          + ds1_zeta_zp_r_t_theta_j_dx
                          + ds1_eta_t_theta_j_dx
                          + ds1_eta_r_t_theta_j_dx
                          + ds1_eta_rdash_t_theta_j_dx)
      + m_dthetas_dd[_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                              + ds1_zeta_zp_r_t_theta_j_p1_dx
                              + ds1_eta_t_theta_j_p1_dx
                              + ds1_eta_r_t_theta_j_p1_dx
                              + ds1_eta_rdash_t_theta_j_p1_dx));

    m_ds1_dnu += ds1_theta_j_dnu + ds1_theta_j_p1_dnu
               + half_theta_range * (
      ds1_zeta_zp_dnu + ds1_eta_dnu
      + m_dthetas_dnu[_j] * (ds1_zeta_zp_t_theta_j_dx
                          + ds1_zeta_zp_r_t_theta_j_dx
                          + ds1_eta_t_theta_j_dx
                          + ds1_eta_r_t_theta_j_dx
                          + ds1_eta_rdash_t_theta_j_dx)
      + m_dthetas_dnu[_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                              + ds1_zeta_zp_r_t_theta_j_p1_dx
                              + ds1_eta_t_theta_j_p1_dx
                              + ds1_eta_r_t_theta_j_p1_dx
                              + ds1_eta_rdash_t_theta_j_p1_dx));
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      std::complex<double> ds1_theta_j_dcn = -fractions::one_half * s1_planet
                                             * m_dthetas_dcs[npN_c][_j];
      std::complex<double> ds1_theta_j_p1_dcn = fractions::one_half * s1_planet
                                                * m_dthetas_dcs[npN_c][_j + 1];

      m_ds1_dcs(npN_c) += ds1_theta_j_dcn + ds1_theta_j_p1_dcn
                          + half_theta_range * (
        ds1_zeta_zp_rp_dcs[npN_c] + ds1_eta_rp_dcs[npN_c]
        + ds1_eta_rpdash_dcs[npN_c]
        + m_dthetas_dcs[npN_c][_j] * (ds1_zeta_zp_t_theta_j_dx
                                    + ds1_zeta_zp_r_t_theta_j_dx
                                    + ds1_eta_t_theta_j_dx
                                    + ds1_eta_r_t_theta_j_dx
                                    + ds1_eta_rdash_t_theta_j_dx)
        + m_dthetas_dcs[npN_c][_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                                        + ds1_zeta_zp_r_t_theta_j_p1_dx
                                        + ds1_eta_t_theta_j_p1_dx
                                        + ds1_eta_r_t_theta_j_p1_dx
                                        + ds1_eta_rdash_t_theta_j_p1_dx));
    }
  } else {
    double s12_planet = 0.;
    double s32_planet = 0.;

    double ds12_zeta_zp_dd = 0., ds12_zeta_zp_dnu = 0.,
           ds12_zeta_zp_t_theta_j_dx = 0., ds12_zeta_zp_t_theta_j_p1_dx = 0.,
           ds12_zeta_zp_r_t_theta_j_dx = 0., ds12_zeta_zp_r_t_theta_j_p1_dx = 0.,
           ds12_eta_dd = 0., ds1_eta_dd = 0., ds32_eta_dd = 0.,
           ds12_theta_j_dd = 0., ds12_theta_j_p1_dd = 0., ds1_theta_j_dd = 0.,
           ds1_theta_j_p1_dd = 0., ds32_theta_j_dd = 0., ds32_theta_j_p1_dd = 0.,
           ds12_eta_dnu = 0., ds1_eta_dnu = 0., ds32_eta_dnu = 0.,
           ds12_theta_j_dnu = 0., ds12_theta_j_p1_dnu = 0., ds1_theta_j_dnu = 0.,
           ds1_theta_j_p1_dnu = 0., ds32_theta_j_dnu = 0., ds32_theta_j_p1_dnu = 0.,
           ds12_eta_r_t_theta_j_dx = 0., ds12_eta_r_t_theta_j_p1_dx = 0.,
           ds12_eta_t_theta_j_dx = 0., ds12_eta_t_theta_j_p1_dx = 0.,
           ds12_eta_rdash_t_theta_j_dx = 0., ds12_eta_rdash_t_theta_j_p1_dx = 0.,
           ds1_eta_r_t_theta_j_dx = 0., ds1_eta_r_t_theta_j_p1_dx = 0.,
           ds1_eta_t_theta_j_dx = 0., ds1_eta_t_theta_j_p1_dx = 0.,
           ds1_eta_rdash_t_theta_j_dx = 0., ds1_eta_rdash_t_theta_j_p1_dx = 0.,
           ds32_zeta_zp_dd = 0., ds32_zeta_zp_dnu = 0.,
           ds32_zeta_zp_t_theta_j_dx = 0., ds32_zeta_zp_t_theta_j_p1_dx = 0.,
           ds32_zeta_zp_r_t_theta_j_dx = 0., ds32_zeta_zp_r_t_theta_j_p1_dx = 0.,
           ds32_eta_r_t_theta_j_dx = 0., ds32_eta_r_t_theta_j_p1_dx = 0.,
           ds32_eta_t_theta_j_dx = 0., ds32_eta_t_theta_j_p1_dx = 0.,
           ds32_eta_rdash_t_theta_j_dx = 0., ds32_eta_rdash_t_theta_j_p1_dx = 0.;
    std::complex<double> ds12_zeta_zp_rp_dcs[m_n_rs], ds12_eta_rp_dcs[m_n_rs],
                         ds12_eta_rpdash_dcs[m_n_rs], ds1_zeta_zp_rp_dcs[m_n_rs],
                         ds1_eta_rp_dcs[m_n_rs], ds1_eta_rpdash_dcs[m_n_rs],
                         ds32_zeta_zp_rp_dcs[m_n_rs], ds32_eta_rp_dcs[m_n_rs],
                         ds32_eta_rpdash_dcs[m_n_rs];

    // Limb-darkening half-integer and odd terms n=1/2, 1, 3/2, using
    // Gauss-legendre quad.
    for (int k = 0; k < m_N_l; k++) {

      // Rescale legendre root.
      double t_k = half_theta_range * (m_l_roots[k] + 1.) + _theta_j;

      // Evaluate integrand at t_k.
      double rp_tk = this->rp_theta(t_k);
      double rp_tks = rp_tk * rp_tk;
      double drp_dtk = this->drp_dtheta(t_k);
      double sintkmnu = std::sin(t_k - nu);
      double costkmnu = std::cos(t_k - nu);
      double d_rp_costkmnu = d * rp_tk * costkmnu;
      double d_drpdtheta_sintkmnu = d * drp_dtk * sintkmnu;
      double zp_tks = m_omdd - rp_tks + 2. * d_rp_costkmnu;
      double zp_tk = std::sqrt(zp_tks);
      double omzp_tks = 1 - zp_tks;
      double zeta12 = (1. - std::pow(zp_tk, fractions::five_halves))
                       / (fractions::five_halves * omzp_tks);
      double zeta = (1. - zp_tks * zp_tk) / (3. * omzp_tks);
      double zeta32 = (1. - std::pow(zp_tk, fractions::seven_halves))
                       / (fractions::seven_halves * omzp_tks);
      double eta = rp_tks - d_rp_costkmnu - d_drpdtheta_sintkmnu;
      double eta_w = eta * m_l_weights[k];
      s12_planet += zeta12 * eta_w;
      s1_planet += zeta * eta_w;
      s32_planet += zeta32 * eta_w;

      double opzp = 1 + zp_tk;
      double ds1_dzeta = eta * m_l_weights[k];
      double dzeta12_dzp = fractions::one_fifth * (
        4. * zp_tk - 5. * std::pow(zp_tk, fractions::three_halves)
        + std::pow(zp_tk, fractions::seven_halves)) / (omzp_tks * omzp_tks);
      double dzeta1_dzp = fractions::one_third * (1. - 1. / (opzp * opzp));
      double dzeta32_dzp = zp_tk * (
        2. + fractions::three_halves * std::pow(zp_tk, fractions::seven_halves)
        - fractions::seven_halves * std::pow(zp_tk, fractions::three_halves))
        / (fractions::seven_halves * omzp_tks * omzp_tks);
      double dzp_dd = (rp_tk * costkmnu - d) / zp_tk;
      double dzp_dnu = d * rp_tk * sintkmnu / zp_tk;
      double dzp_drp = (d * costkmnu - rp_tk) / zp_tk;

      double ds12_deta = zeta12 * m_l_weights[k];
      double ds1_deta = zeta * m_l_weights[k];
      double ds32_deta = zeta32 * m_l_weights[k];
      double deta_dd = -rp_tk * costkmnu - drp_dtk * sintkmnu;
      double deta_dnu = -d * rp_tk * sintkmnu + d * drp_dtk * costkmnu;
      double deta_drp = 2. * rp_tk - d * costkmnu;
      double deta_drdash = -d * sintkmnu;

      ds12_zeta_zp_dd += ds1_dzeta * dzeta12_dzp * dzp_dd;
      ds1_zeta_zp_dd += ds1_dzeta * dzeta1_dzp * dzp_dd;
      ds32_zeta_zp_dd += ds1_dzeta * dzeta32_dzp * dzp_dd;
      ds12_eta_dd += ds12_deta * deta_dd;
      ds1_eta_dd += ds1_deta * deta_dd;
      ds32_eta_dd += ds32_deta * deta_dd;

      ds12_zeta_zp_dnu += ds1_dzeta * dzeta12_dzp * dzp_dnu;
      ds1_zeta_zp_dnu += ds1_dzeta * dzeta1_dzp * dzp_dnu;
      ds32_zeta_zp_dnu += ds1_dzeta * dzeta32_dzp * dzp_dnu;
      ds12_eta_dnu += ds12_deta * deta_dnu;
      ds1_eta_dnu += ds1_deta * deta_dnu;
      ds32_eta_dnu += ds32_deta * deta_dnu;

      for (int n = -m_N_c; n < m_N_c + 1; n++) {
        int npN_c = n + m_N_c;
        std::complex<double> drp_dcn = std::exp((1. * n) * 1.i * t_k);
        std::complex<double> drpdash_dcn = (1. * n) * 1.i
                                           * std::exp((1. * n) * 1.i * t_k);
        ds12_zeta_zp_rp_dcs[npN_c] += ds1_dzeta * dzeta12_dzp
                                      * dzp_drp * drp_dcn;
        ds12_eta_rp_dcs[npN_c] += ds12_deta * deta_drp * drp_dcn;
        ds12_eta_rpdash_dcs[npN_c] += ds12_deta * deta_drdash * drpdash_dcn;
        ds1_zeta_zp_rp_dcs[npN_c] += ds1_dzeta * dzeta1_dzp
                                     * dzp_drp * drp_dcn;
        ds1_eta_rp_dcs[npN_c] += ds1_deta * deta_drp * drp_dcn;
        ds1_eta_rpdash_dcs[npN_c] += ds1_deta * deta_drdash * drpdash_dcn;
        ds32_zeta_zp_rp_dcs[npN_c] += ds1_dzeta * dzeta32_dzp
                                      * dzp_drp * drp_dcn;
        ds32_eta_rp_dcs[npN_c] += ds32_deta * deta_drp * drp_dcn;
        ds32_eta_rpdash_dcs[npN_c] += ds32_deta * deta_drdash * drpdash_dcn;
      }

      if (theta_type_j == intersections::planet) {

        double dtk_dtheta_j = 1. - fractions::one_half * (m_l_roots[k] + 1.);
        double dtk_dtheta_j_p1 = fractions::one_half * (m_l_roots[k] + 1.);

        double dzp_dtk = -d * rp_tk * sintkmnu / zp_tk;
        double ds12_zeta_dzp = ds1_dzeta * dzeta12_dzp;
        double ds12_zeta_zp_dtk = ds12_zeta_dzp * dzp_dtk;
        double ds12_zeta_zp_r_dtk = ds12_zeta_dzp * dzp_drp * drp_dtk;
        double ds1_zeta_dzp = ds1_dzeta * dzeta1_dzp;
        double ds1_zeta_zp_dtk = ds1_zeta_dzp * dzp_dtk;
        double ds1_zeta_zp_r_dtk = ds1_zeta_dzp * dzp_drp * drp_dtk;
        double ds32_zeta_dzp = ds1_dzeta * dzeta32_dzp;
        double ds32_zeta_zp_dtk = ds32_zeta_dzp * dzp_dtk;
        double ds32_zeta_zp_r_dtk = ds32_zeta_dzp * dzp_drp * drp_dtk;

        ds12_zeta_zp_t_theta_j_dx += ds12_zeta_zp_dtk * dtk_dtheta_j;
        ds12_zeta_zp_t_theta_j_p1_dx += ds12_zeta_zp_dtk * dtk_dtheta_j_p1;
        ds12_zeta_zp_r_t_theta_j_dx += ds12_zeta_zp_r_dtk * dtk_dtheta_j;
        ds12_zeta_zp_r_t_theta_j_p1_dx += ds12_zeta_zp_r_dtk * dtk_dtheta_j_p1;
        ds1_zeta_zp_t_theta_j_dx += ds1_zeta_zp_dtk * dtk_dtheta_j;
        ds1_zeta_zp_t_theta_j_p1_dx += ds1_zeta_zp_dtk * dtk_dtheta_j_p1;
        ds1_zeta_zp_r_t_theta_j_dx += ds1_zeta_zp_r_dtk * dtk_dtheta_j;
        ds1_zeta_zp_r_t_theta_j_p1_dx += ds1_zeta_zp_r_dtk * dtk_dtheta_j_p1;
        ds32_zeta_zp_t_theta_j_dx += ds32_zeta_zp_dtk * dtk_dtheta_j;
        ds32_zeta_zp_t_theta_j_p1_dx += ds32_zeta_zp_dtk * dtk_dtheta_j_p1;
        ds32_zeta_zp_r_t_theta_j_dx += ds32_zeta_zp_r_dtk * dtk_dtheta_j;
        ds32_zeta_zp_r_t_theta_j_p1_dx += ds32_zeta_zp_r_dtk * dtk_dtheta_j_p1;

        double drdash_dtk = this->d2rp_dtheta2(t_k);
        double deta_dtk = d * (rp_tk * sintkmnu - drp_dtk * costkmnu);
        double ds12_eta_dtk = ds12_deta * deta_dtk;
        double ds12_eta_rp_dtk = ds12_deta * deta_drp * drp_dtk;
        double ds12_eta_rdash_dtk = ds12_deta * deta_drdash * drdash_dtk;
        double ds1_eta_dtk = ds1_deta * deta_dtk;
        double ds1_eta_rp_dtk = ds1_deta * deta_drp * drp_dtk;
        double ds1_eta_rdash_dtk = ds1_deta * deta_drdash * drdash_dtk;
        double ds32_eta_dtk = ds32_deta * deta_dtk;
        double ds32_eta_rp_dtk = ds32_deta * deta_drp * drp_dtk;
        double ds32_eta_rdash_dtk = ds32_deta * deta_drdash * drdash_dtk;

        ds12_eta_t_theta_j_dx += ds12_eta_dtk * dtk_dtheta_j;
        ds12_eta_t_theta_j_p1_dx += ds12_eta_dtk * dtk_dtheta_j_p1;
        ds12_eta_r_t_theta_j_dx += ds12_eta_rp_dtk * dtk_dtheta_j;
        ds12_eta_r_t_theta_j_p1_dx += ds12_eta_rp_dtk * dtk_dtheta_j_p1;
        ds12_eta_rdash_t_theta_j_dx += ds12_eta_rdash_dtk * dtk_dtheta_j;
        ds12_eta_rdash_t_theta_j_p1_dx += ds12_eta_rdash_dtk * dtk_dtheta_j_p1;
        ds1_eta_t_theta_j_dx += ds1_eta_dtk * dtk_dtheta_j;
        ds1_eta_t_theta_j_p1_dx += ds1_eta_dtk * dtk_dtheta_j_p1;
        ds1_eta_r_t_theta_j_dx += ds1_eta_rp_dtk * dtk_dtheta_j;
        ds1_eta_r_t_theta_j_p1_dx += ds1_eta_rp_dtk * dtk_dtheta_j_p1;
        ds1_eta_rdash_t_theta_j_dx += ds1_eta_rdash_dtk * dtk_dtheta_j;
        ds1_eta_rdash_t_theta_j_p1_dx += ds1_eta_rdash_dtk * dtk_dtheta_j_p1;
        ds32_eta_t_theta_j_dx += ds32_eta_dtk * dtk_dtheta_j;
        ds32_eta_t_theta_j_p1_dx += ds32_eta_dtk * dtk_dtheta_j_p1;
        ds32_eta_r_t_theta_j_dx += ds32_eta_rp_dtk * dtk_dtheta_j;
        ds32_eta_r_t_theta_j_p1_dx += ds32_eta_rp_dtk * dtk_dtheta_j_p1;
        ds32_eta_rdash_t_theta_j_dx += ds32_eta_rdash_dtk * dtk_dtheta_j;
        ds32_eta_rdash_t_theta_j_p1_dx += ds32_eta_rdash_dtk * dtk_dtheta_j_p1;
      }
    }
    m_s12 += half_theta_range * s12_planet;
    m_s1 += half_theta_range * s1_planet;
    m_s32 += half_theta_range * s32_planet;

    ds12_theta_j_dd += -fractions::one_half * s12_planet * m_dthetas_dd[_j];
    ds12_theta_j_p1_dd += fractions::one_half * s12_planet
                              * m_dthetas_dd[_j + 1];
    ds1_theta_j_dd += -fractions::one_half * s1_planet * m_dthetas_dd[_j];
    ds1_theta_j_p1_dd += fractions::one_half * s1_planet
                             * m_dthetas_dd[_j + 1];
    ds32_theta_j_dd += -fractions::one_half * s32_planet * m_dthetas_dd[_j];
    ds32_theta_j_p1_dd += fractions::one_half * s32_planet
                              * m_dthetas_dd[_j + 1];

    ds12_theta_j_dnu += -fractions::one_half * s12_planet * m_dthetas_dnu[_j];
    ds12_theta_j_p1_dnu += fractions::one_half * s12_planet
                               * m_dthetas_dnu[_j + 1];
    ds1_theta_j_dnu += -fractions::one_half * s1_planet * m_dthetas_dnu[_j];
    ds1_theta_j_p1_dnu += fractions::one_half * s1_planet
                              * m_dthetas_dnu[_j + 1];
    ds32_theta_j_dnu += -fractions::one_half * s32_planet * m_dthetas_dnu[_j];
    ds32_theta_j_p1_dnu += fractions::one_half * s32_planet
                               * m_dthetas_dnu[_j + 1];

    m_ds12_dd += ds12_theta_j_dd + ds12_theta_j_p1_dd
              + half_theta_range * (
      ds12_zeta_zp_dd + ds12_eta_dd
      + m_dthetas_dd[_j] * (ds12_zeta_zp_t_theta_j_dx
                          + ds12_zeta_zp_r_t_theta_j_dx
                          + ds12_eta_t_theta_j_dx
                          + ds12_eta_r_t_theta_j_dx
                          + ds12_eta_rdash_t_theta_j_dx)
      + m_dthetas_dd[_j + 1] * (ds12_zeta_zp_t_theta_j_p1_dx
                              + ds12_zeta_zp_r_t_theta_j_p1_dx
                              + ds12_eta_t_theta_j_p1_dx
                              + ds12_eta_r_t_theta_j_p1_dx
                              + ds12_eta_rdash_t_theta_j_p1_dx));
    m_ds1_dd += ds1_theta_j_dd + ds1_theta_j_p1_dd
              + half_theta_range * (
      ds1_zeta_zp_dd + ds1_eta_dd
      + m_dthetas_dd[_j] * (ds1_zeta_zp_t_theta_j_dx
                          + ds1_zeta_zp_r_t_theta_j_dx
                          + ds1_eta_t_theta_j_dx
                          + ds1_eta_r_t_theta_j_dx
                          + ds1_eta_rdash_t_theta_j_dx)
      + m_dthetas_dd[_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                              + ds1_zeta_zp_r_t_theta_j_p1_dx
                              + ds1_eta_t_theta_j_p1_dx
                              + ds1_eta_r_t_theta_j_p1_dx
                              + ds1_eta_rdash_t_theta_j_p1_dx));
    m_ds32_dd += ds32_theta_j_dd + ds32_theta_j_p1_dd
              + half_theta_range * (
      ds32_zeta_zp_dd + ds32_eta_dd
      + m_dthetas_dd[_j] * (ds32_zeta_zp_t_theta_j_dx
                          + ds32_zeta_zp_r_t_theta_j_dx
                          + ds32_eta_t_theta_j_dx
                          + ds32_eta_r_t_theta_j_dx
                          + ds32_eta_rdash_t_theta_j_dx)
      + m_dthetas_dd[_j + 1] * (ds32_zeta_zp_t_theta_j_p1_dx
                              + ds32_zeta_zp_r_t_theta_j_p1_dx
                              + ds32_eta_t_theta_j_p1_dx
                              + ds32_eta_r_t_theta_j_p1_dx
                              + ds32_eta_rdash_t_theta_j_p1_dx));

    m_ds12_dnu += ds12_theta_j_dnu + ds12_theta_j_p1_dnu
              + half_theta_range * (
      ds12_zeta_zp_dnu + ds12_eta_dnu
      + m_dthetas_dnu[_j] * (ds12_zeta_zp_t_theta_j_dx
                          + ds12_zeta_zp_r_t_theta_j_dx
                          + ds12_eta_t_theta_j_dx
                          + ds12_eta_r_t_theta_j_dx
                          + ds12_eta_rdash_t_theta_j_dx)
      + m_dthetas_dnu[_j + 1] * (ds12_zeta_zp_t_theta_j_p1_dx
                              + ds12_zeta_zp_r_t_theta_j_p1_dx
                              + ds12_eta_t_theta_j_p1_dx
                              + ds12_eta_r_t_theta_j_p1_dx
                              + ds12_eta_rdash_t_theta_j_p1_dx));
    m_ds1_dnu += ds1_theta_j_dnu + ds1_theta_j_p1_dnu
              + half_theta_range * (
      ds1_zeta_zp_dnu + ds1_eta_dnu
      + m_dthetas_dnu[_j] * (ds1_zeta_zp_t_theta_j_dx
                          + ds1_zeta_zp_r_t_theta_j_dx
                          + ds1_eta_t_theta_j_dx
                          + ds1_eta_r_t_theta_j_dx
                          + ds1_eta_rdash_t_theta_j_dx)
      + m_dthetas_dnu[_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                              + ds1_zeta_zp_r_t_theta_j_p1_dx
                              + ds1_eta_t_theta_j_p1_dx
                              + ds1_eta_r_t_theta_j_p1_dx
                              + ds1_eta_rdash_t_theta_j_p1_dx));
    m_ds32_dnu += ds32_theta_j_dnu + ds32_theta_j_p1_dnu
              + half_theta_range * (
      ds32_zeta_zp_dnu + ds32_eta_dnu
      + m_dthetas_dnu[_j] * (ds32_zeta_zp_t_theta_j_dx
                          + ds32_zeta_zp_r_t_theta_j_dx
                          + ds32_eta_t_theta_j_dx
                          + ds32_eta_r_t_theta_j_dx
                          + ds32_eta_rdash_t_theta_j_dx)
      + m_dthetas_dnu[_j + 1] * (ds32_zeta_zp_t_theta_j_p1_dx
                              + ds32_zeta_zp_r_t_theta_j_p1_dx
                              + ds32_eta_t_theta_j_p1_dx
                              + ds32_eta_r_t_theta_j_p1_dx
                              + ds32_eta_rdash_t_theta_j_p1_dx));
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      int npN_c = n + m_N_c;
      std::complex<double> ds12_theta_j_dcn =
        -fractions::one_half * s12_planet * m_dthetas_dcs[npN_c][_j];
      std::complex<double> ds12_theta_j_p1_dcn =
        fractions::one_half * s12_planet * m_dthetas_dcs[npN_c][_j + 1];
      std::complex<double> ds1_theta_j_dcn =
        -fractions::one_half * s1_planet * m_dthetas_dcs[npN_c][_j];
      std::complex<double> ds1_theta_j_p1_dcn =
        fractions::one_half * s1_planet * m_dthetas_dcs[npN_c][_j + 1];
      std::complex<double> ds32_theta_j_dcn =
        -fractions::one_half * s32_planet * m_dthetas_dcs[npN_c][_j];
      std::complex<double> ds32_theta_j_p1_dcn =
        fractions::one_half * s32_planet * m_dthetas_dcs[npN_c][_j + 1];

      m_ds12_dcs(npN_c) += ds12_theta_j_dcn + ds12_theta_j_p1_dcn
                          + half_theta_range * (
        ds12_zeta_zp_rp_dcs[npN_c] + ds12_eta_rp_dcs[npN_c]
        + ds12_eta_rpdash_dcs[npN_c]
        + m_dthetas_dcs[npN_c][_j] * (ds12_zeta_zp_t_theta_j_dx
                                    + ds12_zeta_zp_r_t_theta_j_dx
                                    + ds12_eta_t_theta_j_dx
                                    + ds12_eta_r_t_theta_j_dx
                                    + ds12_eta_rdash_t_theta_j_dx)
        + m_dthetas_dcs[npN_c][_j + 1] * (ds12_zeta_zp_t_theta_j_p1_dx
                                        + ds12_zeta_zp_r_t_theta_j_p1_dx
                                        + ds12_eta_t_theta_j_p1_dx
                                        + ds12_eta_r_t_theta_j_p1_dx
                                        + ds12_eta_rdash_t_theta_j_p1_dx));
      m_ds1_dcs(npN_c) += ds1_theta_j_dcn + ds1_theta_j_p1_dcn
                          + half_theta_range * (
        ds1_zeta_zp_rp_dcs[npN_c] + ds1_eta_rp_dcs[npN_c]
        + ds1_eta_rpdash_dcs[npN_c]
        + m_dthetas_dcs[npN_c][_j] * (ds1_zeta_zp_t_theta_j_dx
                                    + ds1_zeta_zp_r_t_theta_j_dx
                                    + ds1_eta_t_theta_j_dx
                                    + ds1_eta_r_t_theta_j_dx
                                    + ds1_eta_rdash_t_theta_j_dx)
        + m_dthetas_dcs[npN_c][_j + 1] * (ds1_zeta_zp_t_theta_j_p1_dx
                                        + ds1_zeta_zp_r_t_theta_j_p1_dx
                                        + ds1_eta_t_theta_j_p1_dx
                                        + ds1_eta_r_t_theta_j_p1_dx
                                        + ds1_eta_rdash_t_theta_j_p1_dx));
      m_ds32_dcs(npN_c) += ds32_theta_j_dcn + ds32_theta_j_p1_dcn
                          + half_theta_range * (
        ds32_zeta_zp_rp_dcs[npN_c] + ds32_eta_rp_dcs[npN_c]
        + ds32_eta_rpdash_dcs[npN_c]
        + m_dthetas_dcs[npN_c][_j] * (ds32_zeta_zp_t_theta_j_dx
                                    + ds32_zeta_zp_r_t_theta_j_dx
                                    + ds32_eta_t_theta_j_dx
                                    + ds32_eta_r_t_theta_j_dx
                                    + ds32_eta_rdash_t_theta_j_dx)
        + m_dthetas_dcs[npN_c][_j + 1] * (ds32_zeta_zp_t_theta_j_p1_dx
                                        + ds32_zeta_zp_r_t_theta_j_p1_dx
                                        + ds32_eta_t_theta_j_p1_dx
                                        + ds32_eta_r_t_theta_j_p1_dx
                                        + ds32_eta_rdash_t_theta_j_p1_dx));
    }
  }
}
