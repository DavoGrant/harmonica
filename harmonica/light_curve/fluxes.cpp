#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "fluxes.hpp"
#include "../constants/constants.hpp"

using namespace std::complex_literals;


Fluxes::Fluxes(int ld_law, double us[], int n_rs, double rs[],
               int pnl_c, int pnl_e) :
  m_ld_law(ld_law),
  m_n_rs(n_rs),
  m_precision_nl_centre(pnl_c),
  m_precision_nl_edge(pnl_e) {

  if (m_ld_law == limb_darkening::quadratic) {
    // Normalisation.
    double m_I_0_bt = (1. - us[0] / 3. - us[1] / 6.);
    m_I_0_bts = m_I_0_bt * m_I_0_bt;
    m_I_0 = 1. / (fractions::pi * m_I_0_bt);

    // Quadratic limb-darkening law.
    Eigen::Vector<double, 3> u {1., us[0], us[1]};
    Eigen::Matrix<double, 3, 3> B {{1., -1., -1.},
                                   {0., 1., 2.},
                                   {0., 0., -1.}};

    // Change to polynomial basis.
    m_p = B * u;

  } else {
    // Normalisation.
    double m_I_0_bt = (1. - us[0] / 5. - us[1] / 3.
                     - 3. * us[2] / 7. - us[3] / 2.);
    m_I_0_bts = m_I_0_bt * m_I_0_bt;
    m_I_0 = 1. / (fractions::pi * m_I_0_bt);

    // Non-linear limb-darkening law.
    Eigen::Vector<double, 5> u {1., us[0], us[1], us[2], us[3]};
    Eigen::Matrix<double, 5, 5> B {{1., -1., -1., -1., -1.},
                                   {0., 1., 0., 0., 0.},
                                   {0., 0., 1., 0., 0.},
                                   {0., 0., 0., 1., 0.},
                                   {0., 0., 0., 0., 1.}};

    // Change to polynomial basis.
    m_p = B * u;
  }

  // Convert cosine, sine to complex Fourier coefficients.
  m_N_c = (m_n_rs - 1) * fractions::one_half;
  m_c.resize(m_n_rs);
  m_c(m_N_c) = rs[0];
  for (int n = 0; n < m_N_c; n++) {
    double a_real = rs[m_n_rs - 2 - 2 * n];
    double b_imag = rs[m_n_rs - 1 - 2 * n];
    m_c(n) = (a_real + b_imag * 1.i) * fractions::one_half;
    m_c(m_n_rs - 1 - n) = (a_real - b_imag * 1.i) * fractions::one_half;
  }

  // Pre-compute max and min planet radii.
  m_min_rp = m_c(m_N_c).real();
  m_max_rp = m_c(m_N_c).real();
  if (m_N_c != 0) {
    // Build the extrema companion matrix.
    int D_shape = 2 * m_N_c;
    m_D.resize(D_shape, D_shape);
    for (int j = 1; j < D_shape + 1; j++) {
      for (int k = 1; k < D_shape + 1; k++) {
        m_D(j - 1, k - 1) = this->extrema_companion_matrix_D_jk(j, k, D_shape);
      }
    }

    // Get the extrema companion matrix roots.
    std::vector<double> theta_extrema = this->compute_real_theta_roots(
      m_D, D_shape);

    // Find the max an min radius values.
    for (int j = 0; j < theta_extrema.size(); j++) {
      double _rp = this->rp_theta(theta_extrema[j]);
      if (_rp < m_min_rp) {
        m_min_rp = _rp;
      } else if (_rp > m_max_rp) {
        m_max_rp = _rp;
      }
    }
  }

  if (m_N_c != 0) {
    // Pre-build the intersection eqn companion matrix for the terms
    // that are independent of position, d and nu.
    m_C_shape = 4 * m_N_c;
    m_C0.resize(m_C_shape, m_C_shape);
    for (int j = 1; j < m_C_shape + 1; j++) {
      for (int k = 1; k < m_C_shape + 1; k++) {
        m_C0(j - 1, k - 1) = this->intersection_companion_matrix_C_jk_base(
          j, k, m_C_shape);
      }
    }
  }

  // Pre-compute c (*) c.
  m_len_c_conv_c = 2 * m_n_rs - 1;
  m_c_conv_c = complex_convolve(m_c, m_c, m_n_rs, m_n_rs, m_len_c_conv_c);

  // Pre-compute Delta element-wise multiply c.
  m_Delta_ew_c = m_c;
   for (int n = -m_N_c; n < m_N_c + 1; n++) {
    m_Delta_ew_c(n + m_N_c) *= (1. * n) * 1.i;
  }

  // Pre-compute beta_sin/cos base vectors.
  m_len_beta_conv_c = 3 + m_n_rs - 1;
  m_beta_sin0 << -fractions::one_half, 0., -fractions::one_half;
  m_beta_cos0 << fractions::one_half, 0., fractions::one_half;

  // Pre-compute conv sizes.
  m_len_q_rhs = std::max(m_len_c_conv_c, m_len_beta_conv_c);
  m_mid_q_lhs = (m_len_q_rhs - 1) / 2;
  m_len_q = 2 * m_len_q_rhs - 1;
  m_N_q0 = (m_len_q_rhs - 1) / 2;
  m_N_q2 = (m_len_q - 1) / 2;
}


double Fluxes::rp_theta(const double _theta) {
  std::complex<double> rp = 0.;
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    rp += m_c(n + m_N_c) * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


void Fluxes::transit_flux(const double d, const double z,
                          const double nu, double& out_f) {

  this->compute_solution_vector(d, z, nu);

  // Compute transit flux: alpha=I0sTp.
  if (m_ld_law == limb_darkening::quadratic) {
    m_alpha = m_I_0 * (m_s0 * m_p(0) + m_s1 * m_p(1) + m_s2 * m_p(2));
  } else {
    m_alpha = m_I_0 * (m_s0 * m_p(0) + m_s12 * m_p(1) + m_s1 * m_p(2)
                       + m_s32 * m_p(3) + m_s2 * m_p(4));
  }
  out_f = 1. - m_alpha;
}


void Fluxes::compute_solution_vector(const double d, const double z,
                                     const double nu) {

  // Reset and pre-compute some position-specific quantities.
  this->reset_intersections_integrals();
  this->select_legendre_order(d);
  this->pre_compute_psq(d, nu);

  if (z > 0.) {
    // Find planet-stellar limb intersections.
    this->find_intersections_theta(d, nu);

    // Iterate thetas in adjacent pairs around the enclosed overlap region.
    for (int j = 0; j < m_theta_type.size(); j++) {

      if (m_theta_type[j] == intersections::planet
          || m_theta_type[j] == intersections::entire_planet) {
        // Line integrals, s_n, along planet limb segment.
        this->s_planet(j, m_theta_type[j], m_theta[j], m_theta[j + 1], d, nu);

      } else if (m_theta_type[j] == intersections::star
                 || m_theta_type[j] == intersections::entire_star) {
        // Line integrals, s_n, along stellar limb segment.
        this->s_star(j, m_theta_type[j], m_theta[j], m_theta[j + 1], d, nu);

      } else {
        // Planet is beyond the stellar disc.
      }
    }
  } else {
    // Planet behind star, only eclipses.
  }
}


void Fluxes::find_intersections_theta(const double d, const double nu) {

  // Check cases where no obvious intersections, avoiding eigenvalue runtime.
  if (this->no_obvious_intersections(d, nu)) { return; }

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
  }

  if (m_theta.size() == 0) {
    // No roots, check which trivial case this configuration corresponds to.
    if (this->trivial_configuration(d, nu)) { return; }
  }

  // Sort thetas in ascending order, -pi < theta <= pi.
  std::sort(m_theta.begin(), m_theta.end());

  // To ensure theta vector spans a closed loop, 2pi in total.
  // Thus, duplicate first intersection + 2pi at the end.
  m_theta.push_back(m_theta[0] + fractions::twopi);

  // Characterise theta pairs.
  this->characterise_intersection_pairs(d, nu);
}


void Fluxes::s_planet(int _j, int theta_type_j, double _theta_j,
                      double _theta_j_p1, const double d,
                      const double nu) {

  // Compute the closed-form even terms.
  this->analytic_even_terms(_j, theta_type_j, _theta_j,
                            _theta_j_p1, d, nu);

  // Compute the numerical odd and half-integer terms.
  this->numerical_odd_terms(_j, theta_type_j, _theta_j,
                            _theta_j_p1, d, nu);
}


void Fluxes::s_star(int _j, int theta_type_j, double _theta_j,
                    double _theta_j_p1, const double d,
                    const double nu) {
  double phi_j;
  double phi_j_p1;

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
}


void Fluxes::reset_intersections_integrals() {
  // Reset intersections.
  m_theta.clear();
  m_theta_type.clear();

  // Reset integrals.
  m_s0 = 0., m_s12 = 0., m_s1 = 0., m_s32 = 0., m_s2 = 0.;
}


void Fluxes::select_legendre_order(const double d) {
  // Select regime switch: centre or edge.
  int position_switch;
  double outer_radii = m_max_rp + d;
  if (outer_radii >= 0.99) {
    position_switch = m_precision_nl_edge;
  } else {
    position_switch = m_precision_nl_centre;
  }

  // Set precision by number of legendre roots utilised.
  if (position_switch == 20) {
    // Default for centre.
    m_N_l = 20;
    m_l_roots = legendre::roots_twenty;
    m_l_weights = legendre::weights_twenty;
  } else if (position_switch == 50) {
    // Default for edge.
    m_N_l = 50;
    m_l_roots = legendre::roots_fifty;
    m_l_weights = legendre::weights_fifty;
  } else if (position_switch == 100) {
    m_N_l = 100;
    m_l_roots = legendre::roots_hundred;
    m_l_weights = legendre::weights_hundred;
  } else if (position_switch == 200) {
    m_N_l = 200;
    m_l_roots = legendre::roots_two_hundred;
    m_l_weights = legendre::weights_two_hundred;
  } else if (position_switch == 500) {
    m_N_l = 500;
    m_l_roots = legendre::roots_five_hundred;
    m_l_weights = legendre::weights_five_hundred;
  } else if (position_switch == 10) {
    m_N_l = 10;
    m_l_roots = legendre::roots_ten;
    m_l_weights = legendre::weights_ten;
  } else {
    // Fallback.
    m_N_l = 500;
    m_l_roots = legendre::roots_five_hundred;
    m_l_weights = legendre::weights_five_hundred;
  }
}


void Fluxes::pre_compute_psq(const double d, const double nu) {
  m_td = 2. * d;
  m_dd = d * d;
  m_omdd = 1. - m_dd;
  m_expinu = std::exp(1.i * nu);
  m_expminu = std::exp(-1.i * nu);
  m_d_expinu = d * std::exp(1.i * nu);
  m_d_expminu = d * std::exp(-1.i * nu);
}


double Fluxes::rs_theta(const double d, double dcos_thetamnu,
                        int plus_solution) {
  if (d <= 1.) {
    return dcos_thetamnu + std::sqrt(dcos_thetamnu * dcos_thetamnu
                                     - m_dd + 1.);
  } else {
    if (plus_solution == 1) {
      // r_s+ solution.
      return dcos_thetamnu + std::sqrt(dcos_thetamnu * dcos_thetamnu
                                       - m_dd + 1.);
    } else {
      // r_s- solution.
      return dcos_thetamnu - std::sqrt(dcos_thetamnu * dcos_thetamnu
                                       - m_dd + 1.);
    }
  }
}


double Fluxes::drp_dtheta(double _theta) {
  std::complex<double> rp = 0.;
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    rp += 1.i * (1. * n) * m_c(n + m_N_c) * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


double Fluxes::d2rp_dtheta2(double _theta) {
  std::complex<double> rp = 0.;
  for (int n = -m_N_c; n < m_N_c + 1; n++) {
    rp += (1.i * (1. * n)) * (1.i * (1. * n)) * m_c(n + m_N_c)
           * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


std::complex<double> Fluxes::extrema_companion_matrix_D_jk(
    int j, int k, int shape) {

  // NB. matrix elements are one-indexed.
  // Also, c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> moo_denom = -1. / (1. * m_N_c * m_c(shape));
  if (k == shape) {
    return (j - m_N_c - 1.) * m_c(j - 1) * moo_denom;
  } else {
    if (j == k + 1) {
      return 1.;
    } else {
      return 0.;
    }
  }
}


std::complex<double> Fluxes::intersection_companion_matrix_C_jk_base(
  int j, int k, int shape) {
  // NB. matrix elements are one-indexed.
  // Also, c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  if (k == shape) {
    return this->intersection_polynomial_coefficients_h_j_base(j - 1);
  } else {
    if (j == k + 1) {
      return 1.;
    } else {
      return 0.;
    }
  }
}


std::complex<double> Fluxes::intersection_polynomial_coefficients_h_j_base(
  int j) {
  // NB. verbose on purpose.
  // Also, c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> h_j_base = 0.;
  if (0 <= j && j < m_N_c - 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (m_N_c - 1 <= j && j < m_N_c + 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (j == 2 * m_N_c) {
    h_j_base -= 1.;
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (3 * m_N_c + 2 <= j && j < 4 * m_N_c + 1) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      h_j_base += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  }
  return h_j_base;
}


std::complex<double> Fluxes::intersection_polynomial_coefficients_h_j_update(
  int j) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> h_j_update = 0.;
  if (m_N_c - 1 <= j && j < m_N_c + 1) {
    h_j_update -= m_d_expinu * m_c(j + 1 - m_N_c);
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    h_j_update -= m_d_expinu * m_c(j + 1 - m_N_c);
    h_j_update -= m_d_expminu * m_c(j - 1 - m_N_c);
  } else if (j == 2 * m_N_c) {
    h_j_update -= m_d_expinu * m_c(j + 1 - m_N_c);
    h_j_update -= m_d_expminu * m_c(j - 1 - m_N_c);
    h_j_update += m_dd;
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    h_j_update -= m_d_expinu * m_c(j + 1 - m_N_c);
    h_j_update -= m_d_expminu * m_c(j - 1 - m_N_c);
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    h_j_update -= m_d_expminu * m_c(j - 1 - m_N_c);
  }
  return h_j_update;
}


std::complex<double> Fluxes::intersection_polynomial_coefficient_moo_denom(
  int j) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> h_4Nc = 0.;
  if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    h_4Nc -= m_d_expminu * m_c(j - 1 - m_N_c);
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      h_4Nc += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (3 * m_N_c + 2 <= j && j < 4 * m_N_c + 1) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      h_4Nc += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  }
  return -1. / h_4Nc;
}


std::complex<double> Fluxes::h_j(
  int j) {
  // NB. c_0 requires m_c(0 + m_N_c) as it runs -m_N_c through m_N_c.
  std::complex<double> _h_j = 0.;
  if (0 <= j && j < m_N_c - 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  } else if (m_N_c - 1 <= j && j < m_N_c + 1) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
    _h_j -= m_d_expinu * m_c(j + 1 - m_N_c);
  } else if (m_N_c + 1 <= j && j < 2 * m_N_c) {
    for (int n = -m_N_c; n < -m_N_c + j + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
    _h_j -= m_d_expinu * m_c(j + 1 - m_N_c);
    _h_j -= m_d_expminu * m_c(j - 1 - m_N_c);
  } else if (j == 2 * m_N_c) {
    for (int n = -m_N_c; n < m_N_c + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
    _h_j -= m_d_expinu * m_c(j + 1 - m_N_c);
    _h_j -= m_d_expminu * m_c(j - 1 - m_N_c);
    _h_j -= m_omdd;
  } else if (2 * m_N_c + 1 <= j && j < 3 * m_N_c) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
    _h_j -= m_d_expinu * m_c(j + 1 - m_N_c);
    _h_j -= m_d_expminu * m_c(j - 1 - m_N_c);
  } else if (3 * m_N_c <= j && j < 3 * m_N_c + 2) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
    _h_j -= m_d_expminu * m_c(j - 1 - m_N_c);
  } else if (3 * m_N_c + 2 <= j && j < 4 * m_N_c + 1) {
    for (int n = -3 * m_N_c + j; n < m_N_c + 1; n++) {
      _h_j += m_c(n + m_N_c) * m_c(j - n - m_N_c);
    }
  }
  return _h_j;
}


bool Fluxes::no_obvious_intersections(const double d, const double nu) {

  bool noi = false;
  if (d <= 1.) {
    // Planet centre inside stellar disc.
    if (m_max_rp <= 1. - d) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region enclosed by entire planet's limb.
      m_theta = {nu - fractions::pi, nu + fractions::pi};
      m_theta_type = {intersections::entire_planet};
      noi = true;
    } else if (m_min_rp >= 1. + d) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      m_theta = {-fractions::pi, fractions::pi};
      m_theta_type = {intersections::entire_star};
      noi = true;
    }
  } else {
    // Planet centre outside stellar disc.
    if (m_max_rp <= d - 1.) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region is zero.
      m_theta = {};
      m_theta_type = {intersections::beyond};
      noi = true;
    } else if (m_min_rp >= d + 1.) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      m_theta = {-fractions::pi, fractions::pi};
      m_theta_type = {intersections::entire_star};
      noi = true;
    }
  }
  return noi;
}


std::vector<double> Fluxes::compute_real_theta_roots(
  const Eigen::Matrix<std::complex<double>, EigD, EigD>&
    companion_matrix, int shape) {

  // Solve eigenvalues.
  Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>, EigD, EigD>> ces;
  ces.compute(companion_matrix, false);
  Eigen::Vector<std::complex<double>, EigD> e_vals = ces.eigenvalues();

  // Select real thetas only: angle to e_val on unit circle in complex plane.
  std::vector<double> _theta;
  for (int j = 0; j < shape; j++) {
    double e_vals_abs = std::abs(e_vals(j));
    if (tolerance::unit_circle_lo < e_vals_abs
        && e_vals_abs < tolerance::unit_circle_hi) {
      _theta.push_back(std::arg(e_vals(j)));
    }
  }
  return _theta;
}


bool Fluxes::trivial_configuration(const double d, const double nu) {

  bool tc = false;
  double _nu = nu;
  double _rp_nu = this->rp_theta(_nu);
  if (d <= 1.) {
    // Planet centre inside stellar disc.
    if (_rp_nu < 1. + d) {
      // Planet radius toward stellar centre closer than stellar limb.
      // Overlap region enclosed by entire planet's limb as no intersects.
      m_theta = {nu - fractions::pi, nu + fractions::pi};
      m_theta_type = {intersections::entire_planet};
      tc = true;
    } else if (_rp_nu > 1. + d) {
      // Planet radius toward stellar centre beyond stellar limb.
      // Overlap region enclosed by entire star's limb as no intersects.
      m_theta = {-fractions::pi, fractions::pi};
      m_theta_type = {intersections::entire_star};
      tc = true;
    }
  } else {
    // Planet centre outside stellar disc.
    if (_rp_nu < 1. + d) {
      // Planet radius toward stellar centre closer than stellar limb.
      // Overlap region is zero as no intersects.
      m_theta = {};
      m_theta_type = {intersections::beyond};
      tc = true;
    } else if (_rp_nu > 1. + d) {
      // Planet radius toward stellar centre beyond stellar limb.
      // Overlap region enclosed by entire star's limb as no intersects.
      m_theta = {-fractions::pi, fractions::pi};
      m_theta_type = {intersections::entire_star};
      tc = true;
    }
  }
  return tc;
}


void Fluxes::characterise_intersection_pairs(const double d,
                                             const double nu) {

  int T_theta_j;
  int T_theta_j_p1;
  int dT_dtheta_theta_j;
  int dT_dtheta_theta_j_p1;
  double dcos_thetamnu = d * std::cos(m_theta[0] - nu);
  double dsin_thetamnu = d * std::sin(m_theta[0] - nu);
  this->associate_intersections(0, d, dcos_thetamnu, T_theta_j);
  this->gradient_intersections(0, dsin_thetamnu, dcos_thetamnu,
                               T_theta_j, dT_dtheta_theta_j);
  for (int j = 0; j < m_theta.size() - 1; j++) {

    // Associate theta_j and theta_j+1 with T+ or T- intersection eqn.
    dcos_thetamnu = d * std::cos(m_theta[j + 1] - nu);
    this->associate_intersections(j + 1, d, dcos_thetamnu, T_theta_j_p1);

    // Compute gradients dT+-(theta_j)/dtheta and dT+-(theta_j+1)/dtheta.
    dsin_thetamnu = d * std::sin(m_theta[j + 1] - nu);
    this->gradient_intersections(j + 1, dsin_thetamnu, dcos_thetamnu,
                                 T_theta_j_p1, dT_dtheta_theta_j_p1);

    // Assign enclosing segment.
    if (T_theta_j == 1 && T_theta_j_p1 == 1
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_p1 == 1) {
      m_theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_p1 == 1
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_p1 == 0) {
      m_theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_p1 == 0
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_p1 == 1) {
      m_theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_p1 == 0
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_p1 == 0) {
      m_theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_p1 == 0
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_p1 == 0) {
      m_theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_p1 == 0
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_p1 == 1) {
      m_theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_p1 == 1
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_p1 == 0) {
      m_theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_p1 == 1
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_p1 == 1) {
      m_theta_type.push_back(intersections::planet);
    }

    // Cache theta_j_p1 associations and gradients for next iteration.
    T_theta_j = T_theta_j_p1;
    dT_dtheta_theta_j = dT_dtheta_theta_j_p1;
  }
}


void Fluxes::associate_intersections(
  int j, const double d, double dcos_thetamnu, int& out_T_theta_j) {
  if (d <= 1.) {
    // Always T+ when the planet is inside the stellar disc.
    out_T_theta_j = intersections::T_plus;
  } else {
    // Check residuals of T+ eqn.
    double T_plus_theta_j_res = std::abs(
      this->rp_theta(m_theta[j]) - this->rs_theta(d, dcos_thetamnu, 1));

    if (T_plus_theta_j_res < tolerance::intersect_associate) {
      out_T_theta_j = intersections::T_plus;
    } else {
      out_T_theta_j = intersections::T_minus;
    }
  }
}


void Fluxes::gradient_intersections(
  int j, double dsin_thetamnu, double dcos_thetamnu,
  int plus_solution, int& out_dT_dtheta_theta_j) {

  double grad = this->drp_dtheta(m_theta[j]) + dsin_thetamnu;
  double frac_term = (dsin_thetamnu * dcos_thetamnu) / std::sqrt(
    dcos_thetamnu * dcos_thetamnu - m_dd + 1.);
  if (plus_solution == 1) {
    grad += frac_term;
  } else {
    grad -= frac_term;
  }

  if (grad > 0.) {
    out_dT_dtheta_theta_j = intersections::dT_dtheta_plus;
  } else {
    out_dT_dtheta_theta_j = intersections::dT_dtheta_minus;
  }
}


Eigen::Vector<std::complex<double>, EigD> Fluxes::complex_convolve(
  const Eigen::Vector<std::complex<double>, EigD>& a,
  const Eigen::Vector<std::complex<double>, EigD>& b,
  int len_a, int len_b, int len_c) {

  // Initialise convolved vector of zeroes.
  Eigen::Vector<std::complex<double>, EigD> conv;
  conv.resize(len_c);
  conv.setZero();

  // Compute convolution.
  for (int n = 0; n < len_a; n++) {
    for (int m = 0; m < len_b; m++) {
      conv(n + m) += a(n) * b(m);
    }
  }
  return conv;
}


Eigen::Vector<std::complex<double>, EigD>
Fluxes::complex_ca_vector_addition(
  const Eigen::Vector<std::complex<double>, EigD>& a,
  const Eigen::Vector<std::complex<double>, EigD>& b,
  int len_a, int len_b) {

  // Initialise summed vector.
  Eigen::Vector<std::complex<double>, EigD> ca_sum;

  // Compute centre-aligned addition.
  if (len_a > len_b) {
    ca_sum = a;
    int half_diff = (len_a - len_b) * fractions::one_half;
    for (int n = 0; n < len_b; n++) {
      ca_sum(n + half_diff) += b(n);
    }
  } else if (len_b > len_a) {
    ca_sum = b;
    int half_diff = (len_b - len_a) * fractions::one_half;
    for (int n = 0; n < len_a; n++) {
      ca_sum(n + half_diff) += a(n);
    }
  } else {
    // Equal length, already centre aligned.
    ca_sum = a + b;
  }
  return ca_sum;
}


void Fluxes::analytic_even_terms(int _j, int theta_type_j, double _theta_j,
                                 double _theta_j_p1, const double d,
                                 const double nu) {
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
    } else {
      s0_planet += q0(mpN_q0) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
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
    } else {
      s2_planet += q2(mpN_q2) / (1.i * (1. * m))
                   * (eim_theta_j_p1 - eim_theta_j);
    }
  }
  m_s0 += fractions::one_half * s0_planet.real();
  m_s2 += fractions::one_quarter * s2_planet.real();
}


void Fluxes::numerical_odd_terms(int _j, int theta_type_j, double _theta_j,
                                 double _theta_j_p1, const double d,
                                 const double nu) {
  double s1_planet = 0.;
  double half_theta_range = (_theta_j_p1 - _theta_j) / 2.;

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
    }
    m_s1 += half_theta_range * s1_planet;

  } else {
    double s12_planet = 0.;
    double s32_planet = 0.;

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
    }
    m_s12 += half_theta_range * s12_planet;
    m_s1 += half_theta_range * s1_planet;
    m_s32 += half_theta_range * s32_planet;
  }
}
