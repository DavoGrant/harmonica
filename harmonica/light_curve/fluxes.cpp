#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fluxes.hpp"
#include "../constants/constants.hpp"

using namespace std::complex_literals;


Fluxes::Fluxes(int ld_law,
               py::array_t<double, py::array::c_style> us,
               py::array_t<double, py::array::c_style> rs,
               int pnl_c, int pnl_e, bool require_gradients) {

  // Set switches.
  _precision_nl_centre = pnl_c;
  _precision_nl_edge = pnl_e;
  _require_gradients = require_gradients;

  // Unpack python arrays.
  auto us_ = us.unchecked<1>();
  auto rs_ = rs.unchecked<1>();
  _n_rs = rs_.shape(0);  // Always odd.

  _ld_law = ld_law;
  if (_ld_law == limb_darkening::quadratic) {
    // Normalisation.
    double I_0_bt = (1. - us_(0) / 3. - us_(1) / 6.);
    I_0_bts = I_0_bt * I_0_bt;
    I_0 = 1. / (fractions::pi * I_0_bt);

    // Quadratic limb-darkening law.
    Eigen::Vector<double, 3> u {1., us_(0), us_(1)};
    Eigen::Matrix<double, 3, 3> B {{1., -1., -1.},
                                   {0., 1., 2.},
                                   {0., 0., -1.}};

    // Change to polynomial basis.
    p = B * u;

  } else {
    // Normalisation.
    double I_0_bt = (1. - us_(0) / 5. - us_(1) / 3.
                     - 3. * us_(2) / 7. - us_(3) / 2.);
    I_0_bts = I_0_bt * I_0_bt;
    I_0 = 1. / (fractions::pi * I_0_bt);

    // Non-linear limb-darkening law.
    Eigen::Vector<double, 5> u {1., us_(0), us_(1), us_(2), us_(3)};
    Eigen::Matrix<double, 5, 5> B {{1., -1., -1., -1., -1.},
                                   {0., 1., 0., 0., 0.},
                                   {0., 0., 1., 0., 0.},
                                   {0., 0., 0., 1., 0.},
                                   {0., 0., 0., 0., 1.}};

    // Change to polynomial basis.
    p = B * u;
  }

  // Convert cosine, sine to complex Fourier coefficients.
  N_c = (_n_rs - 1) * fractions::one_half;
  c.resize(_n_rs);
  c(N_c) = rs_(0);
  for (int n = 0; n < N_c; n++) {
    double a_real = rs_(_n_rs - 2 - 2 * n);
    double b_imag = rs_(_n_rs - 1 - 2 * n);
    c(n) = (a_real + b_imag * 1.i) * fractions::one_half;
    c(_n_rs - 1 - n) = (a_real - b_imag * 1.i) * fractions::one_half;
  }

  // Pre-compute max and min planet radii.
  min_rp = c(N_c).real();
  max_rp = c(N_c).real();
  if (N_c != 0) {
    // Build the extrema companion matrix.
    int D_shape = 2 * N_c;
    D.resize(D_shape, D_shape);
    for (int j = 1; j < D_shape + 1; j++) {
      for (int k = 1; k < D_shape + 1; k++) {
        D(j - 1, k - 1) = this->extrema_companion_matrix_D_jk(j, k, D_shape);
      }
    }

    // Get the extrema companion matrix roots.
    std::vector<double> theta_extrema = this->compute_real_theta_roots(
      D, D_shape, false);

    // Find the max an min radius values.
    for (int j = 0; j < theta_extrema.size(); j++) {
      double _rp = this->rp_theta(theta_extrema[j]);
      if (_rp < min_rp) {
        min_rp = _rp;
      } else if (_rp > max_rp) {
        max_rp = _rp;
      }
    }
  }

  if (N_c != 0) {
    // Pre-build the intersection eqn companion matrix for the terms
    // that are independent of position, d and nu.
    C_shape = 4 * N_c;
    C0.resize(C_shape, C_shape);
    for (int j = 1; j < C_shape + 1; j++) {
      for (int k = 1; k < C_shape + 1; k++) {
        C0(j - 1, k - 1) = this->intersection_companion_matrix_C_jk_base(
          j, k, C_shape);
      }
    }
  }

  // Pre-compute c (*) c.
  _len_c_conv_c = 2 * _n_rs - 1;
  _c_conv_c = complex_convolve(c, c, _n_rs, _n_rs, _len_c_conv_c);

  // Pre-compute Delta element-wise multiply c.
  _Delta_ew_c = c;
   for (int n = -N_c; n < N_c + 1; n++) {
    _Delta_ew_c(n + N_c) *= (1. * n) * 1.i;
  }

  // Pre-compute beta_sin/cos base vectors.
  _len_beta_conv_c = 3 + _n_rs - 1;
  _beta_sin0 << -fractions::one_half, 0., -fractions::one_half;
  _beta_cos0 << fractions::one_half, 0., fractions::one_half;

  // Pre-compute conv sizes.
  _len_q_rhs = std::max(_len_c_conv_c, _len_beta_conv_c);
  _mid_q_lhs = (_len_q_rhs - 1) / 2;
  _len_q = 2 * _len_q_rhs - 1;
  N_q0 = (_len_q_rhs - 1) / 2;
  N_q2 = (_len_q - 1) / 2;

  if (_require_gradients == true) {
    // Pre-compute some derivative terms.
    df_dalpha = -1.;
    if (_ld_law == limb_darkening::quadratic) {
      dI0_du1 = 1. / (fractions::threepi * I_0_bts);
      dI0_du2 = 1. / (fractions::sixpi * I_0_bts);

      dalpha_ds0 = I_0 * (1. - us_(0) - us_(1));
      dalpha_ds1 = I_0 * (us_(0) + 2. * us_(1));
      dalpha_ds2 = -I_0 * us_(1);

    } else {
      dI0_du1 = 1. / (fractions::fivepi * I_0_bts);
      dI0_du2 = 1. / (fractions::threepi * I_0_bts);
      dI0_du3 = 1. / (fractions::sevenpi_d_3 * I_0_bts);
      dI0_du4 = 1. / (fractions::twopi * I_0_bts);

      dalpha_ds0 = I_0 * (1. - us_(0) - us_(1) - us_(2) - us_(3));
      dalpha_ds12 = I_0 * us_(0);
      dalpha_ds1 = I_0 * us_(1);
      dalpha_ds32 = I_0 * us_(2);
      dalpha_ds2 = I_0 * us_(3);
    }

    df_dcs.resize(_n_rs);
    ds0_dcs.resize(_n_rs);
    ds12_dcs.resize(_n_rs);
    ds1_dcs.resize(_n_rs);
    ds32_dcs.resize(_n_rs);
    ds2_dcs.resize(_n_rs);

    dc0_da0 = 1.;
    dcplus_dan = 1. / 2.;
    dcminus_dan = 1. / 2.;
    dcplus_dbn = -1.i / 2.;
    dcminus_dbn = 1.i / 2.;

    _zeroes_c_conv_c.resize(_len_c_conv_c);
    _zeroes_c_conv_c.setZero();

    if (N_c != 0) {
      // Pre-build the derivatives of intersection eqn companion matrix.
      dC_dd.resize(C_shape, C_shape);
      dC_dd.setZero();
    }
  }
}


void Fluxes::pre_compute_psq(const double &d, const double &nu) {
  _td = 2. * d;
  _dd = d * d;
  _omdd = 1. - _dd;
  _expinu = std::exp(1.i * nu);
  _expminu = std::exp(-1.i * nu);
  _d_expinu = d * std::exp(1.i * nu);
  _d_expminu = d * std::exp(-1.i * nu);
}


double Fluxes::rs_theta(const double &d, double &dcos_thetamnu,
                        int plus_solution) {
  if (d <= 1.) {
    return dcos_thetamnu + std::sqrt(dcos_thetamnu * dcos_thetamnu
                                     - _dd + 1.);
  } else {
    if (plus_solution == 1) {
      // r_s+ solution.
      return dcos_thetamnu + std::sqrt(dcos_thetamnu * dcos_thetamnu
                                       - _dd + 1.);
    } else {
      // r_s- solution.
      return dcos_thetamnu - std::sqrt(dcos_thetamnu * dcos_thetamnu
                                       - _dd + 1.);
    }
  }
}


double Fluxes::drp_dtheta(double &_theta) {
  std::complex<double> rp = 0.;
  for (int n = -N_c; n < N_c + 1; n++) {
    rp += 1.i * (1. * n) * c(n + N_c) * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


double Fluxes::d2rp_dtheta2(double &_theta) {
  std::complex<double> rp = 0.;
  for (int n = -N_c; n < N_c + 1; n++) {
    rp += (1.i * (1. * n)) * (1.i * (1. * n)) * c(n + N_c)
           * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


std::complex<double> Fluxes::extrema_companion_matrix_D_jk(int j, int k,
                                                           int &shape) {
  // NB. matrix elements are one-indexed.
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> moo_denom = -1. / (1. * N_c * c(shape));
  if (k == shape) {
    return (j - N_c - 1.) * c(j - 1) * moo_denom;
  } else {
    if (j == k + 1) {
      return 1.;
    } else {
      return 0.;
    }
  }
}


std::complex<double> Fluxes::intersection_companion_matrix_C_jk_base(
  int j, int k, int &shape) {
  // NB. matrix elements are one-indexed.
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
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
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> h_j_base = 0.;
  if (0 <= j && j < N_c - 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (N_c - 1 <= j && j < N_c + 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (N_c + 1 <= j && j < 2 * N_c) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (j == 2 * N_c) {
    h_j_base -= 1.;
    for (int n = -N_c; n < N_c + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (2 * N_c + 1 <= j && j < 3 * N_c) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (3 * N_c <= j && j < 3 * N_c + 2) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  } else if (3 * N_c + 2 <= j && j < 4 * N_c + 1) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j_base += c(n + N_c) * c(j - n - N_c);
    }
  }
  return h_j_base;
}


std::complex<double> Fluxes::intersection_polynomial_coefficients_h_j_update(
  int j) {
  // NB. c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> h_j_update = 0.;
  if (N_c - 1 <= j && j < N_c + 1) {
    h_j_update -= _d_expinu * c(j + 1 - N_c);
  } else if (N_c + 1 <= j && j < 2 * N_c) {
    h_j_update -= _d_expinu * c(j + 1 - N_c);
    h_j_update -= _d_expminu * c(j - 1 - N_c);
  } else if (j == 2 * N_c) {
    h_j_update -= _d_expinu * c(j + 1 - N_c);
    h_j_update -= _d_expminu * c(j - 1 - N_c);
    h_j_update += _dd;
  } else if (2 * N_c + 1 <= j && j < 3 * N_c) {
    h_j_update -= _d_expinu * c(j + 1 - N_c);
    h_j_update -= _d_expminu * c(j - 1 - N_c);
  } else if (3 * N_c <= j && j < 3 * N_c + 2) {
    h_j_update -= _d_expminu * c(j - 1 - N_c);
  }
  return h_j_update;
}


std::complex<double> Fluxes::intersection_polynomial_coefficient_moo_denom(
  int j) {
  // NB. c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> h_4Nc = 0.;
  if (3 * N_c <= j && j < 3 * N_c + 2) {
    h_4Nc -= _d_expminu * c(j - 1 - N_c);
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_4Nc += c(n + N_c) * c(j - n - N_c);
    }
  } else if (3 * N_c + 2 <= j && j < 4 * N_c + 1) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_4Nc += c(n + N_c) * c(j - n - N_c);
    }
  }
  return -1. / h_4Nc;
}


std::complex<double> Fluxes::h_j(
  int j) {
  // NB. c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> _h_j = 0.;
  if (0 <= j && j < N_c - 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (N_c - 1 <= j && j < N_c + 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
    _h_j -= _d_expinu * c(j + 1 - N_c);
  } else if (N_c + 1 <= j && j < 2 * N_c) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
    _h_j -= _d_expinu * c(j + 1 - N_c);
    _h_j -= _d_expminu * c(j - 1 - N_c);
  } else if (j == 2 * N_c) {
    for (int n = -N_c; n < N_c + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
    _h_j -= _d_expinu * c(j + 1 - N_c);
    _h_j -= _d_expminu * c(j - 1 - N_c);
    _h_j -= _omdd;
  } else if (2 * N_c + 1 <= j && j < 3 * N_c) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
    _h_j -= _d_expinu * c(j + 1 - N_c);
    _h_j -= _d_expminu * c(j - 1 - N_c);
  } else if (3 * N_c <= j && j < 3 * N_c + 2) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
    _h_j -= _d_expminu * c(j - 1 - N_c);
  } else if (3 * N_c + 2 <= j && j < 4 * N_c + 1) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      _h_j += c(n + N_c) * c(j - n - N_c);
    }
  }
  return _h_j;
}


std::complex<double> Fluxes::dh_j_dd(
  int j) {
  // NB. c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> _dh_j_dd = 0.;
  if (N_c - 1 <= j && j < N_c + 1) {
    _dh_j_dd -= _expinu * c(j + 1 - N_c);
  } else if (N_c + 1 <= j && j < 2 * N_c) {
    _dh_j_dd -= _expinu * c(j + 1 - N_c);
    _dh_j_dd -= _expminu * c(j - 1 - N_c);
  } else if (j == 2 * N_c) {
    _dh_j_dd -= _expinu * c(j + 1 - N_c);
    _dh_j_dd -= _expminu * c(j - 1 - N_c);
    _dh_j_dd += _td;
  } else if (2 * N_c + 1 <= j && j < 3 * N_c) {
    _dh_j_dd -= _expinu * c(j + 1 - N_c);
    _dh_j_dd -= _expminu * c(j - 1 - N_c);
  } else if (3 * N_c <= j && j < 3 * N_c + 2) {
    _dh_j_dd -= _expminu * c(j - 1 - N_c);
  }
  return _dh_j_dd;
}


void Fluxes::find_intersections_theta(const double &d, const double &nu) {

  // Check cases where no obvious intersections, avoiding eigenvalue runtime.
  if (this->no_obvious_intersections(d, nu)) { return; }

  if (N_c != 0) {
    // Update intersection companion matrix for current position.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      C = C0;
    std::complex<double> moo_denom =
      intersection_polynomial_coefficient_moo_denom(C_shape);
    for (int j = 1; j < C_shape + 1; j++) {
      C(j - 1, C_shape - 1) +=
        this->intersection_polynomial_coefficients_h_j_update(j - 1);
      C(j - 1, C_shape - 1) *= moo_denom;
    }

    // Get the intersection companion matrix roots.
    theta = this->compute_real_theta_roots(C, C_shape, _require_gradients);

  } else {
    // Transmission string is a circle.
    double _c0c0 = c(N_c).real() * c(N_c).real();
    double acos_arg = (_c0c0 + _dd - 1.) / (2. * c(N_c).real() * d);
    double acos_intersect = std::acos(acos_arg);
    theta = {nu - acos_intersect, nu + acos_intersect};
    if (_require_gradients == true) {
      double dtheta_dacos_arg = 1. / std::sqrt(1. - acos_arg * acos_arg);
      double dacos_arg_dd = (-_c0c0 + _dd - 1.) / (2. * c(N_c).real() * _dd);
      double dtheta_dd = dtheta_dacos_arg * dacos_arg_dd;
      dthetas_dd = {dtheta_dd, -dtheta_dd};
    }
  }

  if (theta.size() == 0) {
    // No roots, check which trivial case this configuration corresponds to.
    if (this->trivial_configuration(d, nu)) { return; }
  }

  if (_require_gradients == false) {
    // Sort thetas in ascending order, -pi < theta <= pi.
    std::sort(theta.begin(), theta.end());
  } else {
    // Sort thetas and corresponding theta derivatives.
    std::vector<int> indices(theta.size());
    std::vector<double> theta_unsorted = theta;
    std::vector<double> dthetas_dd_unsorted = dthetas_dd;
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](double A, double B)->bool {
      return theta_unsorted[A] < theta_unsorted[B];
    });
    for (int j = 0; j < theta.size(); j++) {
      theta[j] = theta_unsorted[indices[j]];
      dthetas_dd[j] = dthetas_dd_unsorted[indices[j]];
    }

    // Duplicate first derivative at end of vector.
    dthetas_dd.push_back(dthetas_dd[0]);
  }

  // Ensure theta vector spans a closed loop, 2pi in total.
  // Thus, duplicate first intersection + 2pi at the end.
  theta.push_back(theta[0] + fractions::twopi);

  // Characterise theta pairs.
  this->characterise_intersection_pairs(d, nu);
}


bool Fluxes::no_obvious_intersections(const double &d, const double &nu) {

  bool noi = false;
  if (d <= 1.) {
    // Planet centre inside stellar disc.
    if (max_rp <= 1. - d) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region enclosed by entire planet's limb.
      theta = {nu - fractions::pi, nu + fractions::pi};
      theta_type = {intersections::entire_planet};
      noi = true;
    } else if (min_rp >= 1. + d) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::entire_star};
      noi = true;
    }
  } else {
    // Planet centre outside stellar disc.
    if (max_rp <= d - 1.) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region is zero.
      theta = {};
      theta_type = {intersections::beyond};
      noi = true;
    } else if (min_rp >= d + 1.) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::entire_star};
      noi = true;
    }
  }
  if (noi == true && _require_gradients == true) {
    dthetas_dd = {0., 0.};
  }
  return noi;
}


std::vector<double> Fluxes::compute_real_theta_roots(
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
    companion_matrix, int &shape, bool require_eigenvectors) {

  // Solve eigenvalues.
  Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>,
    Eigen::Dynamic, Eigen::Dynamic>> ces;
  ces.compute(companion_matrix, require_eigenvectors);
  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    e_vals = ces.eigenvalues();

  std::vector<double> _theta;
  if (require_eigenvectors == false) {
    // Select real thetas only: angle to e_val on unit circle in complex plane.
    for (int j = 0; j < shape; j++) {
      double e_vals_abs = std::abs(e_vals(j));
      if (tolerance::unit_circle_lo < e_vals_abs
          && e_vals_abs < tolerance::unit_circle_hi) {
        _theta.push_back(std::arg(e_vals(j)));
      }
    }
  } else {

    // dC_dd.
    std::complex<double> h_4Nc = this->h_j(C_shape);
    for (int j = 1; j < C_shape + 1; j++) {
      dC_dd(j - 1, C_shape - 1) = (
        this->h_j(j - 1) * this->dh_j_dd(C_shape)
        - this->dh_j_dd(j - 1) * h_4Nc) / (h_4Nc * h_4Nc);
    }

    // And use eigenvectors for derivatives.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      e_vecs = ces.eigenvectors();
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      e_vecs_inv = e_vecs.inverse();
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      dlambda_dd_full = e_vecs_inv * dC_dd * e_vecs;

    // Select real thetas only, and corresponding derivatives.
    for (int j = 0; j < shape; j++) {
      double e_vals_abs = std::abs(e_vals(j));
      if (tolerance::unit_circle_lo < e_vals_abs
          && e_vals_abs < tolerance::unit_circle_hi) {
        _theta.push_back(std::arg(e_vals(j)));

        // dtheta_dd.
        std::complex<double> dtheta_dlambda = -1.i / e_vals(j);
        dthetas_dd.push_back((dtheta_dlambda * dlambda_dd_full(j, j)).real());
      }
    }
  }
  return _theta;
}


bool Fluxes::trivial_configuration(const double &d, const double &nu) {

  bool tc = false;
  double _nu = nu;
  double _rp_nu = this->rp_theta(_nu);
  if (d <= 1.) {
    // Planet centre inside stellar disc.
    if (_rp_nu < 1. + d) {
      // Planet radius toward stellar centre closer than stellar limb.
      // Overlap region enclosed by entire planet's limb as no intersects.
      theta = {nu - fractions::pi, nu + fractions::pi};
      theta_type = {intersections::entire_planet};
      tc = true;
    } else if (_rp_nu > 1. + d) {
      // Planet radius toward stellar centre beyond stellar limb.
      // Overlap region enclosed by entire star's limb as no intersects.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::entire_star};
      tc = true;
    }
  } else {
    // Planet centre outside stellar disc.
    if (_rp_nu < 1. + d) {
      // Planet radius toward stellar centre closer than stellar limb.
      // Overlap region is zero as no intersects.
      theta = {};
      theta_type = {intersections::beyond};
      tc = true;
    } else if (_rp_nu > 1. + d) {
      // Planet radius toward stellar centre beyond stellar limb.
      // Overlap region enclosed by entire star's limb as no intersects.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::entire_star};
      tc = true;
    }
  }
  if (tc == true && _require_gradients == true) {
    dthetas_dd = {0., 0.};
  }
  return tc;
}


void Fluxes::characterise_intersection_pairs(const double &d,
                                             const double &nu) {

  int T_theta_j;
  int T_theta_j_plus_1;
  int dT_dtheta_theta_j;
  int dT_dtheta_theta_j_plus_1;
  double dcos_thetamnu = d * std::cos(theta[0] - nu);
  double dsin_thetamnu = d * std::sin(theta[0] - nu);
  this->associate_intersections(0, d, dcos_thetamnu, T_theta_j);
  this->gradient_intersections(0, dsin_thetamnu, dcos_thetamnu,
                               T_theta_j, dT_dtheta_theta_j);
  for (int j = 0; j < theta.size() - 1; j++) {

    // Associate theta_j and theta_j+1 with T+ or T- intersection eqn.
    dcos_thetamnu = d * std::cos(theta[j + 1] - nu);
    this->associate_intersections(j + 1, d, dcos_thetamnu, T_theta_j_plus_1);

    // Compute gradients dT+-(theta_j)/dtheta and dT+-(theta_j+1)/dtheta.
    dsin_thetamnu = d * std::sin(theta[j + 1] - nu);
    this->gradient_intersections(j + 1, dsin_thetamnu, dcos_thetamnu,
                                 T_theta_j_plus_1, dT_dtheta_theta_j_plus_1);

    // Assign enclosing segment.
    if (T_theta_j == 1 && T_theta_j_plus_1 == 1
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_plus_1 == 1) {
      theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_plus_1 == 1
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_plus_1 == 0) {
      theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_plus_1 == 0
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_plus_1 == 1) {
      theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_plus_1 == 0
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_plus_1 == 0) {
      theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_plus_1 == 0
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_plus_1 == 0) {
      theta_type.push_back(intersections::planet);
    } else if (T_theta_j == 1 && T_theta_j_plus_1 == 0
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_plus_1 == 1) {
      theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_plus_1 == 1
        && dT_dtheta_theta_j == 0 && dT_dtheta_theta_j_plus_1 == 0) {
      theta_type.push_back(intersections::star);
    } else if (T_theta_j == 0 && T_theta_j_plus_1 == 1
        && dT_dtheta_theta_j == 1 && dT_dtheta_theta_j_plus_1 == 1) {
      theta_type.push_back(intersections::planet);
    }

    // Cache theta_j_plus_1 associations and gradients for next iteration.
    T_theta_j = T_theta_j_plus_1;
    dT_dtheta_theta_j = dT_dtheta_theta_j_plus_1;
  }
}


void Fluxes::associate_intersections(
  int j, const double &d, double &dcos_thetamnu, int &T_theta_j) {
  if (d <= 1.) {
    // Always T+ when the planet is inside the stellar disc.
    T_theta_j = intersections::T_plus;
  } else {
    // Check residuals of T+ eqn.
    double T_plus_theta_j_res = std::abs(
      this->rp_theta(theta[j]) - this->rs_theta(d, dcos_thetamnu, 1));

    if (T_plus_theta_j_res < tolerance::intersect_associate) {
      T_theta_j = intersections::T_plus;
    } else {
      T_theta_j = intersections::T_minus;
    }
  }
}


void Fluxes::gradient_intersections(
  int j, double &dsin_thetamnu, double &dcos_thetamnu,
  int plus_solution, int &dT_dtheta_theta_j) {

  double grad = this->drp_dtheta(theta[j]) + dsin_thetamnu;
  double frac_term = (dsin_thetamnu * dcos_thetamnu) / std::sqrt(
    dcos_thetamnu * dcos_thetamnu - _dd + 1.);
  if (plus_solution == 1) {
    grad += frac_term;
  } else {
    grad -= frac_term;
  }

  if (grad > 0.) {
    dT_dtheta_theta_j = intersections::dT_dtheta_plus;
  } else {
    dT_dtheta_theta_j = intersections::dT_dtheta_minus;
  }
}


Eigen::Vector<std::complex<double>, Eigen::Dynamic> Fluxes::complex_convolve(
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> a,
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> b,
  int len_a, int len_b, int len_c) {

  // Initialise convolved vector of zeroes.
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> conv;
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


Eigen::Vector<std::complex<double>, Eigen::Dynamic>
Fluxes::complex_ca_vector_addition(
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> a,
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> b,
  int len_a, int len_b) {

  // Initialise summed vector.
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> ca_sum;

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


void Fluxes::s_planet(int _j, int theta_type_j, double &_theta_j,
                      double &_theta_j_plus_1, const double &d,
                      const double &nu) {

  // Compute the closed-form even terms.
  this->analytic_even_terms(_j, theta_type_j, _theta_j,
                            _theta_j_plus_1, d, nu);

  // Compute the numerical odd and half-integer terms.
  this->numerical_odd_terms(_j, theta_type_j, _theta_j,
                            _theta_j_plus_1, d, nu);
}


void Fluxes::analytic_even_terms(int _j, int theta_type_j, double &_theta_j,
                                 double &_theta_j_plus_1, const double &d,
                                 const double &nu) {
  // Build and convolve beta_sin, beta_cos vectors.
  double _theta_diff = _theta_j_plus_1 - _theta_j;
  double sin_nu = std::sin(nu);
  double cos_nu = std::cos(nu);

  Eigen::Vector<std::complex<double>, 3> beta_sin = _beta_sin0;
  beta_sin(0) *= sin_nu - cos_nu * 1.i;
  beta_sin(2) *= sin_nu + cos_nu * 1.i;

  Eigen::Vector<std::complex<double>, 3> beta_cos = _beta_cos0;
  beta_cos(0) *= cos_nu + sin_nu * 1.i;
  beta_cos(2) *= cos_nu - sin_nu * 1.i;

  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    beta_cos_conv_c = complex_convolve(
      beta_cos, c, 3, _n_rs, _len_beta_conv_c);

  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    beta_sin_conv_Delta_ew_c = complex_convolve(
      beta_sin, _Delta_ew_c, 3, _n_rs, _len_beta_conv_c);

  // Generate q0, equivalent to q_rhs for n=2.
  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    q0 = complex_ca_vector_addition(
      _c_conv_c, -d * (beta_cos_conv_c + beta_sin_conv_Delta_ew_c),
      _len_c_conv_c, _len_beta_conv_c);

  // Generate q lhs.
  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    q_lhs = complex_ca_vector_addition(
      -_c_conv_c, _td * beta_cos_conv_c,
      _len_c_conv_c, _len_beta_conv_c);
  q_lhs(_mid_q_lhs) += _omdd;
  q_lhs(_mid_q_lhs) += 1.;

  // Generate q2.
  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    q2 = complex_convolve(q_lhs, q0, _len_q_rhs, _len_q_rhs, _len_q);

  Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    dq0_dd, dq_lhs_dd, dq2_dd;
  std::complex<double> ds0_q0_dd, ds0_theta_j_dd, ds0_theta_j_plus_1_dd,
                       ds2_q2_dd, ds2_theta_j_dd, ds2_theta_j_plus_1_dd;
  if (_require_gradients == true) {
    ds0_q0_dd = 0., ds0_theta_j_dd = 0., ds0_theta_j_plus_1_dd = 0.,
    ds2_q2_dd = 0., ds2_theta_j_dd = 0., ds2_theta_j_plus_1_dd = 0.;

    dq0_dd = complex_ca_vector_addition(
        _zeroes_c_conv_c, -(beta_cos_conv_c + beta_sin_conv_Delta_ew_c),
        _len_c_conv_c, _len_beta_conv_c);

    dq_lhs_dd = complex_ca_vector_addition(
        -_zeroes_c_conv_c, 2. * beta_cos_conv_c,
        _len_c_conv_c, _len_beta_conv_c);
    dq_lhs_dd(_mid_q_lhs) += -2 * d;

    dq2_dd = complex_ca_vector_addition(
        complex_convolve(dq_lhs_dd, q0, _len_q_rhs, _len_q_rhs, _len_q),
        complex_convolve(q_lhs, dq0_dd, _len_q_rhs, _len_q_rhs, _len_q),
        _len_q, _len_q);
  }

  // Limb-darkening constant term n=0, analytic line integral.
  std::complex<double> s0_planet = 0.;
  for (int m = -N_q0; m < N_q0 + 1; m++) {
    int mpN_q0 = m + N_q0;
    std::complex<double> eim_theta_j = std::exp(
      (1. * m) * 1.i * _theta_j);
    std::complex<double> eim_theta_j_plus_1 = std::exp(
      (1. * m) * 1.i * _theta_j_plus_1);
    if (m == 0) {
      s0_planet += q0(mpN_q0) * _theta_diff;
      if (_require_gradients == true) {
        ds0_q0_dd += dq0_dd(mpN_q0) * _theta_diff;
      }
    } else {
      s0_planet += q0(mpN_q0) / (1.i * (1. * m))
                   * (eim_theta_j_plus_1 - eim_theta_j);
      if (_require_gradients == true) {
        ds0_q0_dd += dq0_dd(mpN_q0) / (1.i * (1. * m))
                     * (eim_theta_j_plus_1 - eim_theta_j);
      }
    }
    if (_require_gradients == true && theta_type_j == intersections::planet) {
      ds0_theta_j_dd += q0(mpN_q0) * -eim_theta_j;
      ds0_theta_j_plus_1_dd += q0(mpN_q0) * eim_theta_j_plus_1;
    }
  }

  // Limb-darkening even term n=2, analytic line integral.
  std::complex<double> s2_planet = 0.;
  for (int m = -N_q2; m < N_q2 + 1; m++) {
    int mpN_q2 = m + N_q2;
    std::complex<double> eim_theta_j = std::exp(
      (1. * m) * 1.i * _theta_j);
    std::complex<double> eim_theta_j_plus_1 = std::exp(
      (1. * m) * 1.i * _theta_j_plus_1);
    if (m == 0) {
      s2_planet += q2(mpN_q2) * _theta_diff;
      if (_require_gradients == true) {
        ds2_q2_dd += dq2_dd(mpN_q2) * _theta_diff;
      }
    } else {
      s2_planet += q2(mpN_q2) / (1.i * (1. * m))
                   * (eim_theta_j_plus_1 - eim_theta_j);
      if (_require_gradients == true) {
        ds2_q2_dd += dq2_dd(mpN_q2) / (1.i * (1. * m))
                     * (eim_theta_j_plus_1 - eim_theta_j);
      }
    }
    if (_require_gradients == true && theta_type_j == intersections::planet) {
      ds2_theta_j_dd += q2(mpN_q2) * -eim_theta_j;
      ds2_theta_j_plus_1_dd += q2(mpN_q2) * eim_theta_j_plus_1;
    }
  }

  s0 += fractions::one_half * s0_planet.real();
  s2 += fractions::one_quarter * s2_planet.real();

  if (_require_gradients == true) {
    if (theta_type_j == intersections::planet) {
      ds0_theta_j_dd *= dthetas_dd[_j];
      ds0_theta_j_plus_1_dd *= dthetas_dd[_j + 1];
      ds2_theta_j_dd *= dthetas_dd[_j];
      ds2_theta_j_plus_1_dd *= dthetas_dd[_j + 1];
    }

    ds0_dd += fractions::one_half * (ds0_q0_dd + ds0_theta_j_dd
                                     + ds0_theta_j_plus_1_dd).real();
    ds2_dd += fractions::one_quarter * (ds2_q2_dd + ds2_theta_j_dd
                                        + ds2_theta_j_plus_1_dd).real();

    ds0_dnu = 0.;
    ds2_dnu = 0.;

    ds0_dcs.setZero();
    ds2_dcs.setZero();
  }
}


void Fluxes::numerical_odd_terms(int _j, int theta_type_j, double &_theta_j,
                                 double &_theta_j_plus_1, const double &d,
                                 const double &nu) {
  double s1_planet = 0.;
  double half_theta_range = (_theta_j_plus_1 - _theta_j) / 2.;

  double ds1_zeta_z_dd, ds1_eta_dd, ds1_theta_j_dd, ds1_theta_j_plus_1_dd,
         ds1_zeta_z_t_theta_j_dd, ds1_zeta_z_t_theta_j_plus_1_dd,
         ds1_zeta_z_r_t_theta_j_dd, ds1_zeta_z_r_t_theta_j_plus_1_dd,
         ds1_eta_r_t_theta_j_dd, ds1_eta_r_t_theta_j_plus_1_dd,
         ds1_eta_t_theta_j_dd, ds1_eta_t_theta_j_plus_1_dd,
         ds1_eta_rdash_t_theta_j_dd, ds1_eta_rdash_t_theta_j_plus_1_dd;
  if (_require_gradients == true) {
    ds1_zeta_z_dd = 0., ds1_eta_dd = 0.,
    ds1_theta_j_dd = 0., ds1_theta_j_plus_1_dd = 0.,
    ds1_zeta_z_t_theta_j_dd = 0., ds1_zeta_z_t_theta_j_plus_1_dd = 0.,
    ds1_zeta_z_r_t_theta_j_dd = 0., ds1_zeta_z_r_t_theta_j_plus_1_dd = 0.,
    ds1_eta_r_t_theta_j_dd = 0., ds1_eta_r_t_theta_j_plus_1_dd = 0.,
    ds1_eta_t_theta_j_dd = 0., ds1_eta_t_theta_j_plus_1_dd = 0.,
    ds1_eta_rdash_t_theta_j_dd = 0., ds1_eta_rdash_t_theta_j_plus_1_dd = 0.;
  }

  if (_ld_law == limb_darkening::quadratic) {
    // Limb-darkening half-integer and odd terms n=1, using
    // Gauss-legendre quad.
    for (int k = 0; k < _N_l; k++) {

      // Rescale legendre root.
      double t_k = half_theta_range * (_l_roots[k] + 1.) + _theta_j;

      // Evaluate integrand at t_k.
      double rp_tk = this->rp_theta(t_k);
      double rp_tks = rp_tk * rp_tk;
      double drp_dtk = this->drp_dtheta(t_k);
      double sintkmnu = std::sin(t_k - nu);
      double costkmnu = std::cos(t_k - nu);
      double d_rp_costkmnu = d * rp_tk * costkmnu;
      double d_drpdtheta_sintkmnu = d * drp_dtk * sintkmnu;
      double zp_tks = _omdd - rp_tks + 2. * d_rp_costkmnu;
      double zp_tk = std::sqrt(zp_tks);
      double zeta = (1. - zp_tks * zp_tk) / (3. * (1. - zp_tks));
      double eta = rp_tks - d_rp_costkmnu - d_drpdtheta_sintkmnu;
      s1_planet += zeta * eta * _l_weights[k];

      if (_require_gradients == true) {

        double opzp = 1 + zp_tk;
        double ds1_dzeta = eta * _l_weights[k];
        double dzeta_dzp = fractions::one_third * (1. - 1. / (opzp * opzp));
        double dzp_dd = (rp_tk * costkmnu - d) / zp_tk;
        double ds1_deta = zeta * _l_weights[k];
        double deta_dd = -rp_tk * costkmnu - drp_dtk * sintkmnu;

        ds1_zeta_z_dd += ds1_dzeta * dzeta_dzp * dzp_dd;
        ds1_eta_dd += ds1_deta * deta_dd;

        if (theta_type_j == intersections::planet) {

          double dtk_dtheta_j = 1. - fractions::one_half * (_l_roots[k] + 1.);
          double dtk_dtheta_j_plus_1 = fractions::one_half * (_l_roots[k] + 1.);

          double dz_drp = (d * costkmnu - rp_tk) / zp_tk;
          double dz_dtk = -d * rp_tk * sintkmnu / zp_tk;
          double ds1_zeta_dz = ds1_dzeta * dzeta_dzp;
          double ds1_zeta_z_dtk = ds1_zeta_dz * dz_dtk;
          double ds1_zeta_z_r_dtk = ds1_zeta_dz * dz_drp * drp_dtk;

          ds1_zeta_z_t_theta_j_dd += ds1_zeta_z_dtk * dtk_dtheta_j;
          ds1_zeta_z_t_theta_j_plus_1_dd += ds1_zeta_z_dtk * dtk_dtheta_j_plus_1;
          ds1_zeta_z_r_t_theta_j_dd += ds1_zeta_z_r_dtk * dtk_dtheta_j;
          ds1_zeta_z_r_t_theta_j_plus_1_dd += ds1_zeta_z_r_dtk * dtk_dtheta_j_plus_1;

          double deta_drp = 2. * rp_tk - d * costkmnu;
          double deta_drdash = -d * sintkmnu;
          double drdash_dtk = this->d2rp_dtheta2(t_k);
          double deta_dtk = d * (rp_tk * sintkmnu - drp_dtk * costkmnu);
          double ds1_eta_dtk = ds1_deta * deta_dtk;
          double ds1_eta_rp_dtk = ds1_deta * deta_drp * drp_dtk;
          double ds1_eta_rdash_dtk = ds1_deta * deta_drdash * drdash_dtk;

          ds1_eta_t_theta_j_dd += ds1_eta_dtk * dtk_dtheta_j;
          ds1_eta_t_theta_j_plus_1_dd += ds1_eta_dtk * dtk_dtheta_j_plus_1;
          ds1_eta_r_t_theta_j_dd += ds1_eta_rp_dtk * dtk_dtheta_j;
          ds1_eta_r_t_theta_j_plus_1_dd += ds1_eta_rp_dtk * dtk_dtheta_j_plus_1;
          ds1_eta_rdash_t_theta_j_dd += ds1_eta_rdash_dtk * dtk_dtheta_j;
          ds1_eta_rdash_t_theta_j_plus_1_dd += ds1_eta_rdash_dtk * dtk_dtheta_j_plus_1;
        }
      }
    }
    s1 += half_theta_range * s1_planet;

    if (_require_gradients == true) {
      // todo consolidate above to _dz (eg. for multiple _dnu etc.).

      ds1_theta_j_dd += -fractions::one_half * s1_planet * dthetas_dd[_j];
      ds1_theta_j_plus_1_dd += fractions::one_half * s1_planet * dthetas_dd[_j + 1];

      ds1_dd = ds1_theta_j_dd + ds1_theta_j_plus_1_dd + half_theta_range * (
        ds1_zeta_z_dd + ds1_eta_dd
        + dthetas_dd[_j] * (ds1_zeta_z_t_theta_j_dd
                            + ds1_zeta_z_r_t_theta_j_dd
                            + ds1_eta_t_theta_j_dd
                            + ds1_eta_r_t_theta_j_dd
                            + ds1_eta_rdash_t_theta_j_dd)
        + dthetas_dd[_j + 1] * (ds1_zeta_z_t_theta_j_plus_1_dd
                                + ds1_zeta_z_r_t_theta_j_plus_1_dd
                                + ds1_eta_t_theta_j_plus_1_dd
                                + ds1_eta_r_t_theta_j_plus_1_dd
                                + ds1_eta_rdash_t_theta_j_plus_1_dd));

      ds1_dnu = 0.;

      ds1_dcs.setZero();
    }
  } else {
    // Limb-darkening half-integer and odd terms n=1/2, 1, 3/2, using
    // Gauss-legendre quad.
    double s12_planet = 0.;
    double s32_planet = 0.;
    for (int k = 0; k < _N_l; k++) {

      // Rescale legendre root.
      double t_k = half_theta_range * (_l_roots[k] + 1.) + _theta_j;

      // Evaluate integrand at t_k.
      double rp_tk = this->rp_theta(t_k);
      double rp_tks = rp_tk * rp_tk;
      double drp_dtk = this->drp_dtheta(t_k);
      double sintkmnu = std::sin(t_k - nu);
      double costkmnu = std::cos(t_k - nu);
      double d_rp_costkmnu = d * rp_tk * costkmnu;
      double d_drpdtheta_sintkmnu = d * drp_dtk * sintkmnu;
      double zp_tks = _omdd - rp_tks + 2. * d_rp_costkmnu;
      double zp_tk = std::sqrt(zp_tks);
      double omzp_tks = 1 - zp_tks;
      double zeta12 = (1. - std::pow(zp_tk, fractions::five_halves))
                       / (fractions::five_halves * omzp_tks);
      double zeta = (1. - zp_tks * zp_tk) / (3. * omzp_tks);
      double zeta32 = (1. - std::pow(zp_tk, fractions::seven_halves))
                       / (fractions::seven_halves * omzp_tks);
      double eta = rp_tks - d_rp_costkmnu - d_drpdtheta_sintkmnu;
      double eta_w = eta * _l_weights[k];
      s12_planet += zeta12 * eta_w;
      s1_planet += zeta * eta_w;
      s32_planet += zeta32 * eta_w;

      if (_require_gradients == true) {
        // todo: add s12 and s23 terms.

        double opzp = 1 + zp_tk;
        double ds1_dzeta = eta * _l_weights[k];
        double dzeta_dzp = fractions::one_third * (1. - 1. / (opzp * opzp));
        double dzp_dd = (rp_tk * costkmnu - d) / zp_tk;
        double ds1_deta = zeta * _l_weights[k];
        double deta_dd = -rp_tk * costkmnu - drp_dtk * sintkmnu;

        ds1_zeta_z_dd += ds1_dzeta * dzeta_dzp * dzp_dd;
        ds1_eta_dd += ds1_deta * deta_dd;

        if (theta_type_j == intersections::planet) {

          double dtk_dtheta_j = 1. - fractions::one_half * (_l_roots[k] + 1.);
          double dtk_dtheta_j_plus_1 = fractions::one_half * (_l_roots[k] + 1.);

          double dz_drp = (d * costkmnu - rp_tk) / zp_tk;
          double dz_dtk = -d * rp_tk * sintkmnu / zp_tk;
          double ds1_zeta_dz = ds1_dzeta * dzeta_dzp;
          double ds1_zeta_z_dtk = ds1_zeta_dz * dz_dtk;
          double ds1_zeta_z_r_dtk = ds1_zeta_dz * dz_drp * drp_dtk;

          ds1_zeta_z_t_theta_j_dd += ds1_zeta_z_dtk * dtk_dtheta_j;
          ds1_zeta_z_t_theta_j_plus_1_dd += ds1_zeta_z_dtk * dtk_dtheta_j_plus_1;
          ds1_zeta_z_r_t_theta_j_dd += ds1_zeta_z_r_dtk * dtk_dtheta_j;
          ds1_zeta_z_r_t_theta_j_plus_1_dd += ds1_zeta_z_r_dtk * dtk_dtheta_j_plus_1;

          double deta_drp = 2. * rp_tk - d * costkmnu;
          double deta_drdash = -d * sintkmnu;
          double drdash_dtk = this->d2rp_dtheta2(t_k);
          double deta_dtk = d * (rp_tk * sintkmnu - drp_dtk * costkmnu);
          double ds1_eta_dtk = ds1_deta * deta_dtk;
          double ds1_eta_rp_dtk = ds1_deta * deta_drp * drp_dtk;
          double ds1_eta_rdash_dtk = ds1_deta * deta_drdash * drdash_dtk;

          ds1_eta_t_theta_j_dd += ds1_eta_dtk * dtk_dtheta_j;
          ds1_eta_t_theta_j_plus_1_dd += ds1_eta_dtk * dtk_dtheta_j_plus_1;
          ds1_eta_r_t_theta_j_dd += ds1_eta_rp_dtk * dtk_dtheta_j;
          ds1_eta_r_t_theta_j_plus_1_dd += ds1_eta_rp_dtk * dtk_dtheta_j_plus_1;
          ds1_eta_rdash_t_theta_j_dd += ds1_eta_rdash_dtk * dtk_dtheta_j;
          ds1_eta_rdash_t_theta_j_plus_1_dd += ds1_eta_rdash_dtk * dtk_dtheta_j_plus_1;
        }
      }
    }
    s12 += half_theta_range * s12_planet;
    s1 += half_theta_range * s1_planet;
    s32 += half_theta_range * s32_planet;

    if (_require_gradients == true) {
      // todo consolidate above to _dz (eg. for multiple _dnu etc.).

      ds1_theta_j_dd += -fractions::one_half * s1_planet * dthetas_dd[_j];
      ds1_theta_j_plus_1_dd += fractions::one_half * s1_planet * dthetas_dd[_j + 1];

      ds12_dd = 0.;
      ds1_dd = 0.;
      ds32_dd = 0.;

      ds1_dd = ds1_theta_j_dd + ds1_theta_j_plus_1_dd + half_theta_range * (
        ds1_zeta_z_dd + ds1_eta_dd
        + dthetas_dd[_j] * (ds1_zeta_z_t_theta_j_dd
                            + ds1_zeta_z_r_t_theta_j_dd
                            + ds1_eta_t_theta_j_dd
                            + ds1_eta_r_t_theta_j_dd
                            + ds1_eta_rdash_t_theta_j_dd)
        + dthetas_dd[_j + 1] * (ds1_zeta_z_t_theta_j_plus_1_dd
                                + ds1_zeta_z_r_t_theta_j_plus_1_dd
                                + ds1_eta_t_theta_j_plus_1_dd
                                + ds1_eta_r_t_theta_j_plus_1_dd
                                + ds1_eta_rdash_t_theta_j_plus_1_dd));

      ds12_dnu = 0.;
      ds1_dnu = 0.;
      ds32_dnu = 0.;

      ds12_dcs.setZero();
      ds1_dcs.setZero();
      ds32_dcs.setZero();
    }
  }
}


void Fluxes::s_star(int _j, int theta_type_j, double &_theta_j,
                    double &_theta_j_plus_1, const double &d,
                    const double &nu) {
  double phi_j;
  double phi_j_plus_1;

  // Check if entire star -pi to pi.
  if (theta_type_j == intersections::entire_star) {
    phi_j = -fractions::pi;
    phi_j_plus_1 = fractions::pi;
  } else {
    // Convert theta_j to phi_j (stellar centred frame).
    double rp_theta_j = this->rp_theta(_theta_j);
    double theta_jmnu = _theta_j - nu;
    phi_j = std::atan2(-rp_theta_j * std::sin(theta_jmnu),
                       -rp_theta_j * std::cos(theta_jmnu) + d);

    // Convert theta_j_plus_1 to phi_j_plus_1.
    double rp_theta_j_plus_1 = this->rp_theta(_theta_j_plus_1);
    double theta_j_plus_1mnu = _theta_j_plus_1 - nu;
    phi_j_plus_1 = std::atan2(
      -rp_theta_j_plus_1 * std::sin(theta_j_plus_1mnu),
      -rp_theta_j_plus_1 * std::cos(theta_j_plus_1mnu) + d);
  }

  // Evaluate line integral anticlockwise.
  double phi_diff = phi_j_plus_1 - phi_j;
    if (_ld_law == limb_darkening::quadratic) {
    // Limb-darkening terms n=0,1,2.
    s0 += fractions::one_half * phi_diff;
    s1 += fractions::one_third * phi_diff;
    s2 += fractions::one_quarter * phi_diff;

  } else {
    // Limb-darkening terms n=0,1/2,1,3/2,2.
    s0 += fractions::one_half * phi_diff;
    s12 += fractions::two_fifths * phi_diff;
    s1 += fractions::one_third * phi_diff;
    s32 += fractions::two_sevenths * phi_diff;
    s2 += fractions::one_quarter * phi_diff;
  }

  if (_require_gradients == true) {
    double dsn_phi_j_dd = 0.,
           dsn_phi_j_plus_1_dd = 0.,
           dsn_phi_j_theta_j_dd = 0.,
           dsn_phi_j_plus_1_theta_j_plus_1_dd = 0.,
           dsn_phi_j_rp_theta_j_dd = 0.,
           dsn_phi_j_plus_1_rp_theta_j_plus_1_dd = 0.;
    if (theta_type_j == intersections::star) {

      double _rp_j = this->rp_theta(_theta_j);
      double sin_theta_jmnu = std::sin(_theta_j - nu);
      double cos_theta_jmnu = std::cos(_theta_j - nu);
      double dphi_j_denom = _dd - 2. * d * _rp_j * cos_theta_jmnu
                            + _rp_j * _rp_j;

      double _rp_j_plus_1 = this->rp_theta(_theta_j_plus_1);
      double sin_theta_j_plus_1mnu = std::sin(_theta_j_plus_1 - nu);
      double cos_theta_j_plus_1mnu = std::cos(_theta_j_plus_1 - nu);
      double dphi_j_plus_1_denom = _dd - 2. * d * _rp_j_plus_1
                                   * cos_theta_j_plus_1mnu
                                   + _rp_j_plus_1 * _rp_j_plus_1;

      double _dphi_j_dd = (_rp_j * sin_theta_jmnu) / dphi_j_denom;
      double _dphi_j_dtheta_j = (_rp_j * _rp_j - d * _rp_j * cos_theta_jmnu)
                                / dphi_j_denom;
      double _dphi_j_drp = (-d * sin_theta_jmnu) / dphi_j_denom;
      double _drp_dtheta_j = this->drp_dtheta(_theta_j);

      double _dphi_j_plus_1_dd = (_rp_j_plus_1 * sin_theta_j_plus_1mnu)
                                 / dphi_j_plus_1_denom;
      double _dphi_j_plus_1_dtheta_j =
        (_rp_j_plus_1 * _rp_j_plus_1 - d * _rp_j_plus_1
         * cos_theta_j_plus_1mnu) / dphi_j_plus_1_denom;
      double _dphi_j_plus_1_drp = (-d * sin_theta_j_plus_1mnu)
                                  / dphi_j_plus_1_denom;
      double _drp_dtheta_j_plus_1 = this->drp_dtheta(_theta_j_plus_1);

      dsn_phi_j_dd = _dphi_j_dd;
      dsn_phi_j_plus_1_dd = _dphi_j_plus_1_dd;
      dsn_phi_j_theta_j_dd = _dphi_j_dtheta_j * dthetas_dd[_j];
      dsn_phi_j_plus_1_theta_j_plus_1_dd = _dphi_j_plus_1_dtheta_j
                                           * dthetas_dd[_j + 1];
      dsn_phi_j_rp_theta_j_dd = _dphi_j_drp * _drp_dtheta_j * dthetas_dd[_j];
      dsn_phi_j_plus_1_rp_theta_j_plus_1_dd = _dphi_j_plus_1_drp
                                              * _drp_dtheta_j_plus_1
                                              * dthetas_dd[_j + 1];
    }

    double dsn_dd = dsn_phi_j_dd
                    - dsn_phi_j_plus_1_dd
                    + dsn_phi_j_theta_j_dd
                    - dsn_phi_j_plus_1_theta_j_plus_1_dd
                    + dsn_phi_j_rp_theta_j_dd
                    - dsn_phi_j_plus_1_rp_theta_j_plus_1_dd;

    if (_ld_law == limb_darkening::quadratic) {
      ds0_dd += fractions::one_half * dsn_dd;
      ds1_dd += fractions::one_third * dsn_dd;
      ds2_dd += fractions::one_quarter * dsn_dd;

      ds0_dnu += 0.;
      ds1_dnu += 0.;
      ds2_dnu += 0.;

      ds0_dcs.setZero();
      ds1_dcs.setZero();
      ds2_dcs.setZero();
    } else {
      ds0_dd += fractions::one_half * dsn_dd;
      ds12_dd += fractions::two_fifths * dsn_dd;
      ds1_dd += fractions::one_third * dsn_dd;
      ds32_dd += fractions::two_sevenths * dsn_dd;
      ds2_dd += fractions::one_quarter * dsn_dd;

      ds0_dnu += 0.;
      ds12_dnu += 0.;
      ds1_dnu += 0.;
      ds32_dnu += 0.;
      ds2_dnu += 0.;

      ds0_dcs.setZero();
      ds12_dcs.setZero();
      ds1_dcs.setZero();
      ds32_dcs.setZero();
      ds2_dcs.setZero();
    }
  }
}


void Fluxes::f_derivatives(const double* dd_dz[], const double* dnu_dz[],
                           double* df_dz[]) {

  double dalpha_dI0 = alpha;
  if (_ld_law == limb_darkening::quadratic) {

    // df_dt0, df_dp, df_da, df_di, df_de, df_dw.
    double df_dd = df_dalpha * (dalpha_ds0 * ds0_dd + dalpha_ds1 * ds1_dd
                                + dalpha_ds2 * ds2_dd);
    double df_dnu = df_dalpha * (dalpha_ds0 * ds0_dnu + dalpha_ds1 * ds1_dnu
                                 + dalpha_ds2 * ds2_dnu);
    *df_dz[0] = df_dd * *dd_dz[0] + df_dnu * *dnu_dz[0];
    *df_dz[1] = df_dd * *dd_dz[1] + df_dnu * *dnu_dz[1];
    *df_dz[2] = df_dd * *dd_dz[2] + df_dnu * *dnu_dz[2];
    *df_dz[3] = df_dd * *dd_dz[3] + df_dnu * *dnu_dz[3];
    *df_dz[4] = df_dd * *dd_dz[4] + df_dnu * *dnu_dz[4];
    *df_dz[5] = df_dd * *dd_dz[5] + df_dnu * *dnu_dz[5];

    // df_du1, df_du2.
    double dalpha_du1 = I_0 * (s1 - s0);
    double dalpha_du2 = I_0 * (2. * s1 - s0 - s2);
    *df_dz[6] = df_dalpha * (dalpha_dI0 * dI0_du1 + dalpha_du1);
    *df_dz[7] = df_dalpha * (dalpha_dI0 * dI0_du2 + dalpha_du2);

    // df_drs.
    df_dcs = df_dalpha * (dalpha_ds0 * ds0_dcs + dalpha_ds1 * ds1_dcs
                          + dalpha_ds2 * ds2_dcs);
    for (int n = 0; n < N_c + 1; n++) {
      if (n == 0){
        *df_dz[8] = (df_dcs(N_c) * dc0_da0).real();
      } else {
        *df_dz[7 + 2 * n] = (df_dcs(n + N_c) * dcplus_dan
                             + df_dcs(-n + N_c) * dcminus_dan).real();
        *df_dz[8 + 2 * n] = (df_dcs(n + N_c) * dcplus_dbn
                             + df_dcs(-n + N_c) * dcminus_dbn).real();
      }
    }
  } else {

    // df_dt0, df_dp, df_da, df_di, df_de, df_dw.
    double df_dd = df_dalpha * (
      dalpha_ds0 * ds0_dd + dalpha_ds12 * ds12_dd + dalpha_ds1 * ds1_dd
      + dalpha_ds32 * ds32_dd + dalpha_ds2 * ds2_dd);
    double df_dnu = df_dalpha * (
      dalpha_ds0 * ds0_dnu + dalpha_ds12 * ds12_dnu + dalpha_ds1 * ds1_dnu
      + dalpha_ds32 * ds32_dnu + dalpha_ds2 * ds2_dnu);
    *df_dz[0] = df_dd * *dd_dz[0] + df_dnu * *dnu_dz[0];
    *df_dz[1] = df_dd * *dd_dz[1] + df_dnu * *dnu_dz[1];
    *df_dz[2] = df_dd * *dd_dz[2] + df_dnu * *dnu_dz[2];
    *df_dz[3] = df_dd * *dd_dz[3] + df_dnu * *dnu_dz[3];
    *df_dz[4] = df_dd * *dd_dz[4] + df_dnu * *dnu_dz[4];
    *df_dz[5] = df_dd * *dd_dz[5] + df_dnu * *dnu_dz[5];

    // df_du1, df_du2, df_du3, df_du4.
    double dalpha_du1 = I_0 * (s1 - s0);
    double dalpha_du2 = I_0 * (s12 - s0);
    double dalpha_du3 = I_0 * (s1 - s0);
    double dalpha_du4 = I_0 * (s32 - s0);
    *df_dz[6] = df_dalpha * (dalpha_dI0 * dI0_du1 + dalpha_du1);
    *df_dz[7] = df_dalpha * (dalpha_dI0 * dI0_du2 + dalpha_du2);
    *df_dz[8] = df_dalpha * (dalpha_dI0 * dI0_du3 + dalpha_du3);
    *df_dz[9] = df_dalpha * (dalpha_dI0 * dI0_du4 + dalpha_du4);

    // df_drs.
    df_dcs = df_dalpha * (dalpha_ds0 * ds0_dcs + dalpha_ds1 * ds1_dcs
                          + dalpha_ds2 * ds2_dcs);
    for (int n = 0; n < N_c + 1; n++) {
      if (n == 0){
        *df_dz[10] = (df_dcs(N_c) * dc0_da0).real();
      } else {
        *df_dz[9 + 2 * n] = (df_dcs(n + N_c) * dcplus_dan
                             + df_dcs(-n + N_c) * dcminus_dan).real();
        *df_dz[10 + 2 * n] = (df_dcs(n + N_c) * dcplus_dbn
                              + df_dcs(-n + N_c) * dcminus_dbn).real();
      }
    }
  }
}


void Fluxes::select_legendre_order(const double &d) {
  // Select regime switch: centre or edge.
  int position_switch;
  double outer_radii = max_rp + d;
  if (outer_radii >= 0.99) {
    position_switch = _precision_nl_edge;
  } else {
    position_switch = _precision_nl_centre;
  }

  // Set precision by number of legendre roots utilised.
  if (position_switch == 50) {
    // Default for centre.
    _N_l = 50;
    _l_roots = legendre::roots_fifty;
    _l_weights = legendre::weights_fifty;
  } else if (position_switch == 500) {
    // Default for edge.
    _N_l = 500;
    _l_roots = legendre::roots_five_hundred;
    _l_weights = legendre::weights_five_hundred;
  } else if (position_switch == 200) {
    _N_l = 200;
    _l_roots = legendre::roots_two_hundred;
    _l_weights = legendre::weights_two_hundred;
  } else if (position_switch == 100) {
    _N_l = 100;
    _l_roots = legendre::roots_hundred;
    _l_weights = legendre::weights_hundred;
  } else if (position_switch == 20) {
    _N_l = 20;
    _l_roots = legendre::roots_twenty;
    _l_weights = legendre::weights_twenty;
  } else if (position_switch == 10) {
    _N_l = 10;
    _l_roots = legendre::roots_ten;
    _l_weights = legendre::weights_ten;
  } else {
    // Fallback.
    _N_l = 500;
    _l_roots = legendre::roots_five_hundred;
    _l_weights = legendre::weights_five_hundred;
  }
}


void Fluxes::reset_intersections_integrals_and_derivatives() {
  // Reset intersections.
  theta = {};
  theta_type = {};
  dthetas_dd = {};

  // Reset integrals.
  s0 = 0., s12 = 0., s1 = 0., s32 = 0., s2 = 0.;

  if (_require_gradients == true) {
    // Reset derivatives.
    ds0_dd = 0., ds12_dd = 0., ds1_dd = 0., ds32_dd = 0., ds2_dd = 0.;
    ds0_dnu = 0., ds12_dnu = 0., ds1_dnu = 0., ds32_dnu = 0., ds2_dnu = 0.;
    ds0_dcs.setZero(), ds12_dcs.setZero(), ds1_dcs.setZero(),
                       ds32_dcs.setZero(), ds2_dcs.setZero();
  }
}


double Fluxes::rp_theta(double &_theta) {
  std::complex<double> rp = 0.;
  for (int n = -N_c; n < N_c + 1; n++) {
    rp += c(n + N_c) * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


void Fluxes::transit_flux(const double &d, const double &nu, double &f,
                          const double* dd_dz[], const double* dnu_dz[],
                          double* df_dz[]) {

  // Reset and pre-compute some position-specific quantities.
  // todo, check when derivatives can be safely cast to real values.
  std::cout << d << std::endl;
  this->reset_intersections_integrals_and_derivatives();
  this->select_legendre_order(d);
  this->pre_compute_psq(d, nu);

  // Find planet-stellar limb intersections.
  this->find_intersections_theta(d, nu);

  // Iterate thetas in adjacent pairs around the enclosed overlap region.
  for (int j = 0; j < theta_type.size(); j++) {

    if (theta_type[j] == intersections::planet
        || theta_type[j] == intersections::entire_planet) {
      // Line integrals, s_n, along planet limb segment.
      this->s_planet(j, theta_type[j], theta[j], theta[j + 1], d, nu);

    } else if (theta_type[j] == intersections::star
               || theta_type[j] == intersections::entire_star) {
      // Line integrals, s_n, along stellar limb segment.
      this->s_star(j, theta_type[j], theta[j], theta[j + 1], d, nu);

    } else {
      // Planet is beyond the stellar disc.
      break;
    }
  }

  // Compute transit flux: alpha=I0sTp.
  if (_ld_law == limb_darkening::quadratic) {
    alpha = I_0 * (s0 * p(0) + s1 * p(1) + s2 * p(2));
  } else {
    alpha = I_0 * (s0 * p(0) + s12 * p(1) + s1 * p(2)
                   + s32 * p(3) + s2 * p(4));
  }
  f = 1. - alpha;

  if (_require_gradients == true) {
    this->f_derivatives(dd_dz, dnu_dz, df_dz);
  }
}
