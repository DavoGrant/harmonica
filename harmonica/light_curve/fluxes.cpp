#include <cmath>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// TODO: temp debug.
#include <iomanip>
#include <iostream>

#include "fluxes.hpp"
#include "../constants/constants.hpp"

using namespace std::complex_literals;


Fluxes::Fluxes(int ld_law,
               py::array_t<double, py::array::c_style> us,
               py::array_t<double, py::array::c_style> rs,
               bool require_gradients) {

  // Unpack python arrays.
  auto us_ = us.unchecked<1>();
  auto rs_ = rs.unchecked<1>();
  int n_rs = rs_.shape(0);  // Always odd.

  _ld_law = ld_law;
  if (_ld_law == 0) {
    // Normalisation.
    I_0 = 1 / ((1 - us_(0) / 3. - us_(1) / 6.) * fractions::pi);

    // Quadratic limb-darkening law.
    Eigen::Vector<double, 3> u {1, us_(0), us_(1)};
    Eigen::Matrix<double, 3, 3> B {{1., -1., -1.},
                                   {0., 1., 2.},
                                   {0., 0., -1.}};

    // Change to polynomial basis.
    p = B * u;

  } else if (_ld_law == 1) {
    // Normalisation.
    I_0 = 1 / ((1 - us_(0) / 5. - us_(1) / 3.
                - 3. * us_(2) / 7. - us_(3) / 2.) * fractions::pi);

    // Non-linear limb-darkening law.
    Eigen::Vector<double, 5> u {1, us_(0), us_(1), us_(2), us_(3)};
    Eigen::Matrix<double, 5, 5> B {{1., -1., -1., -1., -1.},
                                   {0., 1., 0., 0., 0.},
                                   {0., 0., 1., 0., 0.},
                                   {0., 0., 0., 1., 0.},
                                   {0., 0., 0., 0., 1.}};

    // Change to polynomial basis.
    p = B * u;
  }

  // Convert cosine, sine to complex Fourier coefficients.
  N_c = (n_rs - 1) * fractions::one_half;
  c.resize(n_rs);
  c(N_c) = rs_(0);
  for (int n = 0; n < N_c; n++) {
    double a_real = rs_(n_rs - 2 - 2 * n);
    double b_imag = rs_(n_rs - 1 - 2 * n);
    c(n) = (a_real + b_imag * 1.i) * fractions::one_half;
    c(n_rs - 1 - n) = (a_real - b_imag * 1.i) * fractions::one_half;
  }

  // Pre-compute max and min planet radii.
  min_rp = c(N_c).real();
  max_rp = c(N_c).real();
  if (N_c != 0) {
    // Build the extrema companion matrix.
    const int D_shape = 2 * N_c;
    D.resize(D_shape, D_shape);
    for (int j = 1; j < D_shape + 1; j++) {
      for (int k = 1; k < D_shape + 1; k++) {
        D(j - 1, k - 1) = this->extrema_companion_matrix_D_jk(j, k, D_shape);
      }
    }

    // Get the extrema companion matrix roots.
    std::vector<double> theta_extrema = this->compute_real_theta_roots(
      D, D_shape);

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
    C.resize(C_shape, C_shape);
    for (int j = 1; j < C_shape + 1; j++) {
      for (int k = 1; k < C_shape + 1; k++) {
        C(j - 1, k - 1) = this->intersection_companion_matrix_C_jk_base(
          j, k, C_shape);
      }
    }
  }

  // Todo: more pre-compute
  // eg. c conv c.
}


std::complex<double> Fluxes::extrema_companion_matrix_D_jk(int j, int k,
                                                           int shape) {
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
  int j, int k, int shape) {
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


void Fluxes::find_intersections_theta(const double &d, const double &nu) {

  // Check cases where no obvious intersections, avoiding eigenvalue runtime.
  if (d <= 1.) {
    // Planet centre inside stellar disc.
    if (max_rp <= 1. - d) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region enclosed by entire planet's limb.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::planet};
      return;
    } else if (min_rp >= 1. + d) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::star};
      return;
    }
  } else {
    // Planet centre outside stellar disc.
    if (max_rp <= d - 1.) {
      // Max planet radius would not intersect closest stellar limb.
      // Overlap region is zero.
      theta = {};
      theta_type = {intersections::beyond};
      return;
    } else if (min_rp >= d + 1.) {
      // Min planet radius beyond furthest stellar limb.
      // Overlap region enclosed by entire star's limb.
      theta = {-fractions::pi, fractions::pi};
      theta_type = {intersections::star};
      return;
    }
  }

  if (N_c != 0) {
    // Update intersection companion matrix for current position.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> _C = C;
    std::complex<double> moo_denom = intersection_polynomial_coefficient_moo_denom(C_shape);
    for (int j = 1; j < C_shape + 1; j++) {
      _C(j - 1, C_shape - 1) += this->intersection_polynomial_coefficients_h_j_update(j - 1);
      _C(j - 1, C_shape - 1) *= moo_denom;
    }

    // Get the intersection companion matrix roots.
    theta = this->compute_real_theta_roots(_C, C_shape);

  } else {
    double acos_intersect = std::acos(
      (c(N_c).real() * c(N_c).real() + _dd - 1.) / (2. * c(N_c).real() * d));
    theta = {nu - acos_intersect, nu + acos_intersect};
  }

  if (theta.size() == 0) {
    // No roots, check which trivial case this configuration corresponds to.
    double _rp_nu = this->rp_theta(nu);
    if (d <= 1.) {
      // Planet centre inside stellar disc.
      if (_rp_nu < 1. + d) {
        // Planet radius toward stellar centre closer than stellar limb.
        // Overlap region enclosed by entire planet's limb as no intersects.
        theta = {-fractions::pi, fractions::pi};
        theta_type = {intersections::planet};
        return;
      } else if (_rp_nu > 1. + d) {
        // Planet radius toward stellar centre beyond stellar limb.
        // Overlap region enclosed by entire star's limb as no intersects.
        theta = {-fractions::pi, fractions::pi};
        theta_type = {intersections::star};
        return;
      }
    } else {
      // Planet centre outside stellar disc.
      if (_rp_nu < 1. + d) {
        // Planet radius toward stellar centre closer than stellar limb.
        // Overlap region is zero as no intersects.
        theta = {};
        theta_type = {intersections::beyond};
        return;
      } else if (_rp_nu > 1. + d) {
        // Planet radius toward stellar centre beyond stellar limb.
        // Overlap region enclosed by entire star's limb as no intersects.
        theta = {-fractions::pi, fractions::pi};
        theta_type = {intersections::star};
        return;
      }
    }
  } else {
    // Sort roots in ascending order, -pi < theta <= pi.
    std::sort(theta.begin(), theta.end());

    // Ensure theta vector spans a closed loop, 2pi in total.
    // Thus, duplicate first intersection + 2pi at the end.
    theta.push_back(theta[0] + fractions::twopi);

    // Characterise theta pairs.
    for (int j = 0; j < theta.size() - 1; j++) {
      std::cout << std::setprecision(15) << theta[j] << ' ' << theta[j + 1] << std::endl;
    }
  }
}


std::vector<double> Fluxes::compute_real_theta_roots(
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
    companion_matrix, int shape) {

  // Solve eigenvalues.
  Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>,
    Eigen::Dynamic, Eigen::Dynamic>> ces;
  ces.compute(companion_matrix, false);
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> evs = ces.eigenvalues();

  // Select real thetas only: angle to point on unit circle in complex plane.
  std::vector<double> _theta;
  for (int j = 0; j < shape; j++) {
    double ev_abs = std::abs(evs(j));
    if (tolerance::unit_circle_lo < ev_abs
        && ev_abs < tolerance::unit_circle_hi) {
      _theta.push_back(std::arg(evs(j)));
    }
  }
  return _theta;
}


double Fluxes::rp_theta(double _theta) {
  std::complex<double> rp = 0.;
  for (int n = -N_c; n < N_c + 1; n++) {
    rp += c(n + N_c) * std::exp((1. * n) * 1.i * _theta);
  }
  return rp.real();
}


void Fluxes::transit_flux(const double &d, const double &nu, double &f,
                          const double* dd_dz[], const double* dnu_dz[],
                          double* df_dz[]) {

  // Pre-compute some position specific quantities.
  _dd = d * d;
  _d_expinu = d * std::exp(1.i * nu);
  _d_expminu = d * std::exp(-1.i * nu);

  // Find planet-stellar limb intersections, sorted theta.
  this->find_intersections_theta(d, nu);
  std::cout << std::setprecision(15) << theta.size() << std::endl;

  // Iterate thetas in adjacent pairs.
  // Iterate s_n terms.
  // Which way around to nest these..?

  // Todo: Ensure attributes are reset for new position,
  // Todo: or copies have been made.

}
