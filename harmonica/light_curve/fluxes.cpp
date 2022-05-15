#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
    I_0 = 1 / ((1 - us_(0) / 3. - us_(1) / 6.) * M_PI);

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
                - 3. * us_(2) / 7. - us_(3) / 2.) * M_PI);

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

  // Pre-build the intersection eqn companion matrix for the terms that
  // are independent of position, d and nu.
  const int C_shape = 4 * N_c;
  C.resize(C_shape, C_shape);
  for (int j = 1; j < C_shape + 1; j++) {
    for (int k = 1; k < C_shape + 1; k++) {
      C(j - 1, k - 1) = this->intersection_companion_matrix_C_jk_base(
        j, k, C_shape);
    }
  }

  // Todo: more pre-compute
  // eg. c conv c.
}


std::complex<double> Fluxes::extrema_companion_matrix_D_jk(int j, int k,
                                                           const int shape) {
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
  int j, int k, const int shape) {
  // NB. matrix elements are one-indexed.
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  std::complex<double> moo_denom = -1.
    / this->intersection_polynomial_coefficients_h_j_base(shape);
  if (k == shape) {
    return this->intersection_polynomial_coefficients_h_j_base(j - 1) * moo_denom;
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
  std::complex<double> h_j = 0.;
  // NB. verbose on purpose.
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  if (0 <= j && j < N_c - 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (N_c - 1 <= j && j < N_c + 1) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (N_c + 1 <= j && j < 2 * N_c) {
    for (int n = -N_c; n < -N_c + j + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (j == 2 * N_c) {
    for (int n = -N_c; n < N_c + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c) - 1.;
    }
  } else if (2 * N_c + 1 <= j && j < 3 * N_c) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (3 * N_c <= j && j < 3 * N_c + 2) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  } else if (3 * N_c + 2 <= j && j < 4 * N_c + 1) {
    for (int n = -3 * N_c + j; n < N_c + 1; n++) {
      h_j += c(n + N_c) * c(j - n - N_c);
    }
  }
  return h_j;
}


std::vector<double> Fluxes::compute_real_theta_roots(
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
    companion_matrix, const int shape) {

  // Solve eigenvalues.
  Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<double>,
    Eigen::Dynamic, Eigen::Dynamic>> ces;
  ces.compute(companion_matrix, false);
  Eigen::Vector<std::complex<double>, Eigen::Dynamic> evs = ces.eigenvalues();

  // Select real thetas only: angle to unit circle in complex plane.
  std::vector<double> theta;
  for (int j = 0; j < shape; j++) {
    double ev_abs = std::abs(evs(j));
    if (tolerance::unit_circle_lo < ev_abs
        && ev_abs < tolerance::unit_circle_hi) {
      theta.push_back(std::arg(evs(j)));
    }
  }
  return theta;
}


double Fluxes::rp_theta(double theta) {
  std::complex<double> rp = 0.;
  for (int n = -N_c; n < N_c + 1; n++) {
    rp += c(n + N_c) * std::exp((1. * n) * 1.i * theta);
  }
  return rp.real();
}


void Fluxes::transit_flux(const double &d, const double &nu, double &f,
                          const double* dd_dz[], const double* dnu_dz[],
                          double* df_dz[]) {

//    std::cout << I_0 << std::endl;
//    std::cout << p << std::endl;
//    std::cout << c << std::endl;
//    std::cout << D << std::endl;
    std::cout << C << std::endl;

}
