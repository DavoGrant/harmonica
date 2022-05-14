#include <cmath>
#include <vector>
#include <iostream>
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
  const int D_shape = 2 * N_c;
  D.resize(D_shape, D_shape);
  for (int j = 1; j < D_shape + 1; j++) {
    for (int k = 1; k < D_shape + 1; k++) {
      D(j - 1, k - 1) = this->extrema_companion_matrix_D_jk(j, k, D_shape);
    }
  }
  std::vector<double> theta_extrema = this->compute_real_theta_roots(D, D_shape);


  // Pre-compute intersection eqn companion matrix where terms are
  // independent of d and nu.
  const int C_shape = 4 * N_c;
  C.resize(C_shape, C_shape);
  for (int j = 1; j < C_shape + 1; j++) {
    for (int k = 1; k < C_shape + 1; k++) {
      C(j - 1, k - 1) = 1. - 2.i;
      // C_jk private method.
      // and h_j pre-compute func. ie. just final col first summations indep of d, nu.
      // later have an update companion matrix method, that adds dep d, nu terms.
    }
  }
}


std::complex<double> Fluxes::extrema_companion_matrix_D_jk(int j, int k,
                                                           const int shape) {
  // NB. matrix elements are one-indexed.
  // Also, c_0 requires c(0 + N_c) as it runs -N_c through N_c.
  if (k == shape) {
    return -(j - N_c - 1.) * c(j - 1) / (1. * N_c * c(shape));
  } else {
    if (j == k + 1) {
      return 1.;
    } else {
      return 0.;
    }
  }
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
    if (tolerance::unit_circle_lo < ev_abs && ev_abs <
                                              tolerance::unit_circle_hi) {
      theta.push_back(std::arg(evs(j)));
    }
  }
  return theta;
}


void Fluxes::transit_flux(const double &d, const double &nu, double &f,
                          const double* dd_dz[], const double* dnu_dz[],
                          double* df_dz[]) {

//    std::cout << I_0 << std::endl;
//    std::cout << p << std::endl;
//    std::cout << c << std::endl;
    std::cout << D << std::endl;
//    std::cout << C << std::endl;

}
