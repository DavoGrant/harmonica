#ifndef FLUXES_HPP
#define FLUXES_HPP

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * Fluxes class.
 */
class Fluxes {

  private:

    // Limb-darkening variables.
    double _ld_law;
    double I_0;
    Eigen::Vector<double, Eigen::Dynamic> p;

    // Transmission-string variables.
    int N_c;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> c;
    double min_rp;
    double max_rp;

    // Intersection variables.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> D;
    int C_shape;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> C;
    std::vector<double> theta;
    std::vector<double> theta_type;

    // Position variables.
    double _dd;
    std::complex<double> _d_expinu;
    std::complex<double> _d_expminu;

    // Derivatives switch.
    bool _require_gradients;

    /**
     * Companion matrix elements for computing the max and min radii of
     * a transmission string, D_jk(c).
     *
     * @param j row index, 1 <= j <= 2N_c.
     * @param k column index, 1 <= k <= 2N_c.
     * @param shape number of rows=cols of matrix.
     * @return complex matrix element.
     */
    std::complex<double> extrema_companion_matrix_D_jk(int j, int k,
                                                       int shape);

    /**
     * Companion matrix elements for computing the planet-star limb
     * intersections, C_jk(H), but only for terms that are independent
     * of the relative position, d and nu.
     *
     * @param j row index, 1 <= j <= 4N_c.
     * @param k column index, 1 <= k <= 4N_c.
     * @param shape number of rows=cols of matrix.
     * @return complex matrix element.
     */
    std::complex<double> intersection_companion_matrix_C_jk_base(
      int j, int k, int shape);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j,
     * but only for terms that are independent of the relative position,
     * d and nu.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @return complex polynomial coefficient.
     */
    std::complex<double> intersection_polynomial_coefficients_h_j_base(int j);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j,
     * but only update terms based on the relative position, d and nu. All
     * these values are added to the final column of the companion matrix
     * that will already have been build using the _base methods.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @return complex polynomial coefficient.
     */
    std::complex<double> intersection_polynomial_coefficients_h_j_update(
      int j);

    /**
     * Complex polynomial coefficient for the intersection equation, but
     * only the h_4Nc term. Here we return -1/h_4Nc.
     *
     * @param j polynomial term exponent, j = 4N_c.
     * @return complex polynomial coefficient.
     */
    std::complex<double> intersection_polynomial_coefficient_moo_denom(
      int j);

    /**
     * Compute the real roots, as a vector of thetas, from a given companion
     * matrix. The real roots correspond to angles in the complex plane where
     * the matrix eigenvalues lie on the unit circle.
     *
     * @param companion_matrix the complex-valued companion matrix.
     * @param shape number of rows=cols of matrix and complex eigenvalues.
     * @return vector of real roots in theta.
     */
    std::vector<double> compute_real_theta_roots(
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        companion_matrix, int shape);

    /**
     * Find and characterise the planet-stellar limb intersections vector,
     * theta, and sort in ascending order, -pi < theta <= pi. Each adjacent
     * pair of thetas corresponds to a segment of the closed loop piecewise
     * integral around the overlap region. These pairs are assigned as either
     * planet=0 or star=1 integral types.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return void.
     */
    void find_intersections_theta(const double &d, const double &nu);

  public:

    /**
     * Constructor.
     *
     * @param ld_law limb darkening law, 0=quadratic, 1=non-linear.
     * @param us array of stellar limb darkening coefficients [].
     * @param rs array of planet radius harmonic coefficients [stellar radii].
     * @param require_gradients derivatives switch.
     */
    Fluxes(int ld_law,
           py::array_t<double, py::array::c_style> us,
           py::array_t<double, py::array::c_style> rs,
           bool require_gradients);

    /**
     * Compute the planet radius at a given theta.
     *
     * @param theta angle in the terminator plane from v_orb [radians].
     * @return rp, the planet radius, is always real.
     */
    double rp_theta(double _theta);

    /**
     * Compute normalised transit flux.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @param f empty normalised light curve flux [].
     * @param dd_dz array of derivatives dd/dz z={t0, p, a, i, e, w}.
     * @param dnu_dz array of derivatives dnu/dz z={t0, p, a, i, e, w}.
     * @param df_dz empty array of derivatives df/dz z={t0, p, a, i, e, w, us, rs}.
     * @return void.
     */
    void transit_flux(const double &d, const double &nu, double &f,
                      const double* dd_dz[], const double* dnu_dz[],
                      double* df_dz[]);


};


#endif
