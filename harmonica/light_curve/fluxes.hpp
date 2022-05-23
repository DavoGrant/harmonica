#ifndef FLUXES_HPP
#define FLUXES_HPP

#include <Eigen/Core>
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
    int _n_rs;
    int N_c;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> c;
    double min_rp;
    double max_rp;

    // Intersection variables.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> D;
    int C_shape;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> C0;
    std::vector<double> theta;
    std::vector<int> theta_type;

    // Position and integral variables.
    double _dd;
    double _omdd;
    std::complex<double> _d_expinu;
    std::complex<double> _d_expminu;
    int _len_c_conv_c;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> _c_conv_c;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> _Delta_ew_c;
    int _len_beta_conv_c;
    Eigen::Vector<std::complex<double>, 3> _beta_sin0;
    Eigen::Vector<std::complex<double>, 3> _beta_cos0;
    int _len_q_rhs;
    int _mid_q_lhs;
    int _len_q;
    int N_q0;
    int N_q2;
    double _sp_star;

    // Derivatives switch.
    bool _require_gradients;

    /**
     * Compute the distance to the stellar limb from the planet centred
     * coordinate system, for a given d, nu, and theta. Both +- solutions
     * exist when d > 1, however we only take the + solution inside the
     * stellar disc, d < 1.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param dcos_thetamnu d*cos(theta - nu) [stellar radii].
     * @param plus_solution +=1, -=0, if d < 1 ignored.
     * @return rs, the stellar radius in the planet's frame.
     */
    double rs_theta(const double &d, double &dcos_thetamnu,
                    int plus_solution);

    /**
     * Compute derivative of the planet radius wrt theta at a given theta.
     *
     * @param theta angle in the terminator plane from v_orb [radians].
     * @return drp_dtheta, the derivative, is always real.
     */
    double drp_dtheta(double &_theta);

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
                                                       int &shape);

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
      int j, int k, int &shape);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j,
     * but only for terms that are independent of the relative position,
     * d and nu.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @return complex polynomial coefficient.
     */
    std::complex<double> intersection_polynomial_coefficients_h_j_base(
      int j);

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
     * Find and characterise the planet-stellar limb intersections vector,
     * theta, and sort in ascending order, -pi < theta <= pi. Each adjacent
     * pair of thetas corresponds to a segment of the closed loop piecewise
     * integral around the overlap region. These pairs are assigned as either
     * planet=0, star=1, or beyond=2 (flux=0) integral types.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return void.
     */
    void find_intersections_theta(const double &d, const double &nu);

    /**
     * Quick check if there are no obvious intersections based on the planet
     * position and its max and min radius. This reduces the need to compute
     * the costly eigenvalues for much of a light curve.
     *
     * @param d planet-star centre separation [stellar radii].
     * @return bool if there are no obvious intersections.
     */
    bool no_obvious_intersections(const double &d);

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
        companion_matrix, int &shape);

    /**
     * Check if there are no intersections found which trivial configuration
     * the system is in. For example, planet completely inside/outside the
     * the stellar disc, or visa versa. This is only hit when the check for
     * this->no_obvious_intersections() does not return true owing to the
     * coarseness of the max/min radius checking. Always true when there are
     * no intersections found.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return bool if there is a trivial configuration.
     */
    bool trivial_configuration(const double &d, const double &nu);

    /**
     * Characterise the bodies limb that forms a segment of the closed loop
     * piecewise integral around the overlap region. These pairs are assigned
     * as either planet=0 or star=1 integral types. NB. the theta vector must
     * be pre-sorted in ascending order, and span the full two pi.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return void.
     */
    void characterise_intersection_pairs(const double &d, const double &nu);

    /**
     * Check intersection associations with either the T+ or T- intersection
     * equation.
     *
     * @param j theta index.
     * @param d planet-star centre separation [stellar radii].
     * @param dcos_thetamnu d*cos(theta - nu) [stellar radii].
     * @param T_theta_j empty label, 0=-ve and 1=+ve association.
     * @return void.
     */
    void associate_intersections(int j, const double &d,
                                 double &dcos_thetamnu, int &T_theta_j);

    /**
     * Check intersection gradients are dT_dtheta+ or dT_dtheta- at
     * intersection.
     *
     * @param j theta index.
     * @param dsin_thetamnu d*din(theta - nu) [stellar radii].
     * @param dcos_thetamnu d*cos(theta - nu) [stellar radii].
     * @param plus_solution +=1, -=0.
     * @param dT_dtheta_theta_j empty label, 0=-ve and 1=+ve gradient.
     * @return void.
     */
    void gradient_intersections(int j, double &dsin_thetamnu,
                                double &dcos_thetamnu, int plus_solution,
                                int &dT_dtheta_theta_j);

    /**
     * Convolve two 1d vectors of complex values fully.
     *
     * @param a input vector.
     * @param b other input vector.
     * @param len_a size of vector a.
     * @param len_b size of vector b.
     * @param len_c size of the output vector c.
     * @return c = a (*) b with size len_a + len_b - 1.
     */
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> complex_convolve(
      Eigen::Vector<std::complex<double>, Eigen::Dynamic> a,
      Eigen::Vector<std::complex<double>, Eigen::Dynamic> b,
      int len_a, int len_b, int len_c);

    /**
     * Add two 1d vectors of complex values centre-aligned.
     *
     * @param a input vector.
     * @param b other input vector.
     * @param len_a size of vector a.
     * @param len_b size of vector b.
     * @return a + b with size max(len_a, len_b).
     */
    Eigen::Vector<std::complex<double>, Eigen::Dynamic>
    complex_ca_vector_addition(
      Eigen::Vector<std::complex<double>, Eigen::Dynamic> a,
      Eigen::Vector<std::complex<double>, Eigen::Dynamic> b,
      int len_a, int len_b);

    /**
     * Compute the sum of line integrals sTp along segments of the
     * planet's limb from theta_j to theta_j_plus_1. The sum is over
     * each limb darkening component.
     *
     * @param _theta_j start of line segment [radians].
     * @param _theta_j_plus_1 end of line segment [radians].
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return computed sTp_planet line integral.
     */
    double sTp_planet(double &_theta_j, double &_theta_j_plus_1,
                      const double &d, const double &nu);

    /**
     * Compute the sum of line integrals sTp along segments of the
     * stellar limb from theta_j to theta_j_plus_1. The sum is over
     * each limb darkening component.
     *
     * @param _theta_j start of line segment [radians].
     * @param _theta_j_plus_1 end of line segment [radians].
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return computed sTp_star line integral.
     */
    double sTp_star(double &_theta_j, double &_theta_j_plus_1,
                    const double &d, const double &nu);

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
    double rp_theta(double &_theta);

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
