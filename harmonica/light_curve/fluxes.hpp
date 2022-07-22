#ifndef FLUXES_HPP
#define FLUXES_HPP

#include <vector>
#include <Eigen/Core>


#define EigD Eigen::Dynamic


/**
 * Fluxes class.
 */
class Fluxes {

  public:

    /**
     * Constructor.
     *
     * @param ld_law limb darkening law, 0=quadratic, 1=non-linear.
     * @param us array of stellar limb darkening coefficients [].
     * @param n_rs number of harmonic coefficients.
     * @param rs array of planet radius harmonic coefficients [stellar radii].
     * @param pnl_c N_l precision for planet inside stellar disc.
     * @param pnl_e N_l precision for planet intersecting stellar disc.
     */
    Fluxes(int ld_law, double us[], int n_rs, double rs[],
           int pnl_c, int pnl_e);

    /**
     * Compute the planet radius at a given theta.
     *
     * @param theta angle in the terminator plane from v_orb [radians].
     * @return rp, the planet radius, is always real.
     */
    double rp_theta(const double _theta);

    /**
     * Compute normalised transit flux.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param z distance from sky-plane, if z < 0 no transits [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @param out_f normalised light curve flux [].
     * @return void.
     */
    void transit_flux(double d, const double z,
                      const double nu, double& out_f);

  protected:

    // Limb-darkening variables.
    int m_ld_law;
    double m_I_0, m_I_0_bts;
    Eigen::Vector<double, EigD> m_p;

    // Transmission-string variables.
    int m_n_rs, m_N_c;
    double m_min_rp, m_max_rp;
    Eigen::Vector<std::complex<double>, EigD> m_c;

    // Intersection variables.
    int m_C_shape;
    std::vector<int> m_theta_type;
    std::vector<double> m_theta;
    Eigen::Matrix<std::complex<double>, EigD, EigD> m_D, m_C0;

    // Position and integral variables.
    int m_N_l, m_N_q0, m_N_q2, m_len_c_conv_c, m_len_beta_conv_c,
        m_len_q_rhs, m_mid_q_lhs, m_len_q;
    const double *m_l_roots, *m_l_weights;
    double m_td, m_dd, m_omdd, m_alpha, m_s0, m_s12, m_s1, m_s32, m_s2;
    std::complex<double> m_expinu, m_expminu, m_d_expinu, m_d_expminu;
    Eigen::Vector<std::complex<double>, 3> m_beta_sin0, m_beta_cos0;
    Eigen::Vector<std::complex<double>, EigD> m_c_conv_c, m_Delta_ew_c;

    // Precision switches.
    int m_precision_nl_centre, m_precision_nl_edge;

    /**
     * Compute solution vector S for each element in the polynomial
     * limb-darkening basis.
     */
    void compute_solution_vector(const double d, const double z,
                                 const double nu);

    /**
     * Find and characterise the planet-stellar limb intersections
     * vector, theta, and sort in ascending order, -pi < theta <= pi.
     * Each adjacent pair of thetas corresponds to a segment of the
     * closed loop piecewise integral around the overlap region.
     * These pairs are assigned as either planet=0, entire_planet=1,
     * star=2, entire_star=3, or beyond=4 (flux=0).
     */
    virtual void find_intersections_theta(const double d, const double nu);

    /**
     * Compute the line integrals s_n along a segment of the
     * planet's limb from theta_j to theta_j_p1 anticlockwise.
     */
    void s_planet(int _j, int theta_type_j, double _theta_j,
                  double _theta_j_p1, const double d,
                  const double nu);

    /**
     * Compute the line integrals s_n along a segment of the
     * star's limb from theta_j to theta_j_p1 anticlockwise.
     */
    virtual void s_star(int _j, int theta_type_j, double _theta_j,
                        double _theta_j_p1, const double d,
                        const double nu);

    /**
     * Select the order of legendre polynomial to use in numerical
     * evaluation of an integral. Order n integrates polynomial
     * integrands of order 2n - 1 exactly.
     */
    void select_legendre_order(const double d);

    /**
     * Reset all intersection and integral quantities to zero when
     * the position is new.
     */
    void reset_intersections_integrals();

    /**
     * Compute some position-specific quantities at start of each
     * new flux calculation, ie. avoiding recalc of exponentials.
     */
    void pre_compute_psq(const double d, const double nu);

    /**
     * Compute the distance to the stellar limb from the planet centred
     * coordinate system for a given d, nu, and theta. Both +- solutions
     * exist when d > 1, however we only take the + solution inside the
     * stellar disc, d < 1.
     */
    double rs_theta(const double d, double dcos_thetamnu,
                    int plus_solution);

    /**
     * Compute derivative of the planet radius wrt theta at a given theta.
     */
    double drp_dtheta(double _theta);

    /**
     * Compute second derivative of the planet radius wrt theta at a
     * given theta.
     */
    double d2rp_dtheta2(double _theta);

    /**
     * Companion matrix elements for computing the max and min radii of
     * a transmission string, D_jk(c).
     */
    std::complex<double> extrema_companion_matrix_D_jk(
        int j, int k, int shape);

    /**
     * Companion matrix elements for computing the planet-star limb
     * intersections, C_jk(H), but only for terms that are independent
     * of the relative position, d and nu.
     */
    std::complex<double> intersection_companion_matrix_C_jk_base(
        int j, int k, int shape);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j,
     * but only for terms that are independent of the relative position,
     * d and nu.
     */
    std::complex<double> intersection_polynomial_coefficients_h_j_base(
        int j);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j,
     * but only update terms based on the relative position, d and nu. All
     * these values are added to the final column of the companion matrix
     * that will already have been build using the _base methods.
     */
    std::complex<double> intersection_polynomial_coefficients_h_j_update(
        int j);

    /**
     * Complex polynomial coefficient for the intersection equation, but
     * only the h_4Nc term. Here we return -1/h_4Nc.
     */
    std::complex<double> intersection_polynomial_coefficient_moo_denom(
        int j);

    /**
     * Complex polynomial coefficients for the intersection equation, h_j.
     */
    std::complex<double> h_j(int j);

    /**
     * Quick check if there are no obvious intersections based on the planet
     * position and its max and min radius. This reduces the need to compute
     * the costly eigenvalues for much of a light curve.
     */
    bool no_obvious_intersections(const double d, const double nu);

    /**
     * Compute the real roots, as a vector of thetas, from a given companion
     * matrix. The real roots correspond to angles in the complex plane where
     * the matrix eigenvalues lie on the unit circle.
     */
    virtual std::vector<double> compute_real_theta_roots(
        const Eigen::Matrix<std::complex<double>, EigD, EigD>&
          companion_matrix, int shape);

    /**
     * If there are no intersections found check which trivial configuration
     * the system is in. For example, planet completely inside/outside the
     * the stellar disc, or visa versa. This is only hit when the check for
     * this->no_obvious_intersections() does not return true owing to the
     * coarseness of the max/min radius checking. Always true when there are
     * no intersections found.
     */
    bool trivial_configuration(const double d, const double nu);

    /**
     * Characterise the bodies limb that forms a segment of the closed loop
     * piecewise integral around the overlap region. These pairs are assigned
     * as either planet=0 or star=1 integral types. NB. the theta vector must
     * be pre-sorted in ascending order, and span the full two pi.
     */
    void characterise_intersection_pairs(const double d, const double nu);

    /**
     * Check intersection associations with either the T+ or T- intersection
     * equation.
     */
    void associate_intersections(int j, const double d,
                                 double dcos_thetamnu, int& out_T_theta_j);

    /**
     * Check intersection gradients are dT_dtheta+ or dT_dtheta- at
     * intersection.
     */
    void gradient_intersections(int j, double dsin_thetamnu,
                                double dcos_thetamnu, int plus_solution,
                                int& out_dT_dtheta_theta_j);

    /**
     * Convolve two 1d vectors of complex values fully. c = a (*) b with
     * a final size of len_a + len_b - 1
     */
    Eigen::Vector<std::complex<double>, EigD> complex_convolve(
        const Eigen::Vector<std::complex<double>, EigD>& a,
        const Eigen::Vector<std::complex<double>, EigD>& b,
        int len_a, int len_b, int len_c);

    /**
     * Add two 1d vectors of complex values centre-aligned. c = a + b with
     * a final size of max(len_a, len_b)
     */
    Eigen::Vector<std::complex<double>, EigD>
    complex_ca_vector_addition(
        const Eigen::Vector<std::complex<double>, EigD>& a,
        const Eigen::Vector<std::complex<double>, EigD>& b,
        int len_a, int len_b);

    /**
     * Compute the even terms in the planet limb's line integral.
     * These terms are closed form and rely on a succession of
     * convolutions before the integral is evaluated.
     */
    virtual void analytic_even_terms(
        int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
        const double d, const double nu);

    /**
     * Compute the odd and half-integer terms in the planet limb's
     * line integral. These terms do not have an obvious closed form
     * solution and therefore Gauss-legendre quadrature is employed.
     */
    virtual void numerical_odd_terms(
        int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
        const double d, const double nu);

};


#endif
