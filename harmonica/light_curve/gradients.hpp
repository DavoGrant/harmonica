#ifndef FLUXDERIVATIVES_HPP
#define FLUXDERIVATIVES_HPP

#include <vector>
#include <Eigen/Core>

#include "fluxes.hpp"


/**
 * Flux and derivatives class.
 */
class FluxDerivatives : public Fluxes {

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
    FluxDerivatives(int ld_law, double us[], int n_rs, double rs[],
                    int pnl_c, int pnl_e);

    /**
     * Compute normalised transit flux and derivatives.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param z distance from sky-plane, if z < 0 no transits [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @param out_f normalised light curve flux [].
     * @param out_df_dz derivatives array df/dz z={d, nu, {us}, {rs}}.
     * @return void.
     */
    void transit_flux_and_derivatives(
        const double d, const double z, const double nu,
        double& out_f, double out_df_dz[]);

  private:

    // Derivative variables.
    double m_df_dalpha, m_dI0_du1, m_dI0_du2, m_dI0_du3, m_dI0_du4,
           m_dalpha_ds0, m_dalpha_ds12, m_dalpha_ds1, m_dalpha_ds32,
           m_dalpha_ds2, m_ds0_dd, m_ds12_dd, m_ds1_dd, m_ds32_dd,
           m_ds2_dd, m_ds0_dnu, m_ds12_dnu, m_ds1_dnu, m_ds32_dnu,
           m_ds2_dnu;
    std::complex<double> m_dc0_da0, m_dcplus_dan, m_dcminus_dan,
                         m_dcplus_dbn, m_dcminus_dbn;
    std::vector<double> m_dthetas_dd, m_dthetas_dnu;
    std::vector<std::vector<std::complex<double>>> m_dthetas_dcs;
    Eigen::Vector<std::complex<double>, EigD> m_zeroes_c_conv_c;
    Eigen::Vector<Eigen::Vector<std::complex<double>, EigD>, EigD> m_els;
    Eigen::Vector<Eigen::Matrix<std::complex<double>, EigD, EigD>, EigD> m_dC_dcs;
    Eigen::Vector<std::complex<double>, EigD> m_df_dcs, m_ds0_dcs, m_ds12_dcs,
                                              m_ds1_dcs, m_ds32_dcs, m_ds2_dcs;
    Eigen::Matrix<std::complex<double>, EigD, EigD> m_dC_dd, m_dC_dnu;

    /**
     * Find and characterise the planet-stellar limb intersections
     * vector, theta, and sort in ascending order, -pi < theta <= pi.
     * Each adjacent pair of thetas corresponds to a segment of the
     * closed loop piecewise integral around the overlap region.
     * These pairs are assigned as either planet=0, entire_planet=1,
     * star=2, entire_star=3, or beyond=4 (flux=0).
     */
    void find_intersections_theta(const double d, const double nu) override final;

    /**
     * Compute the line integrals s_n along a segment of the
     * star's limb from theta_j to theta_j_p1 anticlockwise.
     */
    void s_star(int _j, int theta_type_j, double _theta_j,
                double _theta_j_p1, const double d,
                const double nu) override final;

    /**
     * Compute model derivatives: the flux with respect to the model
     * input parameters df/dz z={d, nu, {us}, {rs}}.
     */
    void f_derivatives(double df_dz[]);

    /**
     * Reset all derivative quantities to zero when the position is new.
     */
    void reset_derivatives();

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to d, dh_j_dd.
     */
    std::complex<double> dh_j_dd(int j);

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to nu, dh_j_dnu.
     */
    std::complex<double> dh_j_dnu(int j);

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to cn, dh_j_dcn.
     */
    std::complex<double> dh_j_dcn(int j, int _n);

    /**
     * Compute the real roots, as a vector of thetas, from a given companion
     * matrix. The real roots correspond to angles in the complex plane where
     * the matrix eigenvalues lie on the unit circle.
     */
    std::vector<double> compute_real_theta_roots(
      const Eigen::Matrix<std::complex<double>, EigD, EigD>&
        companion_matrix, int shape) override final;

    /**
     * Compute the even terms in the planet limb's line integral.
     * These terms are closed form and rely on a succession of
     * convolutions before the integral is evaluated.
     */
    void analytic_even_terms(
        int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
        const double d, const double nu) override final;

    /**
     * Compute the odd and half-integer terms in the planet limb's
     * line integral. These terms do not have an obvious closed form
     * solution and therefore Gauss-legendre quadrature is employed.
     */
    void numerical_odd_terms(
        int _j, int theta_type_j, double _theta_j, double _theta_j_p1,
        const double d, const double nu) override final;

};


#endif
