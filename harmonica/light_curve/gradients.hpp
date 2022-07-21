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
     * @param f empty normalised light curve flux [].
     * @param df_dz empty derivatives array df/dz z={d, nu, {us}, {rs}}.
     * @return void.
     */
    void transit_flux(const double &d, const double &z, const double &nu,
                      double &f, double df_dz[]);

  private:

    // Derivative variables.
    double df_dalpha;
    double dI0_du1, dI0_du2, dI0_du3, dI0_du4;
    double dalpha_ds0, dalpha_ds12, dalpha_ds1, dalpha_ds32, dalpha_ds2;
    double ds0_dd, ds12_dd, ds1_dd, ds32_dd, ds2_dd;
    double ds0_dnu, ds12_dnu, ds1_dnu, ds32_dnu, ds2_dnu;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic>
      df_dcs, ds0_dcs, ds12_dcs, ds1_dcs, ds32_dcs, ds2_dcs;
    std::complex<double> dc0_da0, dcplus_dan, dcminus_dan,
                         dcplus_dbn, dcminus_dbn;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
      dC_dd, dC_dnu;
    Eigen::Vector<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic>
      dC_dcs;
    std::vector<double> dthetas_dd, dthetas_dnu;
    std::vector<std::vector<std::complex<double>>> dthetas_dcs;

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
    virtual void find_intersections_theta(const double &d, const double &nu) final;

    /**
     * Compute the line integrals s_n along segments of the
     * star's limb from theta_j to theta_j_plus_1 anticlockwise.
     *
     * @param _j index of theta vecto
     * @param theta_type_j type of stellar line segment.
     * @param _theta_j start of line segment [radians].
     * @param _theta_j_plus_1 end of line segment [radians].
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return computed sTp_star line integral.
     */
    virtual void s_star(int _j, int theta_type_j, double &_theta_j,
                        double &_theta_j_plus_1, const double &d,
                        const double &nu) final;

    /**
     * Compute model derivatives: the flux with respect to the model
     * input parameters d, nu, {us}, and {rs}.
     *
     * @param df_dz empty derivatives array df/dz z={d, nu, {us}, {rs}}.
     * @return void.
     */
    void f_derivatives(double df_dz[]);

    /**
     * Reset all derivative quantities which are summed within the j-loop,
     * to zero. Run when the position is new.
     *
     * @return void.
     */
    void reset_derivatives();

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to d, dh_j_dd.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @return complex dh_j_dd coefficient.
     */
    std::complex<double> dh_j_dd(int j);

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to nu, dh_j_dnu.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @return complex dh_j_dnu coefficient.
     */
    std::complex<double> dh_j_dnu(int j);

    /**
     * Derivative of the complex polynomial coefficients for the
     * intersection equation with respect to cn, dh_j_dcn.
     *
     * @param j polynomial term exponent, 0 <= j <= 4N_c.
     * @param _n dcn term.
     * @return complex dh_j_dcn coefficient.
     */
    std::complex<double> dh_j_dcn(int j, int _n);

    /**
     * Compute the real roots, as a vector of thetas, from a given companion
     * matrix. The real roots correspond to angles in the complex plane where
     * the matrix eigenvalues lie on the unit circle.
     *
     * @param companion_matrix the complex-valued companion matrix.
     * @param shape number of rows=cols of matrix and complex eigenvalues.
     * @param require_eigenvectors derivatives switch.
     * @return vector of real roots in theta.
     */
    virtual std::vector<double> compute_real_theta_roots(
      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
        companion_matrix, int &shape) final;

    /**
     * Compute the even terms in the planet limb's line integral.
     * These terms are closed form and rely on a succession of
     * convolutions before the integral is evaluated.
     *
     * @param _j index of theta vector.
     * @param theta_type_j type of planet line segment.
     * @param _theta_j start of line segment [radians].
     * @param _theta_j_plus_1 end of line segment [radians].
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return void.
     */
    virtual void analytic_even_terms(
        int _j, int theta_type_j, double &_theta_j, double &_theta_j_plus_1,
        const double &d, const double &nu) final;

    /**
     * Compute the odd and half-integer terms in the planet limb's
     * line integral. These terms do not have an obvious closed form
     * solution and therefore Gauss-legendre quadrature is employed.
     *
     * @param _j index of theta vecto
     * @param theta_type_j type of planet line segment.
     * @param _theta_j start of line segment [radians].
     * @param _theta_j_plus_1 end of line segment [radians].
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @return void.
     */
    virtual void numerical_odd_terms(
        int _j, int theta_type_j, double &_theta_j, double &_theta_j_plus_1,
        const double &d, const double &nu) final;

};


#endif
