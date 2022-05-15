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

    // Limb-darkening parameters.
    double _ld_law;
    double I_0;
    Eigen::Vector<double, Eigen::Dynamic> p;

    // Transmission-string parameters.
    int N_c;
    Eigen::Vector<std::complex<double>, Eigen::Dynamic> c;
    double min_rp;
    double max_rp;

    // Intersection variables.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> D;
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> C;

    // Derivatives switch.
    bool _require_gradients;

    /**
     * Companion matrix elements for computing the max and min radii of
     * a transmission string.
     *
     * @param j row index, starts at one.
     * @param k column index, starts at one.
     * @param shape number of rows=cols of matrix.
     * @return complex matrix element.
     */
    std::complex<double> extrema_companion_matrix_D_jk(int j, int k,
                                                       const int shape);

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
        companion_matrix, const int shape);

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
    double rp_theta(double theta);

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
