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

    // Limb darkening parameters.
    double _ld_law;
    Eigen::Vector<double, Eigen::Dynamic> p;
    double I_0;

    // Transmission string parameters.
    int N_c;
    Eigen::Vector<double, Eigen::Dynamic> c;

    bool _require_gradients;

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
     * Name and description.
     *
     * @param d planet-star centre separation [stellar radii].
     * @param nu planet velocity-star centre angle [radians].
     * @param f empty normalised light curve flux [].
     * @param dd_dz array of derivatives dd/dz z={t0, p, a, i, e, w}.
     * @param dnu_dz array of derivatives dnu/dz z={t0, p, a, i, e, w}.
     * @param df_dz empty array of derivatives df/dz z={t0, p, a, i, e, w, us, rs}.
     * @return void.
     */
    void transit_light_curve(const double &d, const double &nu, double &f,
                             const double* dd_dz[], const double* dnu_dz[],
                             double* df_dz[]);


};


#endif
