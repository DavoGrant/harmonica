#ifndef FLUXES_HPP
#define FLUXES_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


/**
 * Fluxes class.
 */
class Fluxes {

  private:

    // Some parameters.
    double _d;
    double _nu;

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
     * @param WIP.
     * @param ds array of planet-star centre separations [stellar radii].
     * @param nus array of planet velocity-star centre angles [radians].
     * @param fs empty array of normalised light curve fluxes [].
     * @param ds_grad (empty) array of derivatives dd/dx x={t0, p, a, i, e, w}.
     * @param nus_grad (empty) array of derivatives dnu/dx x={t0, p, a, i, e, w}.
     * @param fs_grad empty array of derivatives dfs/dx x={t0, p, a, i, e, w, us, rs}.
     * @return void.
     */
    void compute_something();


};


#endif
