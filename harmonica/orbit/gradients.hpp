#ifndef ORBITDERIVATIVES_HPP
#define ORBITDERIVATIVES_HPP

#include "trajectories.hpp"


/**
 * Orbital trajectories and derivatives class.
 */
class OrbitDerivatives : public OrbitTrajectories {

  public:

    /**
     * Constructor.
     *
     * @param t0 time of transit centre [days].
     * @param period orbital period [days].
     * @param a semi-major axis [stellar radii].
     * @param inc orbital inclination [radians].
     * @param ecc eccentricity [].
     * @param omega argument of periastron [radians].
     * @param require_gradients derivatives switch.
     */
    OrbitDerivatives(double t0, double period, double a,
                     double inc, double ecc, double omega);

    /**
     * Compute circular orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     * Optionally, the partial derivatives dd/dz and dnu/dz, for z in the
     * set {t0, p, a, i} may be computed.
     *
     * @param time model evaluation time [days].
     * @param out_d planet-star centre separation [stellar radii].
     * @param out_z distance from sky-plane, z < 0 planet behind [stellar radii].
     * @param out_nu planet velocity-star centre angle [radians].
     * @param out_dd_dz derivatives array dd/dz z={t0, p, a, i}.
     * @param out_dnu_dz derivatives array dnu/dz z={t0, p, a, i}.
     * @return void.
     */
    void compute_circular_orbit_and_derivatives(
        const double time, double& out_d, double& out_z,
        double& out_nu, double out_dd_dz[], double out_dnu_dz[]);

    /**
     * Compute eccentric orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     * Optionally, the partial derivatives dd/dz and dnu/dz, for z in the
     * set {t0, p, a, i, e, w} may be computed.
     *
     * @param time model evaluation time [days].
     * @param out_d planet-star centre separation [stellar radii].
     * @param out_z distance from sky-plane, z < 0 planet behind [stellar radii].
     * @param out_nu planet velocity-star centre angle [radians].
     * @param out_dd_dz derivatives array dd/dz z={t0, p, a, i, e, w}.
     * @param out_dnu_dz derivatives array dnu/dz z={t0, p, a, i, e, w}.
     * @return void.
     */
    void compute_eccentric_orbit_and_derivatives(
        const double time, double& out_d, double& out_z,
        double& out_nu, double out_dd_dz[], double out_dnu_dz[]);

};


#endif
