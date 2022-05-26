#ifndef TRAJECTORIES_HPP
#define TRAJECTORIES_HPP


/**
 * Orbital trajectories class.
 */
class OrbitTrajectories {

  private:

    // Orbital parameters.
    double _t0;
    double _period;
    double _n;
    double _a;
    double _inc;
    double _sin_inc;
    double _cos_inc;
    double _ecc;
    double _omega;
    double _sin_omega;
    double _cos_omega;
    bool _require_gradients;

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
    OrbitTrajectories(double t0, double period, double a,
                      double inc, double ecc, double omega,
                      bool require_gradients);

    /**
     * Compute circular orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     * Optionally, the partial derivatives dd/dz and dnu/dz, for z in the
     * set {t0, p, a, i} may be computed.
     *
     * @param time model evaluation time [days].
     * @param d empty planet-star centre separation [stellar radii].
     * @param nu empty planet velocity-star centre angle [radians].
     * @param dd_dz empty array of derivatives dd/dz z={t0, p, a, i}.
     * @param dnu_dz empty array of derivatives dnu/dz z={t0, p, a, i}.
     * @return void.
     */
    void compute_circular_orbit(const double &time, double &d, double &nu,
                                double* dd_dz[], double* dnu_dz[]);

    /**
     * Compute eccentric orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     * Optionally, the partial derivatives dd/dz and dnu/dz, for z in the
     * set {t0, p, a, i, e, w} may be computed.
     *
     * @param time model evaluation time [days].
     * @param d empty planet-star centre separation [stellar radii].
     * @param nu empty planet velocity-star centre angle [radians].
     * @param dd_dz empty array of derivatives dd/dz z={t0, p, a, i, e, w}.
     * @param dnu_dz empty array of derivatives dnu/dz z={t0, p, a, i, e, w}.
     * @return void.
     */
    void compute_eccentric_orbit(const double &time, double &d, double &nu,
                                 double* dd_dz[], double* dnu_dz[]);

};


#endif
