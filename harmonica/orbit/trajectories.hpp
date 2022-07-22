#ifndef ORBITTRAJECTORIES_HPP
#define ORBITTRAJECTORIES_HPP


/**
 * Orbital trajectories class.
 */
class OrbitTrajectories {

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
     */
    OrbitTrajectories(double t0, double period, double a,
                      double inc, double ecc, double omega);

    /**
     * Compute circular orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     *
     * @param time model evaluation time [days].
     * @param d empty planet-star centre separation [stellar radii].
     * @param z distance from sky-plane, z < 0 planet behind [stellar radii].
     * @param nu empty planet velocity-star centre angle [radians].
     * @return void.
     */
    void compute_circular_orbit(const double& time, double& d,
                                double& z, double& nu);

    /**
     * Compute eccentric orbit trajectories of a planet-star system.
     * The separation distance, d, is between the planet and stellar centres
     * and the angle, nu, is between the planet's velocity and the stellar
     * centre. Both quantities are computed in the plane of the sky.
     *
     * @param time model evaluation time [days].
     * @param d empty planet-star centre separation [stellar radii].
     * @param z distance from sky-plane, z < 0 planet behind [stellar radii].
     * @param nu empty planet velocity-star centre angle [radians].
     * @return void.
     */
    void compute_eccentric_orbit(const double& time, double& d,
                                 double& z, double& nu);

  protected:

    // Orbital parameters.
    double m_t0, m_period, m_n, m_a, m_inc, m_sin_inc, m_cos_inc,
           m_ecc, m_omega, m_sin_omega, m_cos_omega;

    // Intermediate vals.
    double m_x, m_y, m_d_squared, m_r, m_sin_M, m_cos_M, m_atan_mcsM,
           m_sin_fpw, m_cos_fpw, m_ope_cosf, m_omes, m_sin_f, m_cos_f,
           m_sope, m_some, m_E0, m_atan_mcs_fpw;

};


#endif
