#ifndef FLUXES_HPP
#define FLUXES_HPP


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
     * @param name [units].
     */
    Fluxes(bool require_gradients);

    /**
     * Name and description.
     *
     * @param require_gradients derivatives switch.
     * @return void.
     */
    void compute_something(bool require_gradients);


};


#endif
