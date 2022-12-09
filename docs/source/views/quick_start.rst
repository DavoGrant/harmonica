Quick start
===========

After installing the code (see :doc:`installation <installation>`) you
are ready to generate light curves. Below we demonstrate a minimal example.

First, import the HarmonicaTransit class and specify the times at which
you want to evaluate the light curve model.

.. code-block:: python

    import numpy as np
    from harmonica import HarmonicaTransit


    ht = HarmonicaTransit(times=np.linspace(-0.2, 0.2, 500))

Next, set the orbit, limb-darkening, and transmission string parameters.

.. code-block:: python

    ht.set_orbit(t0=0., period=4., a=7., inc=88. * np.pi / 180.)
    ht.set_stellar_limb_darkening(u=np.array([0.074, 0.193]), limb_dark_law='quadratic')
    ht.set_planet_transmission_string(r=np.array([0.1, -0.003, 0.]))

Finally, generate the transit light curve.

.. code-block:: python

    light_curve = ht.get_transit_light_curve()

