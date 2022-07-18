import numpy as np
import jax.numpy as jnp
from jax.config import config
from jaxlib import xla_client
from functools import partial
from jax.interpreters import ad, batching
from jax import abstract_arrays, core, xla

from harmonica import bindings

# Enable double floating precision.
config.update("jax_enable_x64", True)


def harmonica_transit(times, t0, period, a, inc, ecc=0., omega=0.,
                      limb_dark_law='quadratic', *args):
    """ Harmonica transit light curve -- custom JAX.

    todo: update args format here, and docs correspondingly
    todo: check still runs
    todo: update api
    todo: update unit tests for regular flux forward modelling.
    todo: make unti tests for gradients and jax jit etc.

    Parameters
    ----------
    times : ndarray
        1D array of model evaluation times [days].
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    ecc : float, optional
        Eccentricity [], 0 <= ecc < 1. Default=0.
    omega : float, optional
        Argument of periastron [radians]. Default=0.
    u : ndarray
        1D array of limb-darkening coefficients which correspond to
        the limb-darkening law specified by limb_dark_law. The
        quadratic law requires two coefficients and the non-linear
        law requires four coefficients.
    limb_dark_law : string, optional; `quadratic` or `non-linear`
        The stellar limb darkening law. Default=`quadratic`.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane.

        ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
        + \sum_{n=1}^N b_n \csin{(n \theta)}``

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].

    """
    # Limb-darkening parameterization switch.
    if limb_dark_law == 'quadratic':
        ld_mode = 0.
    elif limb_dark_law == 'non-linear':
        ld_mode = 1.
    else:
        raise ValueError('limb_dark_law not recognised.')

    # Define args struc.
    args_struc = (ld_mode, t0, period, a, inc, ecc, omega) + args

    # Broadcast input params to same length as times.
    times, *args_struc = jnp.broadcast_arrays(times, *args_struc)

    return jax_light_curve_prim(times, *args_struc)[0]


def jax_light_curve_prim(times, *args):
    """ Define new JAX primitive. """
    return jax_light_curve_p.bind(times, *args)


def jax_light_curve_abstract_eval(abstract_times, *abstract_args):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = abstract_arrays.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_args) - 1
    abstract_model_derivatives = abstract_arrays.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_xla_translation(c, timesc, *argsc):
    """ XLA compilation rules. """
    # Get `shape` info.
    timesc_shape = c.get_shape(timesc)

    # Define input `shapes`.
    data_type = timesc_shape.element_type()
    shape = timesc_shape.dimensions()
    dims_order = tuple(range(len(shape) - 1, -1, -1))
    input_shape = xla_client.Shape.array_shape(data_type, shape, dims_order)
    rs_input_shapes = tuple(input_shape for ic in argsc)

    # Additionally, define the number of model evaluation points.
    n_times = np.prod(shape).astype(np.int64)
    n_times_input = xla_client.ops.ConstantLiteral(c, n_times)
    n_times_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Additionally, define the number of params.
    n_params = len(argsc) - 1
    n_params_input = xla_client.ops.ConstantLiteral(c, n_params)
    n_params_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Define output `shapes`.
    output_shape_model_eval = input_shape
    shape_derivatives = shape + (n_params,)
    dims_order_derivatives = tuple(range(len(shape), -1, -1))
    output_shape_model_derivatives = xla_client.Shape.array_shape(
        data_type, shape_derivatives, dims_order_derivatives)

    return xla_client.ops.CustomCallWithLayout(
        c,
        b"jax_light_curve",
        operands=(n_times_input, n_params_input, timesc, *argsc),
        operand_shapes_with_layout=
            (n_times_shape, n_params_shape, input_shape)
            + rs_input_shapes,
        shape_with_layout=xla_client.Shape.tuple_shape(
            (output_shape_model_eval,
             output_shape_model_derivatives)))


def jax_light_curve_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    dtimes, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs[1:]):
        if type(pd) is ad.Zero:
            # This partial derivative is not required. It has been
            # set to a deterministic value.
            continue
        df += pd * df_dz[..., idx_pd]

    # None is returned here for the second output as we are not interested
    # in using it for gradient-based inference.
    return (f, df_dz), (df, None)


# Register the C++ models, bytes string required.
xla_client.register_custom_call_target(
    b'jax_light_curve', bindings.jax_registrations()['jax_light_curve'])

# Create the primitives.
jax_light_curve_p = core.Primitive('jax_light_curve')
jax_light_curve_p.multiple_results = True
jax_light_curve_p.def_impl(partial(xla.apply_primitive, jax_light_curve_p))
jax_light_curve_p.def_abstract_eval(jax_light_curve_abstract_eval)
xla.backend_specific_translations['cpu'][jax_light_curve_p] = \
    jax_light_curve_xla_translation
ad.primitive_jvps[jax_light_curve_p] = jax_light_curve_value_and_jvp
