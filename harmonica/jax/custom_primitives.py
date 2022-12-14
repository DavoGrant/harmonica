import numpy as np
import jax.numpy as jnp
from jax.config import config
from jaxlib import xla_client
from functools import partial
from jax.interpreters import ad
from jax import abstract_arrays, core, xla

from harmonica import bindings

# Enable double floating precision.
config.update("jax_enable_x64", True)


def harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                              u1=0., u2=0., r=jnp.array([0.1])):
    """ Harmonica transits with jax -- quadratic limb darkening.

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
    u1, u2 : floats
        Quadratic limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].

    """
    # Unpack model parameters.
    params = [t0, period, a, inc, ecc, omega, u1, u2]
    for rn in r:
        params.append(rn)

    # Broadcast parameters to same length as times.
    times, *args_struc = jnp.broadcast_arrays(times, *params)

    return jax_light_curve_quad_ld_prim(times, *args_struc)[0]


def jax_light_curve_quad_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_quad_ld_p.bind(times, *params)


def jax_light_curve_quad_ld_abstract_eval(abstract_times, *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = abstract_arrays.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = abstract_arrays.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_quad_ld_xla_translation(c, timesc, *paramssc):
    """ XLA compilation rules. """
    # Get `shape` info.
    timesc_shape = c.get_shape(timesc)

    # Define input `shapes`.
    data_type = timesc_shape.element_type()
    shape = timesc_shape.dimensions()
    dims_order = tuple(range(len(shape) - 1, -1, -1))
    input_shape = xla_client.Shape.array_shape(data_type, shape, dims_order)
    rs_input_shapes = tuple(input_shape for ic in paramssc)

    # Additionally, define the number of model evaluation points.
    n_times = np.prod(shape).astype(np.int64)
    n_times_input = xla_client.ops.ConstantLiteral(c, n_times)
    n_times_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Additionally, define the number of transmission string coefficients.
    n_rs = len(paramssc) - 6 - 2
    n_rs_input = xla_client.ops.ConstantLiteral(c, n_rs)
    n_rs_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Define output `shapes`.
    output_shape_model_eval = input_shape
    shape_derivatives = shape + (6 + 2 + n_rs,)
    dims_order_derivatives = tuple(range(len(shape), -1, -1))
    output_shape_model_derivatives = xla_client.Shape.array_shape(
        data_type, shape_derivatives, dims_order_derivatives)

    return xla_client.ops.CustomCallWithLayout(
        c,
        b"jax_light_curve_quad_ld",
        operands=(n_times_input, n_rs_input, timesc, *paramssc),
        operand_shapes_with_layout=
            (n_times_shape, n_rs_shape, input_shape)
            + rs_input_shapes,
        shape_with_layout=xla_client.Shape.tuple_shape(
            (output_shape_model_eval,
             output_shape_model_derivatives)))


def jax_light_curve_quad_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    dtimes, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_quad_ld_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs):
        if type(pd) is ad.Zero:
            # This partial derivative is not required. It has been
            # set to a deterministic value.
            continue
        df += pd * df_dz[..., idx_pd]

    # None is returned here for the second output as we are not interested
    # in using it for gradient-based inference.
    return (f, df_dz), (df, None)


def harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0.,
                                   u1=0., u2=0., u3=0., u4=0.,
                                   r=jnp.array([0.1])):
    """ Harmonica transits with jax -- non-linear limb darkening.

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
    u1, u2, u3, u4 : floats
        Non-linear limb-darkening coefficients.
    r : ndarray
        Transmission string coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. The length of r must be odd,
        and the final two coefficients must not both be zero.

        .. math::

            r_{\\rm{p}}(\\theta) = \\sum_{n=0}^N a_n \\cos{(n \\theta)}
            + \\sum_{n=1}^N b_n \\sin{(n \\theta)}

        The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].

    Returns
    -------
    flux : array
        Normalised transit light curve fluxes [].

    """
    # Unpack model parameters.
    params = [t0, period, a, inc, ecc, omega, u1, u2, u3, u4]
    for rn in r:
        params.append(rn)

    # Broadcast parameters to same length as times.
    times, *args_struc = jnp.broadcast_arrays(times, *params)

    return jax_light_curve_nonlinear_ld_prim(times, *args_struc)[0]


def jax_light_curve_nonlinear_ld_prim(times, *params):
    """ Define new JAX primitive. """
    return jax_light_curve_nonlinear_ld_p.bind(times, *params)


def jax_light_curve_nonlinear_ld_abstract_eval(abstract_times, *abstract_params):
    """ Define the abstract evaluation. """
    # Define first model output.
    abstract_model_eval = abstract_arrays.ShapedArray(
        abstract_times.shape, abstract_times.dtype)

    # Define second model output.
    n_params = len(abstract_params)
    abstract_model_derivatives = abstract_arrays.ShapedArray(
        tuple(abstract_times.shape) + (n_params,), abstract_times.dtype)

    return abstract_model_eval, abstract_model_derivatives


def jax_light_curve_nonlinear_xla_translation(c, timesc, *paramssc):
    """ XLA compilation rules. """
    # Get `shape` info.
    timesc_shape = c.get_shape(timesc)

    # Define input `shapes`.
    data_type = timesc_shape.element_type()
    shape = timesc_shape.dimensions()
    dims_order = tuple(range(len(shape) - 1, -1, -1))
    input_shape = xla_client.Shape.array_shape(data_type, shape, dims_order)
    rs_input_shapes = tuple(input_shape for ic in paramssc)

    # Additionally, define the number of model evaluation points.
    n_times = np.prod(shape).astype(np.int64)
    n_times_input = xla_client.ops.ConstantLiteral(c, n_times)
    n_times_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Additionally, define the number of transmission string coefficients.
    n_rs = len(paramssc) - 6 - 4
    n_rs_input = xla_client.ops.ConstantLiteral(c, n_rs)
    n_rs_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

    # Define output `shapes`.
    output_shape_model_eval = input_shape
    shape_derivatives = shape + (6 + 4 + n_rs,)
    dims_order_derivatives = tuple(range(len(shape), -1, -1))
    output_shape_model_derivatives = xla_client.Shape.array_shape(
        data_type, shape_derivatives, dims_order_derivatives)

    return xla_client.ops.CustomCallWithLayout(
        c,
        b"jax_light_curve_nonlinear_ld",
        operands=(n_times_input, n_rs_input, timesc, *paramssc),
        operand_shapes_with_layout=
            (n_times_shape, n_rs_shape, input_shape)
            + rs_input_shapes,
        shape_with_layout=xla_client.Shape.tuple_shape(
            (output_shape_model_eval,
             output_shape_model_derivatives)))


def jax_light_curve_nonlinear_ld_value_and_jvp(arg_values, arg_tangents):
    """ Evaluate the primal output and the tangents. """
    # Unpack parameter values and tangents.
    times, *args = arg_values
    dtimes, *dargs = arg_tangents

    # Run the model to get the value and derivatives as designed.
    f, df_dz = jax_light_curve_nonlinear_ld_prim(times, *args)

    # Compute grad.
    df = 0.
    for idx_pd, pd in enumerate(dargs):
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
    b'jax_light_curve_quad_ld', bindings.jax_registrations()['jax_light_curve_quad_ld'])
xla_client.register_custom_call_target(
    b'jax_light_curve_nonlinear_ld', bindings.jax_registrations()['jax_light_curve_nonlinear_ld'])

# Create a primitive for quad ld.
jax_light_curve_quad_ld_p = core.Primitive('jax_light_curve_quad_ld')
jax_light_curve_quad_ld_p.multiple_results = True
jax_light_curve_quad_ld_p.def_impl(partial(xla.apply_primitive, jax_light_curve_quad_ld_p))
jax_light_curve_quad_ld_p.def_abstract_eval(jax_light_curve_quad_ld_abstract_eval)
xla.backend_specific_translations['cpu'][jax_light_curve_quad_ld_p] = \
    jax_light_curve_quad_ld_xla_translation
ad.primitive_jvps[jax_light_curve_quad_ld_p] = jax_light_curve_quad_ld_value_and_jvp

# Create a primitive for non-linear ld.
jax_light_curve_nonlinear_ld_p = core.Primitive('jax_light_curve_nonlinear_ld')
jax_light_curve_nonlinear_ld_p.multiple_results = True
jax_light_curve_nonlinear_ld_p.def_impl(partial(xla.apply_primitive, jax_light_curve_nonlinear_ld_p))
jax_light_curve_nonlinear_ld_p.def_abstract_eval(jax_light_curve_nonlinear_ld_abstract_eval)
xla.backend_specific_translations['cpu'][jax_light_curve_nonlinear_ld_p] = \
    jax_light_curve_nonlinear_xla_translation
ad.primitive_jvps[jax_light_curve_nonlinear_ld_p] = jax_light_curve_nonlinear_ld_value_and_jvp
