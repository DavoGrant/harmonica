import numpy as np
import theano
import theano.tensor as tt
from theano.graph import basic, op

from harmonica import bindings


class HarmonicaLC(op.Op):

    __props__ = ('ld_mode', )

    def __init__(self, limb_dark_law):
        if limb_dark_law == 'quadratic':
            self.ld_mode = 0
        elif limb_dark_law == 'non-linear':
            self.ld_mode = 1
        super().__init__()

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a).astype('float64') for a in args]
        out_args = [in_args[0].type()]
        return basic.Apply(self, in_args, out_args)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def perform(self, node, inputs, output_storage):
        print('perform')
        times, t0, period, a, inc, ecc, omega = inputs[:7]
        print(t0)
        us = np.array(inputs[7:9])
        rs = np.array(inputs[9:])

        ds = np.empty(times.shape, dtype='float64')
        nus = np.empty(times.shape, dtype='float64')

        n_od = times.shape + (6,)
        n_lcd = times.shape + (6 + 3 + 5,)
        ds_grad = np.zeros(n_od, dtype='float64')
        nus_grad = np.zeros(n_od, dtype='float64')

        lc = np.empty(times.shape, dtype='float64')
        lc_grad = np.zeros(n_lcd, dtype='float64')

        bindings.orbit(t0, period, a, inc, ecc, omega,
                       times, ds, nus,
                       ds_grad, nus_grad,
                       require_gradients=True)
        bindings.light_curve(self.ld_mode, us, rs, ds, nus, lc,
                             ds_grad, nus_grad, lc_grad, 50, 500,
                             require_gradients=True)

        output_storage[0][0] = lc

    def grad(self, inputs, output_grads):
        print('ello gradient')
        times, t0, period, a, inc, ecc, omega = inputs[:7]
        print(t0)
        lc = self(*inputs)
        print(lc)
        print(88)
        us = np.array(inputs[7:9])
        rs = np.array(inputs[9:])

        ds = np.empty((3,), dtype='float64')
        nus = np.empty((3,), dtype='float64')

        n_od = (3,) + (6,)
        n_lcd = (3,) + (6 + 3 + 5,)
        ds_grad = np.zeros(n_od, dtype='float64')
        nus_grad = np.zeros(n_od, dtype='float64')

        lc = np.empty((3,), dtype='float64')
        lc_grad = np.zeros(n_lcd, dtype='float64')

        bindings.orbit(t0, period, a, inc, ecc, omega,
                       times, ds, nus,
                       ds_grad, nus_grad,
                       require_gradients=True)
        bindings.light_curve(self.ld_mode, us, rs, ds, nus, lc,
                             ds_grad, nus_grad, lc_grad, 50, 500,
                             require_gradients=True)

        for g in output_grads[1:]:
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError('Gradients only supported for flux.')

        if isinstance(output_grads[0].type, theano.gradient.DisconnectedType):
            return [theano.gradient.DisconnectedType()() for i in range(12)]

        # print(at.sum(output_grads[0] * 5, axis=-1))
        # print(at.sum(output_grads[0] * lc_grads[0], axis=-1))
        a = [tt.sum(output_grads[0] * lc_grad[:, i], axis=-1) for i in range(11)]
        a.insert(0, theano.gradient.NullType()())

        return a

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

# def quad_solution_vector(b, r):
#     return _quad_solution_vector(b, r)[0]

_times = tt.vector()
_t0 = tt.scalar()
_period = tt.scalar()
_a = tt.scalar()
_inc = tt.scalar()
_ecc = tt.scalar()
_omega = tt.scalar()
_u1 = tt.scalar()
_u2 = tt.scalar()
_r0 = tt.scalar()
_r1 = tt.scalar()
_r2 = tt.scalar()

_ld_law = 'quadratic'
hlc = HarmonicaLC(_ld_law)

# f = theano.function([_times, _t0, _period, _a, _inc, _ecc, _omega, _u1, _u2, _r0, _r1, _r2],
#                     hlc(_times, _t0, _period, _a, _inc, _ecc, _omega, _u1, _u2, _r0, _r1, _r2))

__times = np.linspace(4.5, 5.5, 3)
__t0 = 5.
__period = 10.
__a = 7.
__inc = 89. * np.pi / 180.
__ecc = 0.1
__omega = 0.1
__u1 = 0.1
__u2 = 0.2
__r0 = 0.1
__r1 = 0.001
__r2 = 0.001

# out = f(__times, __t0, __period, __a, __inc, __ecc, __omega, __u1, __u2, __r0, __r1, __r2)
# print(out)

from tests import unittest_tools as utt

utt.verify_grad(
    hlc, [__times, __t0, __period, __a, __inc, __ecc, __omega, __u1, __u2, __r0, __r1, __r2]
)

# _ld_law = 'quadratic'
# _times = np.linspace(4.5, 5.5, 100)
# obs = np.array([1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 0.99889131, 0.99666368, 0.99407374, 0.99166524,
#        0.99054003, 0.99026628, 0.99006953, 0.98991855, 0.98979899,
#        0.98970255, 0.98962384, 0.9895591 , 0.98950562, 0.98946132,
#        0.98942464, 0.98939431, 0.98936935, 0.98934897, 0.98933253,
#        0.98931954, 0.9893096 , 0.98930241, 0.98929775, 0.98929549,
#        0.98929555, 0.98929794, 0.98930273, 0.98931006, 0.98932016,
#        0.98933332, 0.98934995, 0.98937056, 0.98939579, 0.98942643,
#        0.9894635 , 0.98950825, 0.98956229, 0.98962771, 0.98970729,
#        0.98980485, 0.98992591, 0.990079  , 0.990279  , 0.99055902,
#        0.99178381, 0.9942221 , 0.99680665, 0.99899825, 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ,
#        1.        , 1.        , 1.        , 1.        , 1.        ])
#
#
# with pm.Model() as model:
#     pm_t0 = pm.Normal('t0', mu=5., sigma=0.1)
#     pm_period = pm.Normal('period', mu=10., sigma=0.1)
#     pm_a = pm.Normal('a', mu=89. * np.pi / 180., sigma=0.1)
#     pm_inc = pm.Normal('inc', mu=7., sigma=0.1)
#     pm_ecc = pm.Normal('ecc', mu=0.1, sigma=0.01)
#     pm_omega = pm.Normal('omega', mu=0.1, sigma=0.01)
#
#     pm_u1 = pm.Normal('u1', mu=0.1, sigma=0.01)
#     pm_u2 = pm.Normal('u2', mu=0.2, sigma=0.01)
#
#     pm_r0 = pm.Normal('r0', mu=0.1, sigma=0.01)
#     pm_r1 = pm.Normal('r1', mu=0.001, sigma=0.0001)
#     pm_r2 = pm.Normal('r2', mu=0.001, sigma=0.0001)
#
#     hlc = HarmonicaLC(_ld_law)
#     flux = pm.Deterministic('flux',
#                             hlc(_times, pm_t0, pm_period, pm_a, pm_inc, pm_ecc, pm_omega,
#                                 pm_u1, pm_u2, pm_r0, pm_r1, pm_r2)[0])
#
#     pm.Normal('y', mu=flux, sigma=10.e-6, observed=obs)
#
#     trace = pm.sample(tune=100, draws=100, chains=1, cores=1)
