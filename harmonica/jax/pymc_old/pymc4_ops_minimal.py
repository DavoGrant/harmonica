import aesara
import numpy as np
import arviz as az
import aesara.tensor as at
import pymc as pm
from aesara.graph import basic, op
import corner
import matplotlib.pyplot as plt

from harmonica import bindings


class HarmonicaLC(op.Op):

    __props__ = ()

    def make_node(self, *args):
        in_args = [at.as_tensor_variable(a, dtype='float64') for a in args]
        out_args = [in_args[0].type() for a in args]
        return basic.Apply(self, in_args, out_args)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0] for i in input_shapes]

    def perform(self, node, inputs, output_storage):
        # print('perform')
        times, t0, period, a, inc, ecc, omega = inputs[:7]
        us = np.array(inputs[7:9])
        rs = np.array(inputs[9:])

        n_lcd = times.shape + (6 + 2 + 3,)
        fs = np.empty(times.shape, dtype='float64')
        fs_grad = np.zeros(n_lcd, dtype='float64')

        bindings.test_pymc(t0, period, a, inc, ecc, omega,
                           0, us, rs, times, fs, fs_grad,
                           20, 50)

        output_storage[0][0] = fs
        for i in range(0, 11):
            output_storage[i + 1][0] = fs_grad[:, i]

    def grad(self, inputs, output_grads):
        # print('ello gradient')
        lc, *lc_grads = self(*inputs)

        for g in output_grads[1:]:
            if not isinstance(g.type, aesara.gradient.DisconnectedType):
                raise ValueError('Gradients only supported for flux.')

        if isinstance(output_grads[0].type, aesara.gradient.DisconnectedType):
            return [aesara.gradient.DisconnectedType()() for i in range(12)]

        # print(at.sum(output_grads[0] * 5, axis=-1))
        # print(at.sum(output_grads[0] * lc_grads[0], axis=-1))
        a = [at.sum(output_grads[0] * lc_grads[i], axis=-1) for i in range(11)]
        a.insert(0, aesara.gradient.NullType()())

        return a

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

# def quad_solution_vector(b, r):
#     return _quad_solution_vector(b, r)[0]

# _times = at.vector()
# _t0 = at.scalar()
# _period = at.scalar()
# _a = at.scalar()
# _inc = at.scalar()
# _ecc = at.scalar()
# _omega = at.scalar()
# _u1 = at.scalar()
# _u2 = at.scalar()
# _r0 = at.scalar()
# _r1 = at.scalar()
# _r2 = at.scalar()
#
# _ld_law = 'quadratic'
# hlc = HarmonicaLC(_ld_law)
#
# f = aesara.function([_times, _t0, _period, _a, _inc, _ecc, _omega, _u1, _u2, _r0, _r1, _r2],
#                     hlc(_times, _t0, _period, _a, _inc, _ecc, _omega, _u1, _u2, _r0, _r1, _r2))
#
# __times = np.linspace(4.5, 5.5, 100)
# __t0 = 5.
# __period = 10.
# __a = 7.
# __inc = 89. * np.pi / 180.
# __ecc = 0.1
# __omega = 0.1
# __u1 = 0.1
# __u2 = 0.2
# __r0 = 0.1
# __r1 = 0.001
# __r2 = 0.001
#
# out = f(__times, __t0, __period, __a, __inc, __ecc, __omega, __u1, __u2, __r0, __r1, __r2)
# print(out)

_ld_law = 'quadratic'
_times = np.linspace(4.5, 5.5, 100)
obs = np.array([1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 0.99889131, 0.99666368, 0.99407374, 0.99166524,
       0.99054003, 0.99026628, 0.99006953, 0.98991855, 0.98979899,
       0.98970255, 0.98962384, 0.9895591 , 0.98950562, 0.98946132,
       0.98942464, 0.98939431, 0.98936935, 0.98934897, 0.98933253,
       0.98931954, 0.9893096 , 0.98930241, 0.98929775, 0.98929549,
       0.98929555, 0.98929794, 0.98930273, 0.98931006, 0.98932016,
       0.98933332, 0.98934995, 0.98937056, 0.98939579, 0.98942643,
       0.9894635 , 0.98950825, 0.98956229, 0.98962771, 0.98970729,
       0.98980485, 0.98992591, 0.990079  , 0.990279  , 0.99055902,
       0.99178381, 0.9942221 , 0.99680665, 0.99899825, 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ])


with pm.Model() as model:
    # pm_r0 = pm.Normal('r0', mu=0.1, sigma=0.01)
    pm_r0 = pm.Uniform('r0', lower=0.05, upper=0.15)
    pm_r1 = pm.Normal('r1', mu=0.001, sigma=0.0001)
    pm_r2 = pm.Normal('r2', mu=0.001, sigma=0.0001)
    # pm_r0 = 0.1
    # pm_r1 = 0.001
    # pm_r2 = 0.001

    # pm_u1 = pm.Normal('u1', mu=0.1, sigma=0.01)
    # pm_u2 = pm.Normal('u2', mu=0.2, sigma=0.01)
    pm_u1 = 0.1
    pm_u2 = 0.2

    hlc = HarmonicaLC()
    flux = pm.Deterministic('flux',
                            hlc(_times, 5., 10., 7., 89. * np.pi / 180., 0.1, 0.1,
                                pm_u1, pm_u2, pm_r0, pm_r1, pm_r2)[0])

    pm.Normal('y', mu=flux, sigma=10.e-6, observed=obs)

    trace = pm.sample(
        tune=100, draws=1000,
        chains=1, cores=1,
        initvals={'r0': 0.1, 'r1': 0.001, 'r2': 0.001},
        # initvals={'u1': 0.1, 'u2': 0.2},
        progressbar=False)

    # print(az.summary(trace, var_names=['u1', 'u2']).to_string())



    # fig = corner.corner(trace, truths=[0.1, 0.001, 0.001], labels=['r0', 'r1', 'r2'])
    # plt.show()
