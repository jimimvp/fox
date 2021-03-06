from _pytest.fixtures import fixture
import pytest
from fox.nn import MLP
from fox.core import NormalizingFlowDist, NormalizingFlow
from fox.bijections.simple import RealNVP
from fox.distributions import StandardGaussian
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

rng = jax.random.PRNGKey(0)


@pytest.fixture
def rnvp():
    return RealNVP(MLP([512,512], 2), False)

@pytest.fixture
def rnvp_stacked():
    return  NormalizingFlow([
        RealNVP(MLP([512,512], 2), False),
        RealNVP(MLP([512,512], 2), True),
        RealNVP(MLP([512,512], 2), False)
        ])

@pytest.fixture
def x():
    return jnp.array([
        [1.0, 2.0],
        [2.0, 1.0]
    ])


def test_bijection_rnvp(x, rnvp:RealNVP):
    params = rnvp.init(rng, x, method=rnvp.forward)
    z, ldj = rnvp.apply(params, x,  method=rnvp.forward)
    x_, _ = rnvp.apply(params, z, method=rnvp.backward)
    assert jnp.isclose(x_-x, 0, atol=1e-5).all()


def test_bijection_stacked(x, rnvp_stacked: NormalizingFlow):
    params = rnvp_stacked.init(rng, x, method=rnvp_stacked.forward)
    z, ldj = rnvp_stacked.apply(params, x,  method=rnvp_stacked.forward)
    x_, _ = rnvp_stacked.apply(params, z, method=rnvp_stacked.backward)
    assert jnp.isclose(x_-x, 0, atol=1e-5).all()


#### invertible MM test

from fox.bijections.simple import InvertibleMM

@fixture
def imm():
    return InvertibleMM()


def test_imm_bijection(x, imm: InvertibleMM):
    rng = jax.random.PRNGKey(0)
    params = imm.init(rng, x, method=imm.forward)
    z, ldj = imm.apply(params, x,  method=imm.forward)
    x_, _ = imm.apply(params, z, method=imm.backward)
    assert jnp.isclose(x_-x, 0, atol=1e-5).all()



##### sigmoid layer test

from fox.bijections.simple import Sigmoid

@fixture
def sigmoid():
    return Sigmoid()


def test_sigmoid_bijection(x, sigmoid):
    rng = jax.random.PRNGKey(0)
    layer = sigmoid
    params = layer.init(rng, x, method=layer.forward)
    z, ldj = layer.apply({}, x,  method=layer.forward)
    x_, _ = layer.apply({}, z, method=layer.backward)
    assert jnp.isclose(x_-x, 0, atol=1e-5).all()

