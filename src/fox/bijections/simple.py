from ..core import Transform
import flax
from flax.linen import compact
from typing import Sequence
from jax import numpy as jnp
from jax.nn.initializers import orthogonal
from jax.nn import sigmoid

class RealNVP(Transform):
    net: flax.linen.Module
    flip: bool
        
    def shift_and_log_scale_fn(self, u1: jnp.array) -> list:
        s = self.net(u1)
        return jnp.split(s, 2, axis=-1)
    
    def forward(self, u):
        mid = u.shape[-1] // 2
        u1, u2 = (u[:, :mid], u[:, mid:]) if u.ndim == 2 else (u[:mid], u[mid:])
        if self.flip:
            u2, u1 = u1, u2
        shift, log_scale = self.shift_and_log_scale_fn(u1)
        v2 = u2 * jnp.exp(log_scale) + shift
        if self.flip:
            u1, v2 = v2, u1
        v = jnp.concatenate([u1, v2], axis=-1)
        return v, log_scale.sum(-1) # 0 is incorrect, but it is not used for log-density estimation anyway
    
    def backward(self, v) -> tuple:
        mid = v.shape[-1] // 2
        v1, v2 = (v[:, :mid], v[:, mid:]) if v.ndim == 2 else (v[:mid], v[mid:])

        if self.flip:
            v1, v2 = v2, v1
        shift, log_scale = self.shift_and_log_scale_fn(v1)
        u2 = (v2 - shift) * jnp.exp(-log_scale)
        if self.flip:
            v1, u2 = u2, v1
        u = jnp.concatenate([v1, u2], axis=-1)
        return u, -log_scale.sum(-1)




class InvertibleMM(Transform):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    @compact
    def forward(self, z):
        d = z.shape[-1]
        W = self.param('W', orthogonal(), (d,d))
        return z@W, 0, #jnp.linalg.slogdet(W)[1]
    
    def backward(self, x):
        W = self.get_variable("params", "W")
        inv_W = jnp.linalg.inv(W)
        return x @ inv_W, jnp.linalg.slogdet(inv_W)[1]




class Sigmoid(Transform):

    def forward(self, x):
        return sigmoid(x), jnp.log(sigmoid(x) * (1 - sigmoid(x))).sum(-1)
           
    def backward(self, x):
        return jnp.log(x /(1 - x)), -jnp.log(x - x**2).sum(-1)

class Logit(Transform):
    pass #TODO

