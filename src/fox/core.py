import jax.numpy as jnp
from jax import random
import flax
from typing import List, Sequence
from .distributions import Distribution

class Transform(flax.linen.Module):
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, u):
        """
            Returns inverse-transformed x and ldj
        """
    
    def backward(self, x):
        """
            Returns transformed x and ldj
        """

class NormalizingFlow(Transform):
    transforms: Sequence[Transform]
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        m, _ = x.shape
        log_det = jnp.zeros(m)
        zs = [x]
        for t in self.transforms:
            x, ld = t.forward(x)
            log_det += ld
            zs.append(x)
        return zs[-1], log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = jnp.zeros(m)
        xs = [z]
        for t in self.transforms[::-1]:
            z, ld = t.backward(z)
            log_det += ld
            xs.append(z)
        return xs[-1], log_det


class NormalizingFlowDist(flax.linen.Module):
    prior: Distribution
    flow: NormalizingFlow
    
    def sample(self, key, size):
        z = self.prior.sample(key, size)
        x, ldj = self.flow.forward(z)
        return x
    
    def log_prob(self, x):
        z, logdet = self.flow.backward(x)
        return self.prior.log_prob(z) + logdet
    