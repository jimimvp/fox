"""
    Just standard training loop for simple testing
"""
from dataclasses import dataclass
from types import MethodType
from jax.interpreters.batching import batch
from matplotlib.pyplot import plot
from sklearn import datasets
from tqdm import tqdm
import optax
from jax import numpy as jnp
import numpy as np
import jax
from sklearn.preprocessing import StandardScaler
import functools

def interruptable(func):
    def wrapped(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except KeyboardInterrupt as e:
            print("Interrupted", func)
    return wrapped


def load_moons(N):
    X =  datasets.make_moons(n_samples=N, shuffle=True, noise=0.05)[0]
    X = StandardScaler().fit_transform(X)
    return X

def load_swiss_roll(N):
    X = datasets.make_swiss_roll(N)[0][:,:2]
    X = StandardScaler().fit_transform(X)
    return X

def nll_loss(params, X_batch, flow_dist):
    return - (flow_dist.apply(params,X_batch,  method=flow_dist.log_prob)).mean()

def batch_iter(rng, X, batch_size, steps):
    s = 0
    while True:
        rng, rng1 = jax.random.split(rng)
        X_shuff = jax.random.permutation(rng1, X)
        for i in range(0 ,len(X_shuff)-batch_size, batch_size):
            yield X_shuff[i:i+batch_size]
            s+=1
            if s == steps:
                return
@interruptable
def train(rng,  params: jnp.array, loss, X: np.array, lr: float = 3e-4, steps: int = 100 , batch_size: int = 100) -> list:
    
    optimizer = optax.chain(optax.adam(lr), optax.clip_by_global_norm(5.0))
    @jax.jit
    def step(i, batch, params, opt_state):
        l, g = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return l, params
    
    rng, rng1 = jax.random.split(rng)
    data_generator = batch_iter(rng1, X, batch_size=batch_size, steps=steps)
    
    opt_state = optimizer.init(params)

    tqdm_iter = tqdm(enumerate(data_generator), total=steps)
    try:
        for i, batch in tqdm_iter:
            l, params = step(i, batch, params, opt_state)
            if jnp.any(jnp.isnan(l)) or jnp.any(jnp.isinf(l)):
                print("Run diverged")
                return params
            tqdm_iter.set_postfix({"loss": l})
    except KeyboardInterrupt as e:
        pass
    return params


if __name__ == '__main__':
    from fox.nn import MLP
    from fox.core import NormalizingFlowDist, NormalizingFlow
    from fox.bijections.simple import RealNVP, InvertibleMM, Sigmoid
    from fox.distributions import StandardGaussian
    from matplotlib import pyplot as plt

    rng = jax.random.PRNGKey(0)

    # create model
    flow = NormalizingFlow([
        RealNVP(MLP([512,512], 2), False),
        InvertibleMM(),
        RealNVP(MLP([512,512], 2), True),
        InvertibleMM(),
        RealNVP(MLP([512,512], 2), False),
        ])
    prior = StandardGaussian(2) 
    flow_dist = NormalizingFlowDist(prior, flow)
    params = flow_dist.init(rng, rng, 2, method=flow_dist.sample)

    # create dataset 
    X = load_moons(1000)

    # create train loop
    loss = functools.partial(nll_loss, flow_dist=flow_dist)
    opt_params = train(rng, params, loss, X, lr=1e-4, steps=10000, batch_size=512)
    
    # sample from learned flow
    from  fox.utils import plot_samples_2d
    samples = flow_dist.apply(opt_params, rng, 1000, method=flow_dist.sample)
    samples_initial = flow_dist.apply(params, rng, 1000, method=flow_dist.sample)

    fig, ax = plot_samples_2d(samples, X, samples_initial)

    print("Showing figure...")
    plt.show()