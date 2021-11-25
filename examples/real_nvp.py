from fox.nn import MLP
from fox.core import NormalizingFlowDist, NormalizingFlow
from fox.bijections.simple import RealNVP
from fox.distributions import StandardGaussian
from matplotlib import pyplot as plt
from fox.train import load_moons, nll_loss, train
import functools
import jax

rng = jax.random.PRNGKey(0)

# create model
flow = NormalizingFlow([
    RealNVP(MLP([64,64, 64], 2), False),
    RealNVP(MLP([64,64, 64], 2), True),
    RealNVP(MLP([64,64, 64], 2), False),
    RealNVP(MLP([64,64, 64], 2), True)
    ])
prior = StandardGaussian(2) 
flow_dist = NormalizingFlowDist(prior, flow)
params = flow_dist.init(rng, rng, 2, method=flow_dist.sample)

# create dataset 
X = load_moons(4000)

# create train loop
loss = functools.partial(nll_loss, flow_dist=flow_dist)
opt_params = train(rng, params, loss, X, lr=5e-4, steps=10000, batch_size=256)

# sample from learned flow
from  fox.utils import plot_samples_2d
samples = flow_dist.apply(opt_params, rng, 1000, method=flow_dist.sample)

fig, ax = plot_samples_2d(samples, X)

print("Showing figure...")
plt.show()