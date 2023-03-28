from fox.nn import MLP
from fox.core import NormalizingFlowDist, NormalizingFlow
from fox.bijections.simple import RealNVP
from fox.distributions import StandardGaussian
from matplotlib import pyplot as plt
from fox.train import load_moons, nll_loss, train
import functools
import jax

rng = jax.random.PRNGKey(0)

def get_mlp():
    f = False
    while True:
        yield RealNVP(MLP([256,256, 64], 2, activation=jax.nn.tanh), f)
        f = not f

mlp_gen = get_mlp()
print(next(mlp_gen))
# create model
flow = NormalizingFlow([ next(mlp_gen) for _ in range(5)])
prior = StandardGaussian(2) 
flow_dist = NormalizingFlowDist(prior, flow)
params = flow_dist.init(rng, rng, 2, method=flow_dist.sample)

# create dataset 
X = load_moons(4000)

# create train loop
loss = functools.partial(nll_loss, flow_dist=flow_dist)
opt_params = train(rng, params, loss, X, lr=1e-3, steps=10000, batch_size=256)

# sample from learned flow
from  fox.utils import plot_samples_2d
samples = flow_dist.apply(opt_params, rng, 1000, method=flow_dist.sample)
samples_initial = flow_dist.apply(params, rng, 1000, method=flow_dist.sample)

fig, ax = plot_samples_2d(samples, X, samples_initial)

print("Showing figure...")
plt.show()
plt.savefig("real_nvp.png", dpi=300, bbox_inches='tight')