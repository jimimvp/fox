
from jax import numpy as jnp
from jax.random import normal

class MetropolisHastings:
    """
    Sampling through Metropolis Hastings in Stochastic Normalizing
    Flow, see arXiv: 2002.06707
    """
    def __init__(self, dist, proposal, steps):
        """
        Constructor
        :param dist: Distribution to sample from
        :param proposal: Proposal distribution
        :param steps: Number of MCMC steps to perform
        """
        super().__init__()
        self.dist = dist
        self.proposal = proposal
        self.steps = steps

    def forward(self, z):
        # Initialize number of samples and log(det)
        num_samples = len(z)
        log_det = jnp.zeros(num_samples, dtype=z.dtype, device=z.device)
        # Get log(p) for current samples
        log_p = self.dist.log_prob(z)
        for i in range(self.steps):
            # Make proposal and get log(p)
            z_, log_p_diff = self.proposal(z)
            log_p_ = self.dist.log_prob(z_)
            # Make acceptance decision
            w = normal(num_samples, dtype=z.dtype)
            log_w_accept = log_p_ - log_p + log_p_diff
            w_accept = jnp.clip(jnp.exp(log_w_accept), max=1)
            accept = w <= w_accept
            # Update samples, log(det), and log(p)
            z = jnp.where(accept.unsqueeze(1), z_, z)
            log_det_ = log_p - log_p_
            log_det = jnp.where(accept, log_det + log_det_, log_det)
            log_p = jnp.where(accept, log_p_, log_p)
        return z, log_det

    def backward(self, z):
        # Equivalent to forward pass
        return self.forward(z)

from jax.random import random
class HamiltonianMonteCarlo:
    """
    Flow layer using the HMC proposal in Stochastic Normalising Flows,
    see arXiv: 2002.06707
    """
    def __init__(self, target, steps, log_step_size, log_mass):
        """
        Constructor
        :param target: The stationary distribution of this Markov transition. Should be logp
        :param steps: The number of leapfrog steps
        :param log_step_size: The log step size used in the leapfrog integrator. shape (dim)
        :param log_mass: The log_mass determining the variance of the momentum samples. shape (dim)
        """
        super().__init__()
        self.target = target
        self.steps = steps
        self.register_parameter('log_step_size', torch.nn.Parameter(log_step_size))
        self.register_parameter('log_mass', torch.nn.Parameter(log_mass))

    def forward(self, z):
        # Draw momentum
        p = torch.randn_like(z) * jnp.exp(0.5 * self.log_mass)

        # leapfrog
        z_new = z.clone()
        p_new = p.clone()
        step_size = torch.exp(self.log_step_size)
        for i in range(self.steps):
            p_half = p_new - (step_size/2.0) * -self.gradlogP(z_new)
            z_new = z_new + step_size * (p_half/torch.exp(self.log_mass))
            p_new = p_half - (step_size/2.0) * -self.gradlogP(z_new)

        # Metropolis Hastings correction
        probabilities = jnp.exp(
            self.target.log_prob(z_new) - self.target.log_prob(z) - \
            0.5 * jnp.sum(p_new ** 2 / jnp.exp(self.log_mass), 1) + \
            0.5 * jnp.sum(p ** 2 / jnp.exp(self.log_mass), 1))
        uniforms = jnp.rand_like(probabilities)
        mask = uniforms < probabilities
        z_out = jnp.where(mask.unsqueeze(1), z_new, z)

        return z_out, self.target.log_prob(z) - self.target.log_prob(z_out)

    def backward(self, z):
        return self.forward(z)

    def gradlogP(self, z):
        z_ = z.detach().requires_grad_()
        logp = self.target.log_prob(z_)
        return torch.autograd.grad(logp, z_,
            grad_outputs=torch.ones_like(logp))[0]