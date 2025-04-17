import torch
from sbi.inference import SNPE
from torch.distributions import Distribution
from sbi.inference.posteriors.base_posterior import NeuralPosterior

def train_posterior(prior: Distribution, x: torch.Tensor, y: torch.Tensor) -> NeuralPosterior:
    snpe = SNPE(prior=prior)
    density_estimator = snpe.append_simulations(x, y).train()
    posterior = snpe.build_posterior(density_estimator)
    return posterior