from sbi.inference.posteriors.base_posterior import NeuralPosterior
from roler.datasets import DatasetGenerator
from roler.model import ModelPrior, ModelParams
from roler.datasets import DatasetGenerator, Dataset
from roler.simulation import Simulator
import torch
from sbi.inference import SNPE
from typing import TypedDict

class Observation(TypedDict):
    theta_true: torch.Tensor
    x_obs: torch.Tensor

class RoleR:
    def __init__(self, prior: ModelPrior, load_fp: str = ""):
        self.prior = prior
        self.simulator = Simulator(prior)
        self.generator = DatasetGenerator(self.simulator)
        self.posterior = None
        if load_fp:
            try:
                posterior = torch.load(load_fp, weights_only=False)
                self.posterior = posterior
            except Exception as e:
                print(f"Failed to load posterior: {e}")

    def train(self, samples: int, n_jobs: int = 1, save_fp: str = "") -> NeuralPosterior:
        dataset = self.generator.generate_dataset(samples, n_jobs)
        x, y = RoleR.to_tensor(dataset)
        
        snpe = SNPE(prior=self.prior)
        density_estimator = snpe.append_simulations(x, y).train()
        self.posterior = snpe.build_posterior(density_estimator)
        
        if save_fp:
            try:
                torch.save(self.posterior, save_fp)
            except Exception as e:
                print(f"Failed to save posterior: {e}")
        return self.posterior
    
    def observe(self) -> Observation:
        theta_true = self.prior.sample()
        x_obs = self.simulator(theta_true)
        x_obs_tensor = torch.tensor(x_obs.values, dtype=torch.float32)
        
        return {
            "theta_true": theta_true,
            "x_obs": x_obs_tensor[-1]
        }
    
    def infer(self, x_obs: torch.Tensor) -> torch.Tensor:
        if not self.posterior:
            print("You must train first or specify posterior path.")
            return
        
        posterior_samples = self.posterior.sample((10000,), x=x_obs)
        posterior_mean = posterior_samples.mean(dim=0)
        
        return posterior_mean

    def get_params(self, sample: torch.Tensor) -> ModelParams:
        return self.prior.get_params(sample)
    
    @staticmethod
    def to_tensor(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        params = dataset["params"]
        output = dataset["output"]
        x = torch.tensor(params.to_numpy(), dtype=torch.float32)
        y = torch.tensor(output.to_numpy(), dtype=torch.float32)
        mask = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
        return x[mask], y[mask]