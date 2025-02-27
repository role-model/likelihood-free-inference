import torch
from roler.model import ModelPrior, ModelParams
from roler.simulation import Simulator
import numpy as np

DEFAULT_COLUMNS = ["richness", "hill_abund_1", "hill_abund_2", "hill_abund_3", "hill_abund_4", "hill_trait_1", "hill_trait_2", "hill_trait_3", "hill_trait_4"]

class Dataset:
    def __init__(self, simulator: Simulator, prior: ModelPrior, columns: list[str] = DEFAULT_COLUMNS):
        self.prior = prior
        self.simulator = simulator
        self.columns = columns
    
    def generate_dataset(self, samples: int, select_ratio: float = 0.5):
        tensor_prior = self.prior.get_joint_uniform()
        
        x_samples_transformed = torch.tensor([])
        theta_samples_transformed = torch.tensor([])
        
        for _ in range(samples):
            theta = tensor_prior.sample()
            params = self.prior.get_params_from_tensor(theta)
            
            stats_df = self.simulator.simulate(params)
            stats_df = stats_df[[col for col in stats_df.columns if col in self.columns]]
            stats_df = stats_df.dropna()
            
            x = torch.Tensor(np.array(stats_df))
            x_samples_transformed = torch.cat((x_samples_transformed, x))
            theta_samples_transformed = torch.cat((theta_samples_transformed, torch.tile(theta, (x.shape[0], 1))))

        return [theta_samples_transformed, x_samples_transformed]
    
    def get_params_from_tensor(self, sample: torch.Tensor) -> ModelParams:
        return self.prior.get_params_from_tensor(sample)