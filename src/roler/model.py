from roler.distributions import *

from dataclasses import dataclass, fields
from torch.distributions import Uniform
import torch

@dataclass
class ModelParams:
    individuals_local: int
    individuals_meta: int
    species_meta: int
    speciation_local: float
    speciation_meta: float
    extinction_meta: float
    env_sigma: float
    trait_sigma: float
    comp_sigma: float
    dispersal_prob: float
    mutation_rate: float
    equilib_escape: float
    num_basepairs: int
    init_type: str
    niter: int
    niterTimestep: int

@dataclass
class ModelPrior:
    individuals_local: int | IntDistribution
    individuals_meta: int | IntDistribution
    species_meta: int | IntDistribution
    speciation_local: float | FloatDistribution
    speciation_meta: float | FloatDistribution
    extinction_meta: float | FloatDistribution
    env_sigma: float | FloatDistribution
    trait_sigma: float | FloatDistribution
    comp_sigma: float | FloatDistribution
    dispersal_prob: float | FloatDistribution
    mutation_rate: float | FloatDistribution
    equilib_escape: float | FloatDistribution
    num_basepairs: int | IntDistribution
    init_type: str | ChoiceDistribution
    niter: int | IntDistribution
    niterTimestep: int | IntDistribution
    
    def sample(self) -> ModelParams:
        prior = self.get_joint_uniform()
        theta = prior.sample()
        return self.get_params_from_tensor(theta)
    
    def get_joint_uniform(self) -> Uniform:
        low = []
        high = []
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, Distribution):
                low.append(val.low)
                high.append(val.high)
        return Uniform(low=torch.tensor(low), high=torch.tensor(high))
    
    def get_params_from_tensor(self, sample: torch.Tensor) -> ModelParams:
        sampled_params = {}
        sample_index = 0
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if isinstance(value, Distribution):
                sampled_value = sample[sample_index].item()
                sampled_params[field_obj.name] = value.discretize(sampled_value)
                sample_index += 1
            else:
                sampled_params[field_obj.name] = value
        return ModelParams(**sampled_params)