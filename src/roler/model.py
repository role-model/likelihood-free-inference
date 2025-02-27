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
    
    def is_valid(self, params: ModelParams) -> bool:
        """Check if the parameters satisfy all biological/logical constraints."""
        
        # Population Hierarchies
        if params.individuals_meta < params.individuals_local:
            return False
        
        if params.species_meta > params.individuals_meta:
            return False
        
        # Rate Relationships
        if params.speciation_meta + params.extinction_meta >= 1:
            return False
        
        if params.mutation_rate >= params.speciation_local or params.mutation_rate >= params.speciation_meta:
            return False
        
        # Scale Dependencies
        # Assuming trait_sigma and comp_sigma should be within an order of magnitude of each other
        if params.trait_sigma / params.comp_sigma > 10 or params.comp_sigma / params.trait_sigma > 10:
            return False
        
        # Computational Feasibility
        MAX_INDIVIDUALS = 10000  # Example upper bound
        MAX_SPECIES = 1000       # Example upper bound
        MAX_BASEPAIRS = 10000    # Example upper bound
        
        if params.individuals_meta > MAX_INDIVIDUALS or params.species_meta > MAX_SPECIES or params.num_basepairs > MAX_BASEPAIRS:
            return False
        
        # Ensure simulation has adequate time steps but isn't excessive
        MIN_TOTAL_ITERATIONS = 1000
        MAX_TOTAL_ITERATIONS = 1000000
        total_iterations = params.niter * params.niterTimestep
        
        if total_iterations < MIN_TOTAL_ITERATIONS or total_iterations > MAX_TOTAL_ITERATIONS:
            return False
        
        # Genetic Drift Considerations
        # Ensure population size is large enough compared to mutation rate to avoid excessive drift
        MIN_GENETIC_DIVERSITY = 1.0  # Example threshold
        if params.individuals_local * params.mutation_rate < MIN_GENETIC_DIVERSITY:
            return False
        
        # All constraints passed
        return True

    
    def sample_valid(self) -> ModelParams:
        """Sample from the prior distribution until a valid parameter set is found."""
        max_attempts = 10000  # Safety limit to prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            params = self.sample()
            if self.is_valid(params):
                return params
            attempt += 1
        
        # If we reach here, we couldn't find valid parameters
        raise ValueError(
            f"Could not find valid parameters after {max_attempts} attempts. "
            "Prior distributions may need adjustment to better respect constraints."
        )
    
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