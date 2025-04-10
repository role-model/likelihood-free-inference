from dataclasses import dataclass, fields
from roler.distributions import *
import torch
from typing import Union
from collections import OrderedDict
from sbi.utils import BoxUniform
import math
# from pydantic.dataclasses import dataclass
from pydantic import BaseModel

class ModelParams(BaseModel):
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
    
    alpha: float  # New parameter
    neut_delta: float  # New parameter
    env_comp_delta: float  # New parameter
    
    init_type: str
    niter: int
    niterTimestep: int

@dataclass
class ModelPrior(BoxUniform):
    individuals_local: Union[int, "IntDistribution"]
    add_individuals_meta: Union[int, "IntDistribution"]
    
    prop_species_meta: Union[float, "FloatDistribution"]
    speciation_local: Union[float, "FloatDistribution"]
    speciation_meta: Union[float, "FloatDistribution"]
    
    extinction_meta: Union[float, "FloatDistribution"]
    env_sigma: Union[float, "FloatDistribution"]
    trait_sigma: Union[float, "FloatDistribution"]
    comp_sigma: Union[float, "FloatDistribution"]
    dispersal_prob: Union[float, "FloatDistribution"]
    mutation_rate: Union[float, "FloatDistribution"]
    equilib_escape: Union[float, "FloatDistribution"]
    num_basepairs: Union[int, "IntDistribution"]
    
    alpha: Union[float, "FloatDistribution"]  # New parameter
    neut_delta: Union[float, "FloatDistribution"]  # New parameter
    env_comp_delta: Union[float, "FloatDistribution"]  # New parameter
    
    init_type: Union[str, "ChoiceDistribution"]
    niter: Union[int, "IntDistribution"]
    niterTimestep: Union[int, "IntDistribution"]
    
    def __post_init__(self):
        low, high = self._get_low_high_tensors()
        super().__init__(low=low, high=high)

    def get_params(self, sample: torch.Tensor) -> ModelParams:
        sample = sample.tolist()
        params = OrderedDict()
        
        index = 0
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if isinstance(value, Distribution):
                params[field_obj.name] = value.discretize(sample[index])
                index += 1
            else:
                params[field_obj.name] = value
                
        params["individuals_meta"] = params["individuals_local"] + params["add_individuals_meta"]
        del params["add_individuals_meta"]
        params["species_meta"] = math.ceil(params["individuals_meta"] * params["prop_species_meta"])
        del params["prop_species_meta"]
        
        return ModelParams(**params)
    
    def _get_low_high_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = []
        high = []
        
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if isinstance(value, Distribution):
                low.append(value.low)
                high.append(value.high)

        return torch.tensor(low), torch.tensor(high)


class SimpleModelParams(BaseModel):
    birth_rate: float
    ntaxa: int
    J: int
    colrate: float
    # local_stop_time: int
    Ne_scaling: float
    
@dataclass
class SimpleModelPrior(BoxUniform):
    birth_rate: Union[float, "FloatDistribution"]
    ntaxa: Union[int, "IntDistribution"]
    J: Union[int, "IntDistribution"]
    colrate: Union[float, "FloatDistribution"]
    # local_stop_time: Union[int, "IntDistribution"]
    Ne_scaling: Union[float, "FloatDistribution"]

    def __post_init__(self):
        """Initializes the BoxUniform parent class with low/high bounds."""
        low, high = self._get_low_high_tensors()
        # Ensure tensors are created correctly even if there are no distributions
        if not low and not high:
             # Handle case where all parameters are fixed (no distributions)
             # BoxUniform needs non-empty tensors, create dummy ones
             # Or raise an error, depending on expected SBI behavior for fixed params
             # For now, creating dummy tensors of shape (0,)
             low_tensor = torch.empty((0,), dtype=torch.float32)
             high_tensor = torch.empty((0,), dtype=torch.float32)
             # Note: SBI might require at least one parameter to vary.
             # If using SBI, ensure at least one field uses a Distribution.
        else:
            low_tensor = torch.tensor(low, dtype=torch.float32)
            high_tensor = torch.tensor(high, dtype=torch.float32)

        super().__init__(low=low_tensor, high=high_tensor)

    def get_params(self, sample: torch.Tensor) -> SimpleModelParams:
        """
        Generates a SimpleModelParams instance from a sampled tensor.

        Args:
            sample: A tensor sampled from the prior distribution (e.g., by SBI).
                    Expected shape is (num_varying_params,).

        Returns:
            An instance of SimpleModelParams with values filled according
            to the sample and the prior definitions.
        """
        sample_list = sample.tolist()
        params = OrderedDict()
        param_index = 0

        for field_obj in fields(self):
            field_name = field_obj.name
            prior_value = getattr(self, field_name)

            if isinstance(prior_value, Distribution):
                # Check if we have enough sample values
                if param_index >= len(sample_list):
                    raise IndexError(
                        f"Sample tensor has too few elements. Trying to access index {param_index} "
                        f"for parameter '{field_name}', but sample length is {len(sample_list)}."
                    )
                # Get value from sample and discretize using the distribution
                params[field_name] = prior_value.discretize(sample_list[param_index])
                param_index += 1
            else:
                # Use the fixed value provided in the prior definition
                params[field_name] = prior_value

        # Ensure all sampled values were used (optional check)
        if param_index != len(sample_list):
             print(f"Warning: Sample tensor length ({len(sample_list)}) does not match "
                   f"the number of varying parameters ({param_index}). Extra sample values ignored.")

        # Create and return the SimpleModelParams instance
        return SimpleModelParams(**params)

    def _get_low_high_tensors(self) -> tuple[list, list]:
        """Extracts low and high bounds from Distribution fields."""
        low = []
        high = []

        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            # Check specifically for types inheriting from Distribution
            # or adjust based on your actual Distribution base class/types
            if isinstance(value, (IntDistribution, FloatDistribution)): # Adapt if using a common base class 'Distribution'
                low.append(value.low)
                high.append(value.high)

        return low, high
