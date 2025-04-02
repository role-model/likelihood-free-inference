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