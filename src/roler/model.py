from dataclasses import dataclass, fields
from roler.distributions import *
import torch
from typing import Union
from collections import OrderedDict
from sbi.utils import BoxUniform
from pydantic import BaseModel
from typing import TypedDict    

class ModelParams(BaseModel):
    individuals_local: int = 10000
    individuals_meta: int = 1000000
    species_meta: int = 500
    speciation_local: float = 0.00001
    speciation_meta: float = 1.0
    extinction_meta: float = 0.8
    env_sigma: float = 0.0
    trait_sigma: float = 1.0
    comp_sigma: float = 0
    dispersal_prob: float = 0.01
    mutation_rate: float = 0.0
    equilib_escape: float = 0.0
    num_basepairs: int = 500
    
    alpha: float = 1.0
    neut_delta: float = 1.0
    env_comp_delta: float = 0.5
    
    init_type: str = "oceanic_island"
    niter: int = 500000
    niterTimestep: int = 500000

@dataclass
class ModelPrior(BoxUniform):
    individuals_local: Union[int, "IntDistribution"] = 10000
    add_individuals_meta: Union[int, "IntDistribution"] = 100000
    
    species_meta: Union[int, "IntDistribution"] = 500
    speciation_local: Union[float, "FloatDistribution"] = 0.00001
    speciation_meta: Union[float, "FloatDistribution"] = 1.0
    
    extinction_meta: Union[float, "FloatDistribution"] = 0.8
    env_sigma: Union[float, "FloatDistribution"] = 0.0
    trait_sigma: Union[float, "FloatDistribution"] = 1.0
    comp_sigma: Union[float, "FloatDistribution"] = 0
    dispersal_prob: Union[float, "FloatDistribution"] = 0.01
    mutation_rate: Union[float, "FloatDistribution"] = 0.0
    equilib_escape: Union[float, "FloatDistribution"] = 0.0
    num_basepairs: Union[int, "IntDistribution"] = 500
    
    alpha: Union[float, "FloatDistribution"] = 1.0  # New parameter
    neut_delta: Union[float, "FloatDistribution"] = 1.0  # New parameter
    env_comp_delta: Union[float, "FloatDistribution"] = 0.5  # New parameter
    
    init_type: Union[str, "ChoiceDistribution"] = "oceanic_island"
    niter: Union[int, "IntDistribution"] = 500000
    niterTimestep: Union[int, "IntDistribution"] = 500000
    
    def __post_init__(self):
        self.low, self.high = self._get_low_high_tensors()
        super().__init__(low=self.low, high=self.high)

    def get_uniform(self, device: str) -> BoxUniform:
        low = self.low.clone().to(device)
        high = self.high.clone().to(device)

        return BoxUniform(low=low, high=high)
    
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
        # params["species_meta"] = math.ceil(params["individuals_meta"] * params["prop_species_meta"])
        # del params["prop_species_meta"]
        
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