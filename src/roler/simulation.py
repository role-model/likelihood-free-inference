from rpy2.robjects import pandas2ri

from rpy2.robjects.packages import importr
import pandas as pd
import torch
from roler.model import ModelPrior, ModelParams

DEFAULT_COLUMNS = [
    "richness",
    "hill_abund_1",
    "hill_abund_2",
    "hill_abund_3",
    "hill_abund_4",
    "hill_trait_1",
    "hill_trait_2",
    "hill_trait_3",
    "hill_trait_4",
]

roleR = importr('roleR')

class Simulator:
    def __init__(self, prior: ModelPrior = None):
        self.prior = prior
    
    def simulate(self, theta: ModelParams) -> pd.DataFrame:
        pandas2ri.activate()
        
        params = roleR.roleParams(**theta.model_dump())
        model = roleR.runRole(roleR.roleModel(params))
        stats = roleR.getSumStats(model)

        stats = pandas2ri.rpy2py(stats)

        stats = stats.iloc[1:].reset_index(drop=True)
        stats = stats[DEFAULT_COLUMNS]

        return stats
    
    def simulate_one(self) -> pd.DataFrame:
        params = self.prior.sample()
        return self.simulate(self.prior.get_params(params))
    
    def __call__(self, theta: torch.Tensor) -> pd.DataFrame:        
        params = self.prior.get_params(theta)
        return self.simulate(params)