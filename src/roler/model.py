import subprocess
import warnings

# Import necessary modules
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
except ImportError as e:
    raise ImportError(
        "The 'rpy2' library is required but not installed. "
        "Install it with 'pip install rpy2'."
    ) from e

# Activate pandas conversion for rpy2
pandas2ri.activate()

# Function to check if R is installed
def check_r_installed():
    try:
        result = subprocess.run(
            ["R", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise EnvironmentError("R is not installed or not available in PATH.")
    except FileNotFoundError:
        raise EnvironmentError("R is not installed or not available in PATH.")
    except Exception as e:
        raise EnvironmentError(f"An error occurred while checking for R: {e}")

# Check if R is installed
check_r_installed()

# Install R packages if they are not already installed
try:
    remotes = importr('remotes')
    print("Installing the 'roleR' R package from GitHub...")
    remotes.install_github("role-model/roleR", dependencies=True)
except Exception as e:
    warnings.warn(
        f"Error installing R packages: {e}\n"
        "Make sure you have R and the 'remotes' package installed correctly.",
        RuntimeWarning
    )

# Import the R package
try:
    roleR = importr('roleR')
except Exception as e:
    raise ImportError(
        f"Error importing the 'roleR' R package: {e}\n"
        "Ensure the package is installed and available in your R environment."
    ) from e


from abc import ABC, abstractmethod
from typing import Any, List

class Distribution(ABC):
    def __init__(self, low: float, high: float):
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError("low and high must be numeric (int or float)")
        if low >= high:
            raise ValueError("low must be less than high")

        self.low = float(low)
        self.high = float(high)

    @abstractmethod
    def discretize(self, n: float) -> Any:
        return n


class FloatDistribution(Distribution):
    def __init__(self, low: float, high: float):
        super().__init__(low, high)

    def discretize(self, n: float) -> float:
        return n


class IntDistribution(Distribution):
    def __init__(self, low: int, high: int):
        super().__init__(float(low), float(high))  # Call super with floats

    def discretize(self, n: float) -> int:
        return round(n)


class ChoiceDistribution(Distribution):
    def __init__(self, choices: List[Any]):
        self.choices = choices
        super().__init__(0, len(choices) - 1)

    def discretize(self, n: float) -> Any:
        return self.choices[round(n)]

# human readable prior
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
    niter: int
    niterTimestep: int
    
    def get_joint_uniform(self) -> Uniform:
        low = []
        high = []
        for field in fields(self):
            val = getattr(self, field.name)
            if isinstance(val, Distribution):
                low.append(val.low)
                high.append(val.high)
        
        return Uniform(low=torch.tensor(low), high=torch.tensor(high))
    
    def get_params_from_sample(self, sample: torch.Tensor) -> ModelParams:
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

from roler.types import ModelOutput

def run_simulation(params: ModelParams) -> ModelOutput:    
    p = roleR.roleParams(**params)

    model = roleR.runRole(roleR.roleModel(p))
    stats = roleR.getSumStats(model)
    stats_df = pandas2ri.rpy2py(stats)
    
    return ModelOutput(
        hill_abund_1=stats_df["hill_abund_1"].tolist(),
        hill_abund_2=stats_df["hill_abund_2"].tolist(),
        hill_abund_3=stats_df["hill_abund_3"].tolist(),
        hill_abund_4=stats_df["hill_abund_4"].tolist(),
        
        hill_trait_1=stats_df["hill_trait_1"].tolist(),
        hill_trait_2=stats_df["hill_trait_2"].tolist(),
        hill_trait_3=stats_df["hill_trait_3"].tolist(),
        hill_trait_4=stats_df["hill_trait_4"].tolist(),
        
        richness=stats_df["richness"].tolist(),
    )