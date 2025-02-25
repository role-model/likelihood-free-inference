from typing import TypedDict
from roler.distributions.base import Distribution

class ModelParams(TypedDict):
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
    equilib_escape: int
    num_basepairs: int
    init_type: str
    niter: int
    niterTimestep: int

class ModelPrior(TypedDict):
    """
    Represents a prior distribution for model parameters.
    """
    individuals_local: Distribution
    individuals_meta: Distribution
    species_meta: Distribution
    speciation_local: Distribution
    speciation_meta: Distribution
    extinction_meta: Distribution
    env_sigma: Distribution
    trait_sigma: Distribution
    comp_sigma: Distribution
    dispersal_prob: Distribution
    mutation_rate: Distribution
    equilib_escape: Distribution
    num_basepairs: Distribution
    init_type: Distribution
    niter: Distribution
    niterTimestep: Distribution