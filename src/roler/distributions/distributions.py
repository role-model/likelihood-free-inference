from roler.distributions.base import Distribution, NumberDistribution
from typing import Any, List
import random
import numpy as np

class ConstantDistribution(Distribution):
    """
    Distribution that always returns a constant value.
    """

    def __init__(self, value: Any):
        self.value = value

    def sample(self) -> Any:
        return self.value

class UniformDistribution(NumberDistribution):
    """
    Uniform distribution over a given range.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return round(random.uniform(self.low, self.high), ndigits=2)

class GaussianDistribution(NumberDistribution):
    """
    Gaussian (normal) distribution with given mean and standard deviation.
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return np.random.normal(self.mean, self.std)

class ChoiceDistribution(Distribution):
    """
    Distribution that randomly chooses from a list of objects.
    """

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def sample(self) -> Any:
        return random.choice(self.choices)

class IntegerDistribution(NumberDistribution):
    def __init__(self, dist: NumberDistribution):
        self.dist = dist
    
    def sample(self) -> float:
        return round(self.dist.sample())